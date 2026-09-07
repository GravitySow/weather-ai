"""Read-only Home Assistant sensor ingestion for the Weather AI pipeline.

The add-on can read upstream HA entities through the Supervisor Core API and
feed normalized observations into the same pipeline used by ``POST /reading``.
The feature is deliberately opt-in (``HA_SOURCE_ENABLED=0`` by default), and
never reads entities published by this service itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import requests

logger = logging.getLogger(__name__)

_state_lock = threading.Lock()
_last_snapshot_key: str | None = None
_status: dict[str, Any] = {
    "status": "disabled",
    "reason": "source_disabled",
    "last_success_at_utc": None,
    "last_observation_at_utc": None,
    "last_ingest_at_utc": None,
    "ingest_count": 0,
    "last_prediction_ready": None,
    "last_prediction_reason": None,
    "last_prediction_success_at_utc": None,
    "prediction_count": 0,
    "last_error_at_utc": None,
    "last_error": None,
    "error_count": 0,
    "duplicate_count": 0,
}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(float(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def _entity(name: str) -> str:
    return os.getenv(name, "").strip().lower()


def _entity_valid(entity_id: str) -> bool:
    """Validate a state entity and reject this service's own output loop."""
    if not entity_id or "." not in entity_id:
        return False
    domain, object_id = entity_id.split(".", 1)
    if not domain or not object_id or any(ch.isspace() for ch in entity_id):
        return False
    if domain not in {"sensor", "binary_sensor", "input_number", "number", "weather"}:
        return False
    # Never ingest entities emitted by ha_publisher.py. Besides avoiding a
    # feedback loop this makes a typo in configuration fail closed.
    if domain in {"sensor", "binary_sensor", "weather"} and object_id.startswith("weather_ai"):
        return False
    return True


def _config() -> dict[str, Any]:
    return {
        "enabled": _truthy(os.getenv("HA_SOURCE_ENABLED", "0")),
        "temp_entity": _entity("HA_TEMP_ENTITY"),
        "humidity_entity": _entity("HA_HUMIDITY_ENTITY"),
        "pressure_entity": _entity("HA_PRESSURE_ENTITY"),
        "rain_entity": _entity("HA_RAIN_ENTITY"),
        "light_entity": _entity("HA_LIGHT_ENTITY"),
        "poll_seconds": _int_env("HA_POLL_SECONDS", 60),
        "stale_after_seconds": _int_env("HA_STALE_AFTER_SECONDS", 180),
        "api_base": (os.getenv("HA_API_BASE") or "http://supervisor/core/api").rstrip("/"),
        # HA_TOKEN is intended for an external Docker deployment. Add-ons get
        # SUPERVISOR_TOKEN from the Supervisor automatically.
        "token": os.getenv("HA_TOKEN") or os.getenv("SUPERVISOR_TOKEN", ""),
    }


def _mask_entity(entity_id: str) -> str | None:
    if not entity_id:
        return None
    domain, _, object_id = entity_id.partition(".")
    suffix = object_id[-6:] if object_id else ""
    return f"{domain}.***{suffix}"


def _safe_api_base(value: str) -> str:
    """Remove query, fragment, and URL userinfo before exposing API metadata."""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "[configured]"


def _configured(config: dict[str, Any]) -> tuple[bool, str | None]:
    if not config["enabled"]:
        return False, "source_disabled"
    for key in ("temp_entity", "humidity_entity", "pressure_entity"):
        if not _entity_valid(config[key]):
            return False, f"invalid_or_missing_{key}"
    for key in ("rain_entity", "light_entity"):
        if config[key] and not _entity_valid(config[key]):
            return False, f"invalid_{key}"
    if not config["token"]:
        return False, "token_missing"
    return True, None


def _set_status(**updates: Any) -> None:
    with _state_lock:
        _status.update(updates)


def _now_iso(value: datetime | None = None) -> str:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "unknown", "unavailable", "none", "null"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _temperature(state: dict[str, Any]) -> float | None:
    value = _number(state.get("state"))
    if value is None:
        return None
    unit = str((state.get("attributes") or {}).get("unit_of_measurement") or "°C").strip().lower()
    if unit in {"°f", "f", "ºf"}:
        value = (value - 32.0) * 5.0 / 9.0
    elif unit in {"k", "°k", "ºk"}:
        value -= 273.15
    elif unit not in {"°c", "c", "ºc", ""}:
        return None
    return value if -100.0 <= value <= 80.0 else None


def _humidity(state: dict[str, Any]) -> float | None:
    value = _number(state.get("state"))
    if value is None:
        return None
    unit = str((state.get("attributes") or {}).get("unit_of_measurement") or "%").strip().lower()
    if unit in {"ratio", "fraction", "1"} and 0.0 <= value <= 1.0:
        value *= 100.0
    if unit not in {"%", "percent", "percentage", "ratio", "fraction", "1", ""}:
        return None
    return value if 0.0 <= value <= 100.0 else None


def _pressure(state: dict[str, Any]) -> float | None:
    value = _number(state.get("state"))
    if value is None:
        return None
    unit = str((state.get("attributes") or {}).get("unit_of_measurement") or "hPa").strip().lower()
    if unit in {"pa", "pascal", "pascals"}:
        value /= 100.0
    elif unit in {"kpa", "kilopascal", "kilopascals"}:
        value *= 10.0
    elif unit in {"inhg", "in hg", "in"}:
        value *= 33.8638866667
    elif unit not in {"hpa", "mbar", "mb", "hectopascal", "hectopascals", ""}:
        return None
    return value if 700.0 <= value <= 1100.0 else None


def _rain(state: dict[str, Any]) -> tuple[bool | None, float | None]:
    raw = state.get("state")
    number = _number(raw)
    if number is not None:
        return number > 0.1, number
    token = str(raw or "").strip().lower()
    if token in {"on", "true", "yes", "open", "wet", "raining", "rain"}:
        return True, 1.0
    if token in {"off", "false", "no", "closed", "dry", "clear", "not raining"}:
        return False, 0.0
    return None, None


def _rain_is_binary_state(entity_id: str, state: dict[str, Any]) -> bool:
    """Binary sensors are stateful, not heartbeat sensors.

    Home Assistant normally updates ``last_updated`` only when a binary
    sensor changes state or attributes. A healthy contact that stays ``off``
    for hours is therefore not stale merely because that timestamp is old.
    Numeric rain-rate sensors still require a fresh timestamp below.
    """
    if entity_id.split(".", 1)[0] == "binary_sensor":
        return True
    token = str(state.get("state") or "").strip().lower()
    return token in {
        "on", "off", "true", "false", "yes", "no", "open", "closed",
        "wet", "dry", "raining", "rain", "clear", "not raining",
    }


def _light(state: dict[str, Any]) -> float | str | None:
    number = _number(state.get("state"))
    if number is not None:
        return number
    raw = state.get("state")
    if raw is None or str(raw).strip().lower() in {"", "unknown", "unavailable", "none", "null"}:
        return None
    return str(raw)


def _state_age(state: dict[str, Any], now: datetime) -> tuple[datetime | None, float | None]:
    observed = _parse_timestamp(state.get("last_updated") or state.get("last_changed"))
    if observed is None:
        return None, None
    return observed, max(0.0, (now - observed).total_seconds())


def _fetch_entity(entity_id: str, config: dict[str, Any]) -> dict[str, Any]:
    url = f"{config['api_base']}/states/{entity_id}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {config['token']}", "Content-Type": "application/json"},
        timeout=10,
    )
    if getattr(response, "status_code", None) == 429:
        raise RuntimeError("http_429")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("invalid_state_response")
    return payload


def fetch_reading(now: datetime | None = None) -> dict[str, Any]:
    """Fetch and normalize one HA observation.

    Returns ``status=ready`` with a WeatherReading-compatible payload, or a
    sanitized failure status. No token or response body is included in errors.
    """
    config = _config()
    configured, reason = _configured(config)
    if not configured:
        _set_status(status="disabled" if reason == "source_disabled" else "unavailable", reason=reason)
        return {"status": _status["status"], "reason": reason, "reading": None}

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    entities = {
        "temp": config["temp_entity"],
        "humidity": config["humidity_entity"],
        "pressure": config["pressure_entity"],
    }
    if config["rain_entity"]:
        entities["rain"] = config["rain_entity"]
    if config["light_entity"]:
        entities["light"] = config["light_entity"]
    try:
        states = {key: _fetch_entity(entity, config) for key, entity in entities.items()}
    except requests.RequestException as exc:
        reason = "http_error"
        if getattr(getattr(exc, "response", None), "status_code", None) == 429:
            reason = "http_429"
        _set_status(status="unavailable", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
        return {"status": "unavailable", "reason": reason, "reading": None}
    except (RuntimeError, ValueError, TypeError) as exc:
        reason = str(exc) if str(exc) in {"http_429", "invalid_state_response"} else "http_error"
        _set_status(status="unavailable", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
        return {"status": "unavailable", "reason": reason, "reading": None}

    timestamps: dict[str, datetime] = {}
    for key, state in states.items():
        observed, age = _state_age(state, now)
        if key in {"temp", "humidity", "pressure"}:
            if observed is None:
                reason = f"missing_timestamp_{key}"
                _set_status(status="unavailable", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
                return {"status": "unavailable", "reason": reason, "reading": None}
            if age is None or age > config["stale_after_seconds"]:
                reason = f"stale_{key}"
                _set_status(status="stale", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
                return {"status": "stale", "reason": reason, "reading": None}
            timestamps[key] = observed
        else:
            if key == "rain":
                # A configured rain sensor is part of the rain label. Never
                # turn an unknown state or stale numeric rate into dry.
                if _rain_is_binary_state(config["rain_entity"], state):
                    # For on/off entities the state itself is the observation;
                    # an old last_updated is normal while the contact remains
                    # unchanged. Do not let it move observation_at backwards.
                    if observed is not None and age is not None and age <= config["stale_after_seconds"]:
                        timestamps[key] = observed
                else:
                    if observed is None:
                        reason = "missing_timestamp_rain"
                        _set_status(status="unavailable", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
                        return {"status": "unavailable", "reason": reason, "reading": None}
                    if age is None or age > config["stale_after_seconds"]:
                        reason = "stale_rain"
                        _set_status(status="stale", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
                        return {"status": "stale", "reason": reason, "reading": None}
                    timestamps[key] = observed
            elif observed is not None and age is not None and age > config["stale_after_seconds"]:
                # Optional light does not make the whole weather reading stale.
                states[key] = {}

    temp = _temperature(states["temp"])
    humidity = _humidity(states["humidity"])
    pressure = _pressure(states["pressure"])
    if temp is None or humidity is None or pressure is None:
        missing = "temp" if temp is None else "humidity" if humidity is None else "pressure"
        reason = f"invalid_{missing}"
        _set_status(status="unavailable", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
        return {"status": "unavailable", "reason": reason, "reading": None}

    observation_at = min(timestamps.values())
    rain_value: bool | None = None
    rain_sensor: float | None = None
    if config["rain_entity"] and states.get("rain"):
        rain_value, rain_sensor = _rain(states["rain"])
        if rain_value is None:
            reason = "invalid_rain"
            _set_status(status="unavailable", reason=reason, last_error=reason, last_error_at_utc=_now_iso(), error_count=_status.get("error_count", 0) + 1)
            return {"status": "unavailable", "reason": reason, "reading": None}

    reading: dict[str, Any] = {
        "timestamp": _now_iso(observation_at),
        "temp": round(temp, 4),
        "humidity": round(humidity, 4),
        "pressure": round(pressure, 4),
        "rain": rain_value,
        "rain_flag": 1.0 if rain_value else 0.0,
        "rain_sensor": rain_sensor,
        "light": _light(states["light"]) if states.get("light") else None,
    }
    snapshot = {"reading": reading, "timestamps": {key: _now_iso(value) for key, value in timestamps.items()}}
    snapshot_key = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    global _last_snapshot_key
    with _state_lock:
        if snapshot_key == _last_snapshot_key:
            _status["status"] = "duplicate"
            _status["reason"] = "duplicate_observation"
            _status["duplicate_count"] = int(_status.get("duplicate_count", 0)) + 1
            return {"status": "duplicate", "reason": "duplicate_observation", "reading": None, "observation_at_utc": reading["timestamp"]}
        _last_snapshot_key = snapshot_key
    _set_status(status="ready", reason=None, last_success_at_utc=_now_iso(), last_observation_at_utc=reading["timestamp"], last_error=None)
    return {"status": "ready", "reason": None, "reading": reading, "observation_at_utc": reading["timestamp"]}


def poll_once(callback: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    result = fetch_reading()
    if result.get("status") == "ready" and result.get("reading"):
        callback(result["reading"])
    return result


def record_ingest_result(result: dict[str, Any], observation_at: str | None = None) -> None:
    """Record whether the common pipeline produced a prediction.

    Fetch health and model readiness are separate signals: a valid HA sample
    can be persisted while the model is still waiting for its continuous
    feature window. Keeping that distinction in status makes this diagnosable
    without exposing the full prediction or any credentials.
    """
    ready = bool(result.get("prediction_ready"))
    message = str(result.get("message") or "")
    reason = None if ready else "prediction_not_ready"
    lowered = message.lower()
    if not ready and any(token in lowered for token in ("not enough", "continuous", "history")):
        reason = "waiting_for_history"
    elif not ready and "stale" in lowered:
        reason = "stale_history"
    now = _now_iso()
    with _state_lock:
        if observation_at:
            _status["last_observation_at_utc"] = observation_at
        _status["last_ingest_at_utc"] = now
        _status["ingest_count"] = int(_status.get("ingest_count", 0)) + 1
        _status["last_prediction_ready"] = ready
        _status["last_prediction_reason"] = reason
        if ready:
            _status["last_prediction_success_at_utc"] = now
            _status["prediction_count"] = int(_status.get("prediction_count", 0)) + 1


def poll_loop(callback: Callable[[dict[str, Any]], Any]) -> None:
    """Continuously poll HA. A callback failure is isolated from the loop."""
    backoff_seconds = 60
    while True:
        config = _config()
        interval = config["poll_seconds"]
        cycle_started = time.monotonic()
        result = fetch_reading()
        if result.get("status") == "ready" and result.get("reading"):
            try:
                callback(result["reading"])
            except Exception:
                logger.exception("Home Assistant reading callback failed")
            backoff_seconds = interval
        elif result.get("status") == "duplicate":
            backoff_seconds = interval
        else:
            # A transient timeout/429/5xx should not hammer the Core API. The
            # next successful read resets the cadence to the configured poll.
            backoff_seconds = min(300, max(interval, backoff_seconds * 2))
        # Sleep only for the remainder of the interval. The callback is
        # normally a non-blocking queue put, but this also keeps cadence stable
        # if a caller supplies a synchronous callback in another deployment.
        time.sleep(max(0.0, backoff_seconds - (time.monotonic() - cycle_started)))


def get_source_status() -> dict[str, Any]:
    config = _config()
    configured, reason = _configured(config)
    server_now = datetime.now(timezone.utc)
    with _state_lock:
        current = dict(_status)
    if not config["enabled"]:
        current["status"] = "disabled"
        current["reason"] = "source_disabled"
    elif not configured:
        current["status"] = "unavailable"
        current["reason"] = reason
    elif current.get("status") == "disabled" and current.get("reason") == "source_disabled":
        current["status"] = "starting"
        current["reason"] = "awaiting_first_poll"
    observed_at = _parse_timestamp(current.get("last_observation_at_utc"))
    observation_age = None
    if observed_at is not None:
        observation_age = max(0.0, (server_now - observed_at).total_seconds())
    current.update({
        "enabled": bool(config["enabled"]),
        "configured": bool(configured),
        "reason": current.get("reason") or reason,
        "poll_seconds": config["poll_seconds"],
        "stale_after_seconds": config["stale_after_seconds"],
        "server_now_at_utc": server_now.isoformat().replace("+00:00", "Z"),
        "observation_age_seconds": observation_age,
        "api_base": _safe_api_base(config["api_base"]),
        "entities": {
            "temp": _mask_entity(config["temp_entity"]),
            "humidity": _mask_entity(config["humidity_entity"]),
            "pressure": _mask_entity(config["pressure_entity"]),
            "rain": _mask_entity(config["rain_entity"]),
            "light": _mask_entity(config["light_entity"]),
        },
        "token_configured": bool(config["token"]),
    })
    return current


def is_enabled() -> bool:
    configured, _ = _configured(_config())
    return configured
