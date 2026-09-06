"""Meaningful forecast-revision detection with restart-safe local state.

Forecast arrays move forward every hour, so comparing array positions creates
false revisions.  This module matches entries by their valid timestamp and
only reports changes that exceed user-facing thresholds.  State is stored in a
small JSON file under ``WEATHER_DATA_DIR``; a failed write never blocks a
forecast response.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

RAIN_WINDOW_THRESHOLD = float(os.getenv("REVISION_RAIN_WINDOW_THRESHOLD", "50"))
RAIN_PROBABILITY_DELTA = float(os.getenv("REVISION_RAIN_PROBABILITY_DELTA", "20"))
TEMPERATURE_DELTA_C = float(os.getenv("REVISION_TEMPERATURE_DELTA_C", "2"))
RAIN_WINDOW_SHIFT_MINUTES = int(os.getenv("REVISION_RAIN_WINDOW_SHIFT_MINUTES", "60"))

_lock = threading.Lock()


def _state_path():
    configured = os.getenv("FORECAST_REVISION_STATE_PATH")
    if configured:
        return Path(configured)
    return Path(os.getenv("WEATHER_DATA_DIR", "dataset")) / "forecast_revision_state.json"


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _stable(value):
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    number = _finite(value)
    if isinstance(value, (int, float)) and number is not None:
        return round(number, 6)
    return value


def _snapshot(forecast):
    """Keep only fields needed for a comparable, stable forecast snapshot."""
    hourly = []
    for entry in forecast.get("hourly") or []:
        if not entry.get("time"):
            continue
        hourly.append({
            "time": entry.get("time"),
            "temp": entry.get("temp"),
            "rain_prob": entry.get("rain_prob"),
            "precip_mm": entry.get("precip_mm"),
            "wind_kmh": entry.get("wind_kmh"),
            "condition": entry.get("condition"),
        })
    daily = []
    for entry in forecast.get("daily") or []:
        if not entry.get("date"):
            continue
        daily.append({
            "date": entry.get("date"),
            "temp_min": entry.get("temp_min"),
            "temp_max": entry.get("temp_max"),
            "rain_prob_max": entry.get("rain_prob_max"),
            "precip_mm": entry.get("precip_mm"),
            "condition": entry.get("condition"),
        })
    nowcast = []
    for point in forecast.get("nowcast") or []:
        nowcast.append({"lead_min": point.get("lead_min"), "rain_prob": point.get("rain_prob")})
    source_status = {}
    for name, status in (forecast.get("source_status") or {}).items():
        source_status[name] = {
            "status": status.get("status"),
            # Cache age changes on every request and is not a new forecast run;
            # the fetched/issue timestamp is the stable provider revision key.
            "fetched_at_utc": status.get("fetched_at_utc"),
        }
    payload = _stable({
        "status": forecast.get("status"),
        "sources": forecast.get("sources"),
        "source_status": source_status,
        "station": {
            "latitude": os.getenv("WEATHER_LAT"),
            "longitude": os.getenv("WEATHER_LON"),
        },
        "model_version": (forecast.get("now") or {}).get("model_version"),
        "nowcast": nowcast,
        "hourly": hourly,
        "daily": daily,
    })
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "snapshot_id": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


def _read_state():
    path = _state_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_state(snapshot):
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="forecast-revision-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "snapshot": snapshot}, handle,
                      ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _first_rain_time(hourly):
    for entry in hourly:
        probability = _finite(entry.get("rain_prob"))
        if probability is not None and probability >= RAIN_WINDOW_THRESHOLD:
            try:
                return datetime.fromisoformat(str(entry["time"]).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
    return None


def compare_snapshots(previous, current):
    """Compare two snapshots using matching valid timestamps, never positions."""
    if not previous or not current:
        return {"status": "not_comparable", "comparable": False, "reasons": ["missing_snapshot"]}
    old = previous.get("payload", previous)
    new = current.get("payload", current)
    old_sources = old.get("source_status", {})
    new_sources = new.get("source_status", {})
    nwp_updated = old_sources.get("nwp", {}).get("fetched_at_utc") != new_sources.get("nwp", {}).get("fetched_at_utc")
    model_changed = old.get("model_version") != new.get("model_version")
    old_hours = {item.get("time"): item for item in old.get("hourly", []) if item.get("time")}
    new_hours = {item.get("time"): item for item in new.get("hourly", []) if item.get("time")}
    common = sorted(set(old_hours) & set(new_hours))
    if not common:
        return {"status": "not_comparable", "comparable": False, "reasons": ["no_common_valid_times"]}

    probability_deltas = []
    temperature_deltas = []
    for valid_time in common:
        old_prob = _finite(old_hours[valid_time].get("rain_prob"))
        new_prob = _finite(new_hours[valid_time].get("rain_prob"))
        if old_prob is not None and new_prob is not None:
            probability_deltas.append(abs(new_prob - old_prob))
        old_temp = _finite(old_hours[valid_time].get("temp"))
        new_temp = _finite(new_hours[valid_time].get("temp"))
        if old_temp is not None and new_temp is not None:
            temperature_deltas.append(abs(new_temp - old_temp))

    old_first = _first_rain_time(old.get("hourly", []))
    new_first = _first_rain_time(new.get("hourly", []))
    shift_minutes = None
    if old_first is not None and new_first is not None:
        if old_first.tzinfo is None:
            old_first = old_first.replace(tzinfo=timezone.utc)
        if new_first.tzinfo is None:
            new_first = new_first.replace(tzinfo=timezone.utc)
        shift_minutes = (new_first - old_first).total_seconds() / 60

    reasons = []
    max_probability_delta = max(probability_deltas, default=0.0)
    max_temperature_delta = max(temperature_deltas, default=0.0)
    if max_probability_delta >= RAIN_PROBABILITY_DELTA:
        reasons.append("rain_probability")
    if shift_minutes is not None and abs(shift_minutes) >= RAIN_WINDOW_SHIFT_MINUTES:
        reasons.append("rain_window_shift")
    if max_temperature_delta >= TEMPERATURE_DELTA_C:
        reasons.append("temperature")

    return {
        "status": "changed" if reasons else "unchanged",
        "comparable": True,
        "common_valid_times": len(common),
        "max_rain_probability_delta_points": round(max_probability_delta, 2),
        "max_temperature_delta_c": round(max_temperature_delta, 2),
        "rain_window_shift_minutes": round(shift_minutes, 1) if shift_minutes is not None else None,
        "previous_rain_window_start": old_first.isoformat() if old_first else None,
        "current_rain_window_start": new_first.isoformat() if new_first else None,
        "provider_updated": bool(nwp_updated),
        "model_or_postprocessing_changed": bool(model_changed),
        "reasons": reasons,
    }


def observe(forecast):
    """Return a revision status and persist this forecast as the latest snapshot."""
    current = _snapshot(forecast)
    with _lock:
        state = _read_state()
        previous = state.get("snapshot")
        if previous and previous.get("snapshot_id") == current["snapshot_id"]:
            comparison = {"status": "unchanged", "comparable": True, "reasons": []}
            comparison["snapshot_id"] = current["snapshot_id"]
            return comparison
        comparison = compare_snapshots(previous, current)
        comparison["snapshot_id"] = current["snapshot_id"]
        comparison["previous_snapshot_id"] = previous.get("snapshot_id") if previous else None
        comparison["captured_at_utc"] = current["captured_at_utc"]
        try:
            _write_state(current)
        except OSError:
            logger.exception("Could not persist forecast revision state")
    return comparison


def notification_allowed(revision, cooldown_seconds=3600, now=None):
    """Return whether a changed revision may trigger a proactive message."""
    if not revision or revision.get("status") != "changed":
        return False
    revision_id = revision.get("snapshot_id")
    if not revision_id:
        return False
    now = now or datetime.now(timezone.utc)
    with _lock:
        state = _read_state()
    notification = state.get("notification") or {}
    if notification.get("revision_id") == revision_id:
        return False
    previous = notification.get("notified_at")
    if previous:
        try:
            sent_at = datetime.fromisoformat(str(previous).replace("Z", "+00:00"))
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=timezone.utc)
            if (now - sent_at).total_seconds() < max(0, int(cooldown_seconds)):
                return False
        except (TypeError, ValueError):
            pass
    return True


def mark_notified(revision, now=None):
    """Persist a successful revision notification; failed delivery is retryable."""
    revision_id = revision.get("snapshot_id") if revision else None
    if not revision_id:
        return False
    now = now or datetime.now(timezone.utc)
    with _lock:
        state = _read_state()
        state["notification"] = {"revision_id": revision_id, "notified_at": now.isoformat()}
        snapshot = state.get("snapshot")
        if snapshot:
            try:
                path = _state_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix="forecast-revision-", suffix=".json", dir=path.parent)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(state, handle, ensure_ascii=False, indent=2)
                os.replace(temporary, path)
            except OSError:
                logger.exception("Could not persist revision notification state")
                return False
    return True
