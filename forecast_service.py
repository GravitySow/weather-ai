"""Unified forecast blend layer — see plan.md "Phase 3".

Single entry point, get_forecast(), blending three sources by lead time —
the way real met services do nowcast-to-NWP blending, and the core
architectural idea of plan.md's whole roadmap:

    0-120min: local RF nowcast   (persistence-quality; proven onset ceiling
                                  — recall 0.000 on dry-antecedent rows, see
                                  plan.md "Hard constraints")
    30min-2h: radar advection   (the one genuine onset signal in this
                                  system — still shadow-mode/unvalidated,
                                  see radar_advection.py)
    2h-7days: NWP (Open-Meteo)  (nwp_forecast.py)

This module only assembles the blended object — see plan.md Phase 4 for
wiring it into what Telegram/HA actually display. Building it standalone
first (Phase 3) means telegram_bot.py, ha_publisher.py, and weather_api.py
can all eventually read from ONE object instead of each doing their own
ad-hoc predict()/nwp_forecast() calls and risking disagreement, without
having to land that refactor and the new presentation surfaces in the same
change.
"""

import logging

import nwp_forecast
import predict_weather_ai
import radar_advection
import rain_stop

logger = logging.getLogger(__name__)

# Local RF nowcast owns 0-120min when the bundle contains those horizons;
# radar advection remains a separate arrival diagnostic. Past this, only
# NWP's hourly/daily sections have anything to say — see module docstring.
RADAR_HANDOFF_MINUTES = 120

HOURLY_HOURS = 24
DAILY_DAYS = 7


def _nowcast_points(prediction):
    """Probabilities from trained local classifiers, soonest first.

    Radar motion confidence is not a rain probability. Its experimental ETA
    remains in ``arrival``; rendering it as a rain percentage here is misleading.
    """
    points = []
    for horizon, pred in prediction["predictions"].items():
        point = {
            "lead_min": int(horizon.rstrip("m")),
            "rain_prob": float(pred["probability"]),
            "source": f"local_{prediction.get('model', 'rf')}",
        }
        if "threshold" in pred:
            point["threshold"] = float(pred["threshold"])
        if "model_rain_signal" in pred:
            point["model_signal"] = bool(pred["model_rain_signal"])
        points.append(point)

    return sorted(points, key=lambda p: p["lead_min"])


def _arrival_from_prediction(prediction):
    advection = prediction.get("radar_advection") or {}
    if not advection.get("available") or not advection.get("rain_arriving"):
        return None
    return {
        "eta_minutes": advection["eta_minutes"],
        "expected_intensity": advection["expected_intensity"],
        "motion_deg": advection["motion_deg"],
        "motion_kmh": advection["motion_kmh"],
        "confidence": advection["confidence"],
    }


def get_forecast(model_kind="rf"):
    """Returns the unified forecast object:

      {
        "now":      {temp, humidity, pressure, pressure_trend, trend_arrow,
                      heat_index, comfort_level, is_raining_now},
        "nowcast":  [{lead_min, rain_prob, source}, ...],
        "hourly":   [{time, temp, rain_prob, precip_mm, wind_kmh,
                      wind_dir_deg, cloud_cover, condition}, ...],
        "daily":    [{date, temp_min, temp_max, rain_prob_max, precip_mm,
                      wind_kmh_max, uv_index_max, sunrise, sunset,
                      condition}, ...],
        "arrival":  {eta_minutes, expected_intensity, motion_deg,
                      motion_kmh, confidence} or None,
        "sources":  {nowcast: [...], hourly: "nwp"|None, daily: "nwp"|None,
                      arrival: "radar_advection"|None},
        "source_status": {local, nwp, radar},
        "rain_stop": {status, estimated_stop_at, remaining_minutes, ...},
      }

    ``status`` is ``ready`` when local nowcast is available, ``partial`` when
    at least one source (normally NWP) is usable, and ``unavailable`` when no
    source can provide a forecast.  The hourly/daily/arrival sections degrade
    independently and retain a reason for each unavailable source.
    """
    prediction = None
    local_error = None
    try:
        prediction = predict_weather_ai.predict(model_kind=model_kind)
    except ValueError as exc:
        local_error = str(exc)
        logger.info("get_forecast: local sensor nowcast not ready yet: %s", exc)
    except Exception as exc:
        local_error = str(exc)
        logger.exception("get_forecast: predict() failed")

    nowcast = _nowcast_points(prediction) if prediction else []
    arrival = _arrival_from_prediction(prediction) if prediction else None
    radar_error = None
    if prediction is None:
        # Radar advection is independent of the local sensor model and can
        # still provide a useful arrival diagnostic during sensor outages.
        try:
            radar_signal = radar_advection.get_advection_signal()
            arrival = _arrival_from_prediction({"radar_advection": radar_signal})
        except Exception as exc:
            radar_error = str(exc)
            logger.exception("get_forecast: radar arrival fetch failed")

    nwp_error = None
    try:
        hourly = nwp_forecast.get_hourly_timeline(hours=HOURLY_HOURS)
    except Exception as exc:
        nwp_error = str(exc)
        logger.exception("get_forecast: hourly timeline fetch failed")
        hourly = []

    try:
        daily = nwp_forecast.get_daily_forecast(days=DAILY_DAYS)
    except Exception as exc:
        nwp_error = nwp_error or str(exc)
        logger.exception("get_forecast: daily forecast fetch failed")
        daily = []

    hourly_source = None
    if hourly:
        hourly_source = "nwp_bias_corrected" if any(
            item.get("temp_bias_correction_enabled") for item in hourly
        ) else "nwp"

    local_status = {"status": "ready" if prediction else "unavailable"}
    if local_error:
        local_status["reason"] = local_error
    nwp_status = nwp_forecast.get_source_status()
    if not hourly and not daily:
        nwp_status["status"] = "unavailable"
        if nwp_error:
            nwp_status["reason"] = nwp_error
    radar_status = {"status": "ready" if arrival else "unavailable"}
    if radar_error:
        radar_status["reason"] = radar_error
    available_sources = bool(prediction or hourly or daily or arrival)
    status = "ready" if prediction else "partial" if available_sources else "unavailable"

    result = {
        "status": status,
        "source_status": {"local": local_status, "nwp": nwp_status, "radar": radar_status},
        "now": None,
        "nowcast": nowcast,
        "hourly": hourly,
        "daily": daily,
        "arrival": arrival,
        "sources": {
            "nowcast": sorted({p["source"] for p in nowcast}),
            "hourly": hourly_source,
            "daily": "nwp" if daily else None,
            "arrival": "radar_advection" if arrival else None,
        },
        "confidence": {
            "level": prediction.get("confidence_level") if prediction else "low",
            "data_confidence": prediction.get("data_confidence") if prediction else 0.0,
            "reason": prediction.get("confidence_reason") if prediction else local_error or "no local sensor model",
        },
    }
    try:
        result["rain_stop"] = rain_stop.get_rain_stop_forecast()
    except Exception as exc:
        # Rain-stop is an experimental, history-dependent side product. A
        # database outage must not make an otherwise usable forecast fail.
        logger.exception("get_forecast: rain-stop estimate failed")
        result["rain_stop"] = {"status": "unavailable", "reason": str(exc)}
    if prediction:
        result["now"] = {
            "timestamp": prediction["timestamp"],
            "temp": prediction["temp"],
            "humidity": prediction["humidity"],
            "pressure": prediction["pressure"],
            "pressure_trend": prediction["pressure_trend"],
            "trend_arrow": prediction["trend_arrow"],
            "heat_index": prediction["heat_index"],
            "comfort_level": prediction["comfort_level"],
            "is_raining_now": prediction["is_raining_now"],
            "model_version": prediction.get("model_version"),
            "model_status": prediction.get("model_status"),
            "feature_count": prediction.get("feature_count"),
            "feature_coverage": prediction.get("feature_coverage"),
            "data_age_seconds": prediction.get("data_age_seconds"),
            "data_confidence": prediction.get("data_confidence"),
            "confidence_level": prediction.get("confidence_level"),
            "confidence_reason": prediction.get("confidence_reason"),
        }
    return result if available_sources else None
