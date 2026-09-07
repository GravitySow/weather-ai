"""Pushes prediction results into Home Assistant as sensor/binary_sensor states.

Uses the Supervisor's Home Assistant Core API proxy (requires `homeassistant_api: true`
in config.yaml, which injects SUPERVISOR_TOKEN into the add-on container) — no MQTT
broker or extra HA configuration needed.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import requests

import cloud_nowcast

logger = logging.getLogger(__name__)

HA_API_BASE = "http://supervisor/core/api"
BANGKOK_OFFSET = timedelta(hours=7)

# Simple 06:00-18:00 local heuristic for day/night, matching the same
# convention nwp_forecast.get_today_tonight_forecast() already uses — not
# real sunrise/sunset, but consistent with the rest of this project and
# avoids ha_publisher.py taking on an nwp_forecast dependency just for this.
_DAY_START_HOUR = 6
_DAY_END_HOUR = 18


def _set_state(entity_id, state, attributes=None):
    token = os.getenv("SUPERVISOR_TOKEN")
    if not token:
        logger.warning("SUPERVISOR_TOKEN not set; skipping HA state update for %s", entity_id)
        return False

    url = f"{HA_API_BASE}/states/{entity_id}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"state": state, "attributes": attributes or {}}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException:
        logger.exception("Failed to push HA state for %s", entity_id)
        return False


def publish_prediction(prediction):
    """Publishes one prediction dict (from predict_weather_ai.predict) as HA entities."""

    _set_state(
        "sensor.weather_ai_temperature",
        round(prediction["temp"], 1),
        {
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "state_class": "measurement",
            "friendly_name": "Weather AI Temperature",
        },
    )
    _set_state(
        "sensor.weather_ai_humidity",
        round(prediction["humidity"], 1),
        {
            "unit_of_measurement": "%",
            "device_class": "humidity",
            "state_class": "measurement",
            "friendly_name": "Weather AI Humidity",
        },
    )
    _set_state(
        "sensor.weather_ai_pressure",
        round(prediction["pressure"], 1),
        {
            "unit_of_measurement": "hPa",
            "device_class": "pressure",
            "state_class": "measurement",
            "friendly_name": "Weather AI Pressure",
        },
    )
    _set_state(
        "sensor.weather_ai_heat_index",
        round(prediction["heat_index"], 1),
        {
            "unit_of_measurement": "°C",
            "state_class": "measurement",
            "friendly_name": "Weather AI Heat Index",
            "comfort_level": prediction["comfort_level"],
        },
    )

    _set_state(
        "sensor.weather_ai_dew_point",
        round(prediction["dew_point"], 1),
        {
            "unit_of_measurement": "°C",
            "device_class": "temperature",
            "state_class": "measurement",
            "friendly_name": "Weather AI Dew Point",
        },
    )
    _set_state(
        "sensor.weather_ai_pressure_trend",
        round(prediction["pressure_trend"], 2),
        {
            "unit_of_measurement": "hPa/h",
            "state_class": "measurement",
            "friendly_name": "Weather AI Pressure Trend",
        },
    )
    _set_state(
        "sensor.weather_ai_trend_arrow",
        prediction["trend_arrow"],
        {"friendly_name": "Weather AI Pressure Trend Arrow"},
    )

    for horizon, info in prediction["predictions"].items():
        _set_state(
            f"sensor.weather_ai_rain_probability_{horizon}",
            round(info["probability"] * 100, 1),
            {
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "friendly_name": f"Weather AI Rain Probability {horizon}",
                "rain_alert": info["rain_alert"],
                "threshold": info["threshold"] * 100,
                "model_version": prediction.get("model_version"),
                "data_confidence": prediction.get("data_confidence"),
                "confidence_level": prediction.get("confidence_level"),
                "probability_postprocessing": prediction.get("probability_postprocessing"),
            },
        )

    _set_state(
        "binary_sensor.weather_ai_raining_now",
        "on" if prediction["is_raining_now"] else "off",
        {"device_class": "moisture", "friendly_name": "Weather AI Raining Now"},
    )
    _set_state(
        "binary_sensor.weather_ai_rain_alert",
        "on" if prediction["any_rain_alert"] else "off",
        {
            "device_class": "safety",
            "friendly_name": "Weather AI Rain Alert",
            "next_rain_alert_horizon": prediction["next_rain_alert_horizon"],
            "alert_message": prediction["alert_message"],
        },
    )

    # Shadow-mode diagnostic only (see radar_nowcast.py) — not wired into any
    # alert logic. Skip publishing when unavailable so HA shows the entity as
    # stale rather than a misleading 0/off value.
    radar = prediction.get("radar", {})
    if radar.get("available"):
        _set_state(
            "sensor.weather_ai_radar_intensity",
            radar["point_intensity"],
            {
                "friendly_name": "Weather AI Radar Intensity",
                "state_class": "measurement",
                "close_max_intensity": radar["close_max_intensity"],
                "intensity_0_10km": radar["intensity_0_10km"],
                "intensity_10_25km": radar["intensity_10_25km"],
                "intensity_25_50km": radar["intensity_25_50km"],
                "intensity_50_100km": radar["intensity_50_100km"],
                "trend_rising": radar["trend_rising"],
                # Deprecated — max over the old single-tile box, kept only
                # for continuity with in-flight HA history graphs.
                "nearby_max_intensity": radar["nearby_max_intensity"],
            },
        )
        _set_state(
            "binary_sensor.weather_ai_radar_nearby",
            "on" if radar["close_max_intensity"] > 0 else "off",
            {
                "friendly_name": "Weather AI Radar Rain Nearby",
                "trend_rising": radar["trend_rising"],
            },
        )

    # Shadow-mode diagnostic only (see cloud_nowcast.py) — not wired into any
    # alert logic. Parallel signal to radar above; NWP-forecast-derived, not an
    # observed trend. Skip publishing when unavailable so HA shows the entity
    # as stale rather than a misleading 0/off value.
    cloud = prediction.get("cloud", {})
    if cloud.get("available"):
        _set_state(
            "sensor.weather_ai_cloud_cover",
            cloud["cloud_cover_now"],
            {
                "friendly_name": "Weather AI Cloud Cover",
                "state_class": "measurement",
                "unit_of_measurement": "%",
                "cloud_cover_low_now": cloud["cloud_cover_low_now"],
                "cloud_cover_next_hour": cloud["cloud_cover_next_hour"],
                "trend_rising": cloud["trend_rising"],
            },
        )
        _set_state(
            "binary_sensor.weather_ai_cloud_rising",
            "on" if cloud["trend_rising"] else "off",
            {
                "friendly_name": "Weather AI Cloud Cover Rising",
                "cloud_cover_now": cloud["cloud_cover_now"],
                "overcast": cloud["cloud_cover_now"] >= cloud_nowcast.HIGH_COVER_THRESHOLD,
            },
        )

    # Shadow-mode diagnostic only (see radar_advection.py) — NOT wired into
    # any alert logic; unlike the radar/cloud signals above this one is also
    # logged to forecast_log for backtesting via forecast_scoring.py before
    # it's ever trusted for a Telegram push. Skip publishing when
    # unavailable so HA shows the entity as stale rather than a misleading
    # off/0 value.
    advection = prediction.get("radar_advection", {})
    if advection.get("available"):
        _set_state(
            "binary_sensor.weather_ai_radar_arriving",
            "on" if advection["rain_arriving"] else "off",
            {
                "friendly_name": "Weather AI Radar Rain Arriving (experimental)",
                "eta_minutes": advection["eta_minutes"],
                "expected_intensity": advection["expected_intensity"],
                "motion_deg": advection["motion_deg"],
                "motion_kmh": advection["motion_kmh"],
                "confidence": advection["confidence"],
            },
        )

    _publish_weather_entity(prediction, cloud)

    # Keep a dedicated heartbeat separate from the value sensors.  Home
    # Assistant may leave ``last_updated`` unchanged when a rounded value and
    # its attributes are identical to the previous write, even though a new
    # prediction was calculated successfully.  A timestamp state that changes
    # on every publish gives the dashboard and automations an unambiguous
    # prediction heartbeat and also records which source observation produced
    # it.
    published_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _set_state(
        "sensor.weather_ai_last_update",
        published_at,
        {
            "device_class": "timestamp",
            "friendly_name": "Weather AI Last Update",
            "source_observation_at_utc": prediction.get("timestamp"),
            "model_version": prediction.get("model_version"),
            "model": prediction.get("model"),
        },
    )


# HA's weather domain only accepts states from this fixed set — see
# https://www.home-assistant.io/integrations/weather/. We only have
# rain/cloud/day-night signals, so this is a coarse mapping onto it, not a
# translation of every possible condition (fog/hail/snowy/etc. have no
# equivalent in our sensor data and are never produced here).
def _ha_weather_condition(is_raining_now, cloud_cover_now, is_daytime):
    if is_raining_now:
        return "pouring" if (cloud_cover_now or 0) >= 90 else "rainy"
    if cloud_cover_now is None:
        return "sunny" if is_daytime else "clear-night"
    if cloud_cover_now >= 85:
        return "cloudy"
    if cloud_cover_now >= 30:
        return "partlycloudy"
    return "sunny" if is_daytime else "clear-night"


def _publish_weather_entity(prediction, cloud):
    """Publishes weather.weather_ai so stock HA weather cards work off a
    real weather-domain entity instead of only individual sensors — see
    plan.md "Phase 4".

    Current-conditions only (state + temperature/humidity/pressure
    attributes) — deliberately does NOT populate a `forecast` list. HA
    2023.9+ moved multi-day/hourly forecasts from a static state attribute
    to a service-based `weather.get_forecasts` call that a REST state POST
    like the rest of this module can't satisfy; shipping a stale/fake
    `forecast` attribute would look like it works while silently not
    rendering in current HA versions. Use forecast_service.get_forecast()
    via the Telegram /forecast and /week commands for the actual multi-day
    view instead — this entity exists for the current-conditions card and
    any automations that key off weather.weather_ai's state.
    """
    local_hour = (datetime.utcnow() + BANGKOK_OFFSET).hour
    is_daytime = _DAY_START_HOUR <= local_hour < _DAY_END_HOUR
    cloud_cover_now = cloud.get("cloud_cover_now") if cloud.get("available") else None

    condition = _ha_weather_condition(prediction["is_raining_now"], cloud_cover_now, is_daytime)

    _set_state(
        "weather.weather_ai",
        condition,
        {
            "friendly_name": "Weather AI",
            "temperature": round(prediction["temp"], 1),
            "temperature_unit": "°C",
            "humidity": round(prediction["humidity"], 1),
            "pressure": round(prediction["pressure"], 1),
            "pressure_unit": "hPa",
        },
    )


def publish_forecast_extras(forecast):
    """Publish non-local forecast side products when ``/forecast`` is read.

    These are deliberately separate from ``publish_prediction``: an NWP/TMD
    refresh must not be mistaken for a new sensor observation, and an
    unavailable source must be represented as ``unavailable`` rather than a
    misleading zero/off state.
    """
    warnings = forecast.get("official_warnings") or {}
    warning_status = warnings.get("status")
    active = warnings.get("warnings") or []
    if warning_status in ("ready", "stale"):
        top = active[0] if active else {}
        _set_state(
            "sensor.weather_ai_tmd_warning_count",
            len(active),
            {
                "friendly_name": "Weather AI TMD Warning Count",
                "state_class": "measurement",
                "source": "Thai Meteorological Department CAP",
                "feed_status": warning_status,
                "headline": top.get("headline"),
                "severity": top.get("severity"),
                "expires": top.get("expires"),
                "source_url": top.get("source_url") or warnings.get("source_url"),
            },
        )
        _set_state(
            "binary_sensor.weather_ai_tmd_warning",
            "on" if active else "off",
            {
                "device_class": "safety",
                "friendly_name": "Weather AI TMD Warning",
                "feed_status": warning_status,
                "headline": top.get("headline"),
                "severity": top.get("severity"),
                "expires": top.get("expires"),
                "source_url": top.get("source_url") or warnings.get("source_url"),
            },
        )
    else:
        _set_state("sensor.weather_ai_tmd_warning_count", "unavailable", {
            "friendly_name": "Weather AI TMD Warning Count",
            "source": "Thai Meteorological Department CAP",
            "feed_status": warning_status or "unavailable",
            "reason": warnings.get("reason"),
        })
        _set_state("binary_sensor.weather_ai_tmd_warning", "unavailable", {
            "device_class": "safety",
            "friendly_name": "Weather AI TMD Warning",
            "feed_status": warning_status or "unavailable",
            "reason": warnings.get("reason"),
        })

    stop = forecast.get("rain_stop") or {}
    if stop.get("status") == "experimental":
        remaining = stop.get("remaining_minutes") or {}
        _set_state(
            "sensor.weather_ai_rain_stop_eta",
            stop.get("estimated_stop_at") or "unknown",
            {
                "friendly_name": "Weather AI Rain Stop ETA",
                "device_class": "timestamp",
                "remaining_lower_min": remaining.get("lower"),
                "remaining_median_min": remaining.get("median"),
                "remaining_upper_min": remaining.get("upper"),
                "status": "experimental",
                "completed_events": (stop.get("coverage") or {}).get("completed_events"),
            },
        )
    else:
        _set_state("sensor.weather_ai_rain_stop_eta", "unavailable", {
            "friendly_name": "Weather AI Rain Stop ETA",
            "status": stop.get("status", "unavailable"),
            "reason": stop.get("reason"),
        })
