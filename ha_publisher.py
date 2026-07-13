"""Pushes prediction results into Home Assistant as sensor/binary_sensor states.

Uses the Supervisor's Home Assistant Core API proxy (requires `homeassistant_api: true`
in config.yaml, which injects SUPERVISOR_TOKEN into the add-on container) — no MQTT
broker or extra HA configuration needed.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

HA_API_BASE = "http://supervisor/core/api"


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
                "nearby_max_intensity": radar["nearby_max_intensity"],
                "trend_rising": radar["trend_rising"],
            },
        )
        _set_state(
            "binary_sensor.weather_ai_radar_nearby",
            "on" if radar["nearby_max_intensity"] > 0 else "off",
            {
                "friendly_name": "Weather AI Radar Rain Nearby",
                "trend_rising": radar["trend_rising"],
            },
        )
