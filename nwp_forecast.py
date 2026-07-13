"""Open-Meteo forecast client — free, no API key required.

Only used for the "today / tonight" outlook in the Telegram bot. The local
ML model in predict_weather_ai.py is deliberately not used for this: it was
tuned and validated for 5/10/30-minute nowcasting only (single sensor, no
wind/cloud/synoptic data) and loses accuracy well before the multi-hour
range this forecast covers.
"""

import logging
import os
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

def _clean_env(name):
    """bashio::config (Home Assistant) returns the literal string "null" for
    a schema key the user's saved addon options don't have yet — treat that
    the same as unset rather than passing it straight into a URL param."""
    value = os.getenv(name)
    if not value or value.lower() == "null":
        return None
    return value


WEATHER_LAT = _clean_env("WEATHER_LAT")
WEATHER_LON = _clean_env("WEATHER_LON")
WEATHER_TZ = _clean_env("WEATHER_TZ") or "Asia/Bangkok"

_WEATHER_CODE_TH = {
    0: "ท้องฟ้าแจ่มใส", 1: "แจ่มใสเป็นส่วนใหญ่", 2: "มีเมฆบางส่วน", 3: "มีเมฆมาก",
    45: "หมอก", 48: "หมอกน้ำแข็ง",
    51: "ฝนละอองเบา", 53: "ฝนละออง", 55: "ฝนละอองหนัก",
    61: "ฝนตกเล็กน้อย", 63: "ฝนตกปานกลาง", 65: "ฝนตกหนัก",
    80: "ฝนซู่เล็กน้อย", 81: "ฝนซู่ปานกลาง", 82: "ฝนซู่หนัก",
    95: "พายุฝนฟ้าคะนอง", 96: "พายุฝนฟ้าคะนองมีลูกเห็บ", 99: "พายุฝนฟ้าคะนองมีลูกเห็บหนัก",
}


def _describe_code(code):
    return _WEATHER_CODE_TH.get(int(code), f"code {int(code)}")


def _summarize(indices, temps, precip_probs, codes):
    if not indices:
        return None

    seg_temps = [temps[i] for i in indices]
    seg_probs = [precip_probs[i] for i in indices]
    peak_i = indices[seg_probs.index(max(seg_probs))]

    return {
        "temp_min": min(seg_temps),
        "temp_max": max(seg_temps),
        "rain_prob_max": max(seg_probs),
        "condition": _describe_code(codes[peak_i]),
    }


def get_today_tonight_forecast():
    """Return {"today": {...}, "tonight": {...}} or None on failure.

    "today" = 06:00-18:00 local, "tonight" = 18:00 today - 06:00 tomorrow.
    """
    if not WEATHER_LAT or not WEATHER_LON:
        logger.warning("WEATHER_LAT/WEATHER_LON not set; skipping NWP forecast")
        return None

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": WEATHER_LAT,
                "longitude": WEATHER_LON,
                "timezone": WEATHER_TZ,
                "forecast_days": 2,
                "hourly": "temperature_2m,precipitation_probability,weathercode",
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        logger.exception("Open-Meteo request failed")
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip_probs = hourly.get("precipitation_probability", [])
    codes = hourly.get("weathercode", [])

    if not times:
        return None

    parsed_times = [datetime.fromisoformat(t) for t in times]
    today = parsed_times[0].date()

    day_start = datetime.combine(today, datetime.min.time()).replace(hour=6)
    day_end = datetime.combine(today, datetime.min.time()).replace(hour=18)
    night_start = day_end
    night_end = datetime.combine(today + timedelta(days=1), datetime.min.time()).replace(hour=6)

    day_idx = [i for i, t in enumerate(parsed_times) if day_start <= t < day_end]
    night_idx = [i for i, t in enumerate(parsed_times) if night_start <= t < night_end]

    return {
        "today": _summarize(day_idx, temps, precip_probs, codes),
        "tonight": _summarize(night_idx, temps, precip_probs, codes),
    }
