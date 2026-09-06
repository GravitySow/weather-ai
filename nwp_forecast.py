"""Open-Meteo forecast client — free, no API key required.

Single shared fetch (`_fetch_open_meteo_data`) backs every consumer: the
Telegram bot's today/tonight outlook, the 7-day/hourly timeline added
2026-07-21, and cloud_nowcast.py's cloud-cover signal — one Open-Meteo
request per cache window instead of two+ separate ones for the same
coordinates.

The local ML model in predict_weather_ai.py is deliberately not used for
anything here: it was tuned and validated for 5/10/30-minute nowcasting only
(single sensor, no wind/cloud/synoptic data) and loses accuracy well before
the multi-hour range this forecast covers.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests

import nwp_bias
import temp_interval

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo's underlying NWP models don't refresh faster than ~hourly; a
# 15-minute cache keeps every consumer well within that while still picking
# up a new model run promptly, without hitting the API on every ~1/min
# /reading call or every /forecast command.
_CACHE_TTL_SECONDS = 15 * 60
_HTTP_TIMEOUT_SECONDS = 15

_HOURLY_VARS = (
    "temperature_2m,precipitation_probability,weathercode,cloud_cover,"
    "precipitation,relative_humidity_2m,apparent_temperature,"
    "wind_speed_10m,wind_direction_10m"
)
_DAILY_VARS = (
    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "precipitation_probability_max,wind_speed_10m_max,uv_index_max,"
    "sunrise,sunset"
)
_CURRENT_VARS = "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high"

_FORECAST_DAYS = 7

_fetch_cache = {"data": None, "fetched_at": 0.0, "fetched_at_utc": None}
_bias_cache = {"path": None, "mtime": None, "model": None}


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


def _fetch_open_meteo_data():
    """Return the raw Open-Meteo JSON (cached ~15min), or None on failure/no
    coordinates. Never raises."""
    now = time.monotonic()
    cached = _fetch_cache["data"]
    if cached is not None and (now - _fetch_cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return cached

    if not WEATHER_LAT or not WEATHER_LON:
        logger.warning("WEATHER_LAT/WEATHER_LON not set; skipping Open-Meteo fetch")
        return None

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": WEATHER_LAT,
                "longitude": WEATHER_LON,
                "timezone": WEATHER_TZ,
                "forecast_days": _FORECAST_DAYS,
                "current": _CURRENT_VARS,
                "hourly": _HOURLY_VARS,
                "daily": _DAILY_VARS,
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        logger.exception("Open-Meteo request failed")
        return None

    _fetch_cache["data"] = data
    _fetch_cache["fetched_at"] = now
    _fetch_cache["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    return data


def get_source_status():
    """Describe NWP cache freshness without exposing the monotonic clock."""
    data = _fetch_cache.get("data")
    fetched_at = _fetch_cache.get("fetched_at_utc")
    if data is None or not fetched_at:
        return {"status": "unavailable", "fetched_at_utc": None, "cache_age_seconds": None}
    age = max(0.0, time.monotonic() - _fetch_cache.get("fetched_at", time.monotonic()))
    return {
        "status": "ready" if age < _CACHE_TTL_SECONDS else "stale",
        "fetched_at_utc": fetched_at,
        "cache_age_seconds": round(age, 1),
    }


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
    data = _fetch_open_meteo_data()
    if not data:
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


def get_daily_forecast(days=7):
    """Return a list of up to `days` dicts (soonest first):
      {date, temp_min, temp_max, rain_prob_max, precip_mm, wind_kmh_max,
       uv_index_max, sunrise, sunset, condition}
    or [] on failure.
    """
    data = _fetch_open_meteo_data()
    if not data:
        return []

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return []

    count = min(days, len(dates))
    result = []
    for i in range(count):
        result.append({
            "date": dates[i],
            "temp_min": daily.get("temperature_2m_min", [None] * count)[i],
            "temp_max": daily.get("temperature_2m_max", [None] * count)[i],
            "rain_prob_max": daily.get("precipitation_probability_max", [None] * count)[i],
            "precip_mm": daily.get("precipitation_sum", [None] * count)[i],
            "wind_kmh_max": daily.get("wind_speed_10m_max", [None] * count)[i],
            "uv_index_max": daily.get("uv_index_max", [None] * count)[i],
            "sunrise": daily.get("sunrise", [None] * count)[i],
            "sunset": daily.get("sunset", [None] * count)[i],
            "condition": _describe_code(daily["weather_code"][i]) if daily.get("weather_code") else None,
        })
    return result


def get_hourly_timeline(hours=24):
    """Return a list of up to `hours` dicts starting from the current hour
    (soonest first):
      {time, temp, rain_prob, precip_mm, wind_kmh, wind_dir_deg, cloud_cover,
       condition}
    or [] on failure.
    """
    data = _fetch_open_meteo_data()
    if not data:
        return []

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    current_time = data.get("current", {}).get("time")
    if not times:
        return []

    if current_time:
        current_hour = datetime.fromisoformat(current_time).replace(minute=0, second=0)
    else:
        current_hour = datetime.fromisoformat(times[0])
    current_iso_hour = current_hour.strftime("%Y-%m-%dT%H:00")

    start_idx = times.index(current_iso_hour) if current_iso_hour in times else 0
    end_idx = min(start_idx + hours, len(times))

    result = []
    for i in range(start_idx, end_idx):
        result.append({
            "time": times[i],
            "temp": hourly.get("temperature_2m", [None])[i] if i < len(hourly.get("temperature_2m", [])) else None,
            "rain_prob": hourly.get("precipitation_probability", [None])[i] if i < len(hourly.get("precipitation_probability", [])) else None,
            "precip_mm": hourly.get("precipitation", [None])[i] if i < len(hourly.get("precipitation", [])) else None,
            "wind_kmh": hourly.get("wind_speed_10m", [None])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
            "wind_dir_deg": hourly.get("wind_direction_10m", [None])[i] if i < len(hourly.get("wind_direction_10m", [])) else None,
            "cloud_cover": hourly.get("cloud_cover", [None])[i] if i < len(hourly.get("cloud_cover", [])) else None,
            "condition": _describe_code(hourly["weathercode"][i]) if i < len(hourly.get("weathercode", [])) else None,
        })
    # Every entry gets a stable lead index, even when no bias/interval artifact
    # is configured. This makes later post-processing auditable and keeps
    # entries aligned by valid time rather than list position.
    for offset, item in enumerate(result):
        item["lead_hours"] = offset
        item["issued_hour"] = current_iso_hour

    # Bias correction is opt-in because the first DB audit found a large,
    # likely timezone/units-related NWP offset.  Keep raw values visible so
    # operators can compare before enabling the candidate artifact.
    bias_path = _clean_env("NWP_BIAS_CORRECTION_PATH")
    if bias_path:
        try:
            path = os.path.abspath(bias_path)
            mtime = os.path.getmtime(path)
            if (_bias_cache["path"], _bias_cache["mtime"]) != (path, mtime):
                _bias_cache.update(path=path, mtime=mtime, model=nwp_bias.load_bias_model(path))
            result = nwp_bias.apply_to_timeline(result, _bias_cache["model"])
            for item in result:
                item["temp_bias_correction_enabled"] = True
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid NWP_BIAS_CORRECTION_PATH: %s", exc)
    # Unlike bias correction, intervals are never synthesized from a guessed
    # standard deviation. Without a validated artifact the field is null.
    return temp_interval.apply_to_timeline(result)
