"""Open-Meteo cloud-cover nowcast signal — SHADOW MODE ONLY, parallel to radar_nowcast.py.

RainViewer's free tier does not actually serve satellite/cloud imagery (its
``satellite.infrared`` array in weather-maps.json was confirmed empty across
repeated live checks on 2026-07-16 — likely moved behind a paid tier), so this
uses Open-Meteo's NWP-model cloud_cover fields instead: the same free,
no-API-key forecast source already used by nwp_forecast.py for the
today/tonight outlook.

Important distinction from radar_nowcast.py: this is NOT an observed trend
across real satellite frames. It's a short-range NWP forecast (Open-Meteo's
blended model), sampled at hourly resolution. "trend_rising" here means "the
model's own forecast for the next couple hours is higher than now", not "we
watched the sky visibly build up." Treat it as a coarser, laggier, but still
independent-of-the-single-sensor precursor signal, alongside radar.

Like radar_nowcast.py this is purely additive/diagnostic: published to Home
Assistant for the user to eyeball via predict_weather_ai.py's
``result["cloud"]``, but NOT wired into any_rain_alert / rain_alert /
Telegram logic.
"""

import logging
import time
from datetime import datetime

import requests

from nwp_forecast import WEATHER_LAT, WEATHER_LON, WEATHER_TZ

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo's underlying NWP models don't refresh faster than ~hourly; a
# 15-minute TTL keeps us well within that while still picking up a new run
# promptly, without hitting the API on every ~1/min /reading call.
_CACHE_TTL_SECONDS = 15 * 60

_HTTP_TIMEOUT_SECONDS = 15

# Cloud cover % at/above which we call it "overcast" for the binary sensor.
# Exported (no leading underscore) so ha_publisher.py can reuse the same cutoff.
HIGH_COVER_THRESHOLD = 70

_signal_cache = {"result": None, "computed_at": 0.0}


def _fetch_cloud_data():
    """Return the raw Open-Meteo JSON, or None on any failure. Never raises."""
    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": WEATHER_LAT,
                "longitude": WEATHER_LON,
                "timezone": WEATHER_TZ,
                "forecast_days": 1,
                "current": "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high",
                "hourly": "cloud_cover",
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.exception("Open-Meteo cloud_cover fetch failed")
        return None


def get_cloud_signal():
    """Main entry point. Never raises — returns {"available": False} on any failure.

    On success returns:
      {
        "available": True,
        "cloud_cover_now": <int 0-100, total cloud cover %>,
        "cloud_cover_low_now": <int 0-100, low-level cloud % — most relevant
            to precipitation-bearing convective cloud>,
        "cloud_cover_next_hour": <int 0-100, forecasted total cover 1h ahead>,
        "trend_rising": <bool, forecast rises over the next 2 hourly steps>,
        "fetch_time": <ISO8601 local timestamp of the "current" sample>,
      }
    """
    try:
        now = time.monotonic()
        cached = _signal_cache["result"]
        if cached is not None and (now - _signal_cache["computed_at"]) < _CACHE_TTL_SECONDS:
            return cached

        if not WEATHER_LAT or not WEATHER_LON:
            logger.warning("WEATHER_LAT/WEATHER_LON not set; skipping cloud nowcast")
            return {"available": False}

        data = _fetch_cloud_data()
        if not data:
            return {"available": False}

        current = data.get("current", {})
        hourly = data.get("hourly", {})
        hourly_times = hourly.get("time", [])
        hourly_cover = hourly.get("cloud_cover", [])

        cloud_cover_now = current.get("cloud_cover")
        cloud_cover_low_now = current.get("cloud_cover_low")
        current_time = current.get("time")

        if cloud_cover_now is None or not hourly_times or not current_time:
            return {"available": False}

        current_hour = datetime.fromisoformat(current_time).replace(minute=0, second=0)
        current_iso_hour = current_hour.strftime("%Y-%m-%dT%H:00")

        next_values = []
        if current_iso_hour in hourly_times:
            idx = hourly_times.index(current_iso_hour)
            next_values = hourly_cover[idx : idx + 3]

        trend_rising = bool(
            len(next_values) >= 2
            and all(next_values[i] <= next_values[i + 1] for i in range(len(next_values) - 1))
            and next_values[-1] > next_values[0]
        )

        result = {
            "available": True,
            "cloud_cover_now": cloud_cover_now,
            "cloud_cover_low_now": cloud_cover_low_now,
            "cloud_cover_next_hour": next_values[1] if len(next_values) >= 2 else None,
            "trend_rising": trend_rising,
            "fetch_time": current_time,
        }
        _signal_cache["result"] = result
        _signal_cache["computed_at"] = now
        return result
    except Exception:
        logger.exception("Cloud nowcast signal computation failed")
        return {"available": False}
