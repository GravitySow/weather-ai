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

2026-07-21: this used to make its own separate Open-Meteo request. Now
shares nwp_forecast._fetch_open_meteo_data()'s single cached fetch (same
coordinates, same ~15-minute cache window) instead of hitting the API a
second time per cycle — see nwp_forecast.py's module docstring.
"""

import logging
from datetime import datetime

import nwp_forecast

logger = logging.getLogger(__name__)

# Cloud cover % at/above which we call it "overcast" for the binary sensor.
# Exported (no leading underscore) so ha_publisher.py can reuse the same cutoff.
HIGH_COVER_THRESHOLD = 70


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
        data = nwp_forecast._fetch_open_meteo_data()
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

        return {
            "available": True,
            "cloud_cover_now": cloud_cover_now,
            "cloud_cover_low_now": cloud_cover_low_now,
            "cloud_cover_next_hour": next_values[1] if len(next_values) >= 2 else None,
            "trend_rising": trend_rising,
            "fetch_time": current_time,
        }
    except Exception:
        logger.exception("Cloud nowcast signal computation failed")
        return {"available": False}
