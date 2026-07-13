"""RainViewer ground-radar nowcast signal — SHADOW MODE ONLY.

This module derives a coarse "is there rain nearby / is it trending up"
signal from RainViewer's free public radar mosaic (which, for Bangkok,
composites real Thai Meteorological Department ground radar). It is
diagnostic-only: the output is published to Home Assistant for the user to
eyeball, but is NOT wired into any rain_alert / any_rain_alert logic or the
Telegram bot. See predict_weather_ai.py's predict() for the (additive-only)
call site.

Attribution (RainViewer free-tier requirement): "Weather data by RainViewer"
(https://www.rainviewer.com) must be shown somewhere the data is displayed —
see dashboard.yaml.

Design notes / empirical findings (2026-07-11)
------------------------------------------------
- ``radar.nowcast`` in the weather-maps.json response was empty at the time
  this was built and isn't documented for the free tier, so there is no
  ready-made forecast frame to use. Instead we derive a simple trend from the
  sequence of ``radar.past`` frames ourselves (see ``get_radar_signal``).
- Zoom level: RainViewer's own docs (rainviewer.com/api/weather-maps-api.html)
  state max usable zoom is 7. This was confirmed empirically: fetching the
  same lat/lon at zoom 7 vs 8/9/10 showed zoom 8, 9 and 10 all return
  byte-for-byte IDENTICAL tile images (the server does not have finer detail
  beyond ~zoom 7/8 and just re-serves the same data), and the zoom-8+ tile's
  pixels were mostly grayscale/tan artifacts rather than real radar colors.
  Zoom 7 for tile (99, 59) covers lat ~11.18-13.92, lon ~98.44-101.25 — a
  ~300km box comfortably containing all of Bangkok and central Thailand (not
  clipped to ocean), and showed genuine blue/red radar colors during a live
  check. We use zoom 7.
- Color palette (RainViewer color scheme id 2, "Universal Blue"): RainViewer
  does not publish an official RGBA table for this scheme, so the buckets
  below were reverse-engineered empirically by fetching several live past
  frames over Bangkok and inspecting the distinct opaque RGBA values that came
  back:
    * Fully transparent / alpha ~0: no rain.
    * A blue family (R channel low, B notably higher than R e.g. B-R > ~90)
      ramping from bright cyan ~(136,221,238) for the lightest/weakest signal
      down to dark navy ~(0,71,104) for the heaviest normal rain.
    * A red/orange/yellow family (R==255, B~0, e.g. (255,68,0)..(255,238,0))
      that appears to represent extreme returns (e.g. hail cores).
    * Smoothing (the "1_1" tile option) also produces grayish/tan
      anti-aliased edge-blend pixels (e.g. (194,180,130,140)) that don't match
      either family above; these are treated as bucket 0 (no signal) rather
      than guessed at, per the "when in doubt, no rain" conservative posture.
  Exact calibration doesn't matter here — this only needs to be a monotonic
  relative signal for a coarse trend heuristic, not calibrated mm/hr.
"""

import io
import logging
import time

import requests
from PIL import Image

from nwp_forecast import WEATHER_LAT, WEATHER_LON

logger = logging.getLogger(__name__)

RAINVIEWER_INDEX_URL = "https://api.rainviewer.com/public/weather-maps.json"

# RainViewer publishes a new past-radar frame roughly every 10 minutes; a
# 5-minute TTL means we always pick up a new frame within one cycle of it
# appearing, while never hitting their API more than ~12x/hour regardless of
# how often /reading is polled (~once/minute).
_CACHE_TTL_SECONDS = 5 * 60

# See module docstring for how this was chosen: max usable zoom is ~7,
# confirmed empirically (zoom 8+ returns byte-identical, non-radar-looking
# tiles for this location).
ZOOM = 7
TILE_SIZE = 256

# "2" = RainViewer color scheme id 2, "Universal Blue" (their only free-tier
# scheme). "1_1" = smoothing on, snow-color overlay on (see module docstring).
_TILE_URL_TEMPLATE = "{host}{path}/{size}/{zoom}/{tile_x}/{tile_y}/2/1_1.png"

_HTTP_TIMEOUT_SECONDS = 15

# How many of the most recent past frames to sample for the trend heuristic.
_MAX_SAMPLE_FRAMES = 6

_frame_cache = {"host": None, "frames": [], "fetched_at": 0.0}
_signal_cache = {"result": None, "computed_at": 0.0}


def _fetch_frame_list():
    """Return (host, past_frames), cached with a ~5 minute TTL.

    Never raises: on any request/parse failure, logs via logger.exception and
    returns (None, []).
    """
    now = time.monotonic()
    if _frame_cache["frames"] and (now - _frame_cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return _frame_cache["host"], _frame_cache["frames"]

    try:
        response = requests.get(RAINVIEWER_INDEX_URL, timeout=_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        host = data["host"]
        past_frames = data["radar"]["past"]
    except Exception:
        logger.exception("RainViewer frame-list fetch failed")
        return None, []

    _frame_cache["host"] = host
    _frame_cache["frames"] = past_frames
    _frame_cache["fetched_at"] = now
    return host, past_frames


def _latlon_to_tile_pixel(lat, lon, zoom, tile_size=TILE_SIZE):
    """Standard slippy-map (Web Mercator) lat/lon -> tile x/y + pixel-within-tile.

    Returns (tile_x, tile_y, pixel_x, pixel_y).
    """
    import math

    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n

    tile_x = int(x)
    tile_y = int(y)
    pixel_x = int((x - tile_x) * tile_size)
    pixel_y = int((y - tile_y) * tile_size)
    return tile_x, tile_y, pixel_x, pixel_y


def _tile_url(host, path, tile_x, tile_y, zoom=ZOOM, tile_size=TILE_SIZE):
    return _TILE_URL_TEMPLATE.format(
        host=host, path=path, size=tile_size, zoom=zoom, tile_x=tile_x, tile_y=tile_y
    )


def _fetch_tile_png(url):
    response = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _decode_tile(png_bytes):
    """Decode PNG bytes into a Pillow RGBA image."""
    image = Image.open(io.BytesIO(png_bytes))
    return image.convert("RGBA")


def _intensity_bucket(rgba):
    """Map one RGBA pixel to a coarse 0-4 rain-intensity bucket.

    0 = none/transparent/unrecognized, 1 = light, 2 = moderate, 3 = heavy,
    4 = extreme. See module docstring for how these breakpoints were derived.
    """
    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]

    if a < 32:
        return 0

    if r >= 200 and b <= 20:
        # Red/orange/yellow "extreme" family, e.g. (255,68,0) .. (255,238,0).
        return 4

    if (b - r) >= 60:
        # Genuine "Universal Blue" rain family (as opposed to grayish/tan
        # anti-aliased edge-blend pixels, which have b - r near zero or
        # negative). Darker/less bright blue = heavier rain.
        if b >= 202:
            return 1
        if b >= 149:
            return 2
        return 3

    return 0


def _tile_max_intensity(image):
    """Max intensity bucket across every distinct color in the decoded tile."""
    colors = image.getcolors(maxcolors=TILE_SIZE * TILE_SIZE)
    if not colors:
        return 0
    return max(_intensity_bucket(color) for _count, color in colors)


def get_radar_signal():
    """Main entry point. Never raises — returns {"available": False} on any failure.

    On success returns:
      {
        "available": True,
        "point_intensity": <int 0-4, latest frame's bucket at the sensor's pixel>,
        "nearby_max_intensity": <int 0-4, latest frame's max bucket across the tile>,
        "trend_rising": <bool>,
        "frame_time": <unix ts of latest frame used>,
      }
    """
    try:
        now = time.monotonic()
        cached = _signal_cache["result"]
        if cached is not None and (now - _signal_cache["computed_at"]) < _CACHE_TTL_SECONDS:
            return cached

        if not WEATHER_LAT or not WEATHER_LON:
            logger.warning("WEATHER_LAT/WEATHER_LON not set; skipping radar nowcast")
            return {"available": False}

        host, past_frames = _fetch_frame_list()
        if not host or not past_frames:
            return {"available": False}

        lat = float(WEATHER_LAT)
        lon = float(WEATHER_LON)
        tile_x, tile_y, pixel_x, pixel_y = _latlon_to_tile_pixel(lat, lon, ZOOM)

        sample_frames = past_frames[-_MAX_SAMPLE_FRAMES:]

        nearby_max_sequence = []
        point_intensity_latest = 0
        nearby_max_latest = 0
        frame_time_latest = None

        for frame in sample_frames:
            url = _tile_url(host, frame["path"], tile_x, tile_y)
            png_bytes = _fetch_tile_png(url)
            image = _decode_tile(png_bytes)

            point_intensity = _intensity_bucket(image.getpixel((pixel_x, pixel_y)))
            nearby_max = _tile_max_intensity(image)

            nearby_max_sequence.append(nearby_max)
            point_intensity_latest = point_intensity
            nearby_max_latest = nearby_max
            frame_time_latest = frame["time"]

        trend_rising = bool(
            len(nearby_max_sequence) >= 2
            and all(
                nearby_max_sequence[i] <= nearby_max_sequence[i + 1]
                for i in range(len(nearby_max_sequence) - 1)
            )
            and nearby_max_sequence[-1] > 0
            and nearby_max_sequence[-1] > nearby_max_sequence[0]
        )

        result = {
            "available": True,
            "point_intensity": point_intensity_latest,
            "nearby_max_intensity": nearby_max_latest,
            "trend_rising": trend_rising,
            "frame_time": frame_time_latest,
        }
        _signal_cache["result"] = result
        _signal_cache["computed_at"] = now
        return result
    except Exception:
        logger.exception("Radar nowcast signal computation failed")
        return {"available": False}
