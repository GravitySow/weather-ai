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
  We use zoom 7.
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

Geometry fix (2026-07-21) — single-tile coverage was badly asymmetric
-----------------------------------------------------------------------
The original version fetched exactly one zoom-7 tile (256x256px, ~1.19km/px
at this latitude) containing the sensor's lat/lon. The sensor's pixel inside
that tile turned out to be (189, 5) — 5px from the tile's *north* edge — so
real coverage was 6km north / 299km south / 224km west / 79km east of the
house. "nearby_max_intensity" was the max over that whole lopsided box, and
weather_api.py's live Telegram alert gates on it: rain 250km away over the
Gulf of Thailand scored identically to rain 5km away, and a storm approaching
from the north was invisible until it was essentially overhead.

Fix: fetch a 3x3 tile neighbourhood (768x768px) centred on the sensor's home
tile and derive distance-ring metrics (0-10 / 10-25 / 25-50 / 50-100 km) from
the sensor's exact pixel, instead of one whole-tile max. The alert gate
(weather_api.py) and the "nearby" display text (telegram_bot.py,
ha_publisher.py) now use ``close_max_intensity`` (max of the 0-10km and
10-25km rings) instead of the old whole-tile field.

``nearby_max_intensity`` is still populated, computed exactly as before
(max over just the center/home tile of the 3x3 stitch, i.e. the same box as
the old single-tile fetch) so existing HA history graphs stay numerically
continuous across the upgrade. It should be considered deprecated — remove
it once nothing reads it anymore.

Past frames are immutable once RainViewer publishes them, so decoded tiles
are cached by (frame_time, tile_x, tile_y) — a new radar cycle (~every
10min) only requires fetching the newest frame's 9 tiles; the other frames'
tiles are already in cache from the previous call.
"""

import io
import logging
import math
import time

import numpy as np
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
STITCH_TILES_PER_SIDE = 3
STITCH_SIZE = TILE_SIZE * STITCH_TILES_PER_SIDE

# "2" = RainViewer color scheme id 2, "Universal Blue" (their only free-tier
# scheme). "1_1" = smoothing on, snow-color overlay on (see module docstring).
_TILE_URL_TEMPLATE = "{host}{path}/{size}/{zoom}/{tile_x}/{tile_y}/2/1_1.png"

_HTTP_TIMEOUT_SECONDS = 15

# How many of the most recent past frames to sample for the trend heuristic.
_MAX_SAMPLE_FRAMES = 6

# (near_km, far_km] ring boundaries used for the distance-based intensity
# metrics. Chosen to separate "essentially overhead" from "worth a heads-up"
# from "on the radar but not actionable yet".
_RING_BOUNDS_KM = (
    ("intensity_0_10km", 0.0, 10.0),
    ("intensity_10_25km", 10.0, 25.0),
    ("intensity_25_50km", 25.0, 50.0),
    ("intensity_50_100km", 50.0, 100.0),
)

_frame_cache = {"host": None, "frames": [], "fetched_at": 0.0}
_signal_cache = {"result": None, "computed_at": 0.0}

# Decoded-tile cache keyed by (frame_time, tile_x, tile_y) -> RGBA numpy array.
# Pruned to only the tiles needed by the current sample_frames on every call,
# so it never grows past _MAX_SAMPLE_FRAMES * 9 entries (~14MB at 6 frames).
_tile_array_cache = {}


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
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n

    tile_x = int(x)
    tile_y = int(y)
    pixel_x = int((x - tile_x) * tile_size)
    pixel_y = int((y - tile_y) * tile_size)
    return tile_x, tile_y, pixel_x, pixel_y


def _meters_per_pixel(lat, zoom=ZOOM):
    """Standard Web Mercator ground resolution at a given latitude/zoom."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def _tile_url(host, path, tile_x, tile_y, zoom=ZOOM, tile_size=TILE_SIZE):
    return _TILE_URL_TEMPLATE.format(
        host=host, path=path, size=tile_size, zoom=zoom, tile_x=tile_x, tile_y=tile_y
    )


def _fetch_tile_png(url):
    response = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _decode_tile_array(png_bytes):
    """Decode PNG bytes into an HxWx4 uint8 RGBA numpy array."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return np.array(image)


def _get_tile_array(host, frame, tile_x, tile_y):
    """Fetch+decode one tile, using the (frame_time, tile_x, tile_y) cache.

    Past frames never change once published, so a cache hit needs no network
    call at all.
    """
    key = (frame["time"], tile_x, tile_y)
    cached = _tile_array_cache.get(key)
    if cached is not None:
        return cached

    url = _tile_url(host, frame["path"], tile_x, tile_y)
    png_bytes = _fetch_tile_png(url)
    array = _decode_tile_array(png_bytes)
    _tile_array_cache[key] = array
    return array


def _stitch_frame(host, frame, tile_x, tile_y):
    """Fetch the 3x3 tile neighbourhood centred on (tile_x, tile_y) for one
    frame and stitch into a single STITCH_SIZE x STITCH_SIZE RGBA array."""
    canvas = np.zeros((STITCH_SIZE, STITCH_SIZE, 4), dtype=np.uint8)
    half = STITCH_TILES_PER_SIDE // 2
    for row, dy in enumerate(range(-half, half + 1)):
        for col, dx in enumerate(range(-half, half + 1)):
            tile_array = _get_tile_array(host, frame, tile_x + dx, tile_y + dy)
            y0, x0 = row * TILE_SIZE, col * TILE_SIZE
            canvas[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = tile_array
    return canvas


def _intensity_bucket(rgba):
    """Map one RGBA pixel to a coarse 0-4 rain-intensity bucket.

    0 = none/transparent/unrecognized, 1 = light, 2 = moderate, 3 = heavy,
    4 = extreme. See module docstring for how these breakpoints were derived.
    """
    r, g, b, a = int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])

    if a < 32:
        return 0

    if r >= 200 and b <= 20:
        return 4

    if (b - r) >= 60:
        if b >= 202:
            return 1
        if b >= 149:
            return 2
        return 3

    return 0


def _intensity_bucket_array(rgba_array):
    """Vectorized version of _intensity_bucket over an HxWx4 uint8 array.

    Returns an HxW uint8 array of 0-4 buckets.
    """
    r = rgba_array[..., 0].astype(np.int16)
    b = rgba_array[..., 2].astype(np.int16)
    a = rgba_array[..., 3].astype(np.int16)

    visible = a >= 32
    extreme = visible & (r >= 200) & (b <= 20)
    blue = visible & ~extreme & ((b - r) >= 60)

    bucket = np.zeros(r.shape, dtype=np.uint8)
    bucket = np.where(blue & (b < 149), 3, bucket)
    bucket = np.where(blue & (b >= 149) & (b < 202), 2, bucket)
    bucket = np.where(blue & (b >= 202), 1, bucket)
    bucket = np.where(extreme, 4, bucket)
    return bucket


def _ring_masks(center_x, center_y, meters_per_pixel):
    """Boolean masks over the STITCH_SIZE x STITCH_SIZE canvas for each
    (near_km, far_km] ring in _RING_BOUNDS_KM, centred on the sensor's pixel."""
    yy, xx = np.mgrid[0:STITCH_SIZE, 0:STITCH_SIZE]
    dist_km = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) * meters_per_pixel / 1000.0

    masks = {}
    for label, near_km, far_km in _RING_BOUNDS_KM:
        masks[label] = (dist_km > near_km) & (dist_km <= far_km)
    return masks


def get_radar_signal():
    """Main entry point. Never raises — returns {"available": False} on any failure.

    On success returns:
      {
        "available": True,
        "point_intensity": <int 0-4, latest frame's bucket at the sensor's exact pixel>,
        "intensity_0_10km": <int 0-4, latest frame's max bucket within 10km>,
        "intensity_10_25km": <int 0-4, latest frame's max bucket 10-25km out>,
        "intensity_25_50km": <int 0-4, latest frame's max bucket 25-50km out>,
        "intensity_50_100km": <int 0-4, latest frame's max bucket 50-100km out>,
        "close_max_intensity": <int 0-4, max of the 0-10km and 10-25km rings —
            use this (not nearby_max_intensity) for anything that means
            "is there rain actually close by">,
        "trend_rising": <bool, based on the close_max_intensity sequence>,
        "nearby_max_intensity": <deprecated — max bucket over just the home
            tile, same box as the pre-2026-07-21 single-tile version, kept
            only so in-flight HA history graphs don't break>,
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

        # Sensor's pixel within the stitched STITCH_SIZE canvas: the home
        # tile sits at the center cell of the 3x3 grid.
        half_tiles = STITCH_TILES_PER_SIDE // 2
        center_x = half_tiles * TILE_SIZE + pixel_x
        center_y = half_tiles * TILE_SIZE + pixel_y
        meters_per_pixel = _meters_per_pixel(lat)
        ring_masks = _ring_masks(center_x, center_y, meters_per_pixel)

        sample_frames = past_frames[-_MAX_SAMPLE_FRAMES:]

        # Prune the decoded-tile cache against the FULL past_frames list
        # (everything RainViewer currently serves, ~13 frames), not just
        # this call's own _MAX_SAMPLE_FRAMES subset. radar_advection.py
        # shares this same cache and samples a different-sized tail of the
        # same past_frames list (see its _MOTION_FRAMES) — pruning to only
        # this function's own subset would evict entries the other consumer
        # still needs and vice versa, causing the two callers to repeatedly
        # evict and re-fetch each other's tiles instead of sharing a warm
        # cache. RainViewer's own retention already bounds this (~13 frames
        # * 9 tiles =~ 30MB), so pruning to the wider set is still bounded.
        wanted_frame_times = {frame["time"] for frame in past_frames}
        for key in list(_tile_array_cache.keys()):
            if key[0] not in wanted_frame_times:
                del _tile_array_cache[key]

        close_ring_sequence = []
        ring_values_latest = {}
        point_intensity_latest = 0
        nearby_max_latest = 0
        frame_time_latest = None

        for frame in sample_frames:
            canvas = _stitch_frame(host, frame, tile_x, tile_y)
            bucket_array = _intensity_bucket_array(canvas)

            point_intensity = int(bucket_array[center_y, center_x])

            ring_values = {}
            for label, mask in ring_masks.items():
                ring_pixels = bucket_array[mask]
                ring_values[label] = int(ring_pixels.max()) if ring_pixels.size else 0

            close_max = max(ring_values["intensity_0_10km"], ring_values["intensity_10_25km"])
            close_ring_sequence.append(close_max)

            # Backward-compat field: max over just the home (center) tile,
            # matching the pre-fix single-tile fetch exactly.
            home_tile = bucket_array[
                half_tiles * TILE_SIZE:(half_tiles + 1) * TILE_SIZE,
                half_tiles * TILE_SIZE:(half_tiles + 1) * TILE_SIZE,
            ]
            nearby_max_latest = int(home_tile.max())

            point_intensity_latest = point_intensity
            ring_values_latest = ring_values
            frame_time_latest = frame["time"]

        trend_rising = bool(
            len(close_ring_sequence) >= 2
            and all(
                close_ring_sequence[i] <= close_ring_sequence[i + 1]
                for i in range(len(close_ring_sequence) - 1)
            )
            and close_ring_sequence[-1] > 0
            and close_ring_sequence[-1] > close_ring_sequence[0]
        )

        result = {
            "available": True,
            "point_intensity": point_intensity_latest,
            **ring_values_latest,
            "close_max_intensity": max(
                ring_values_latest["intensity_0_10km"], ring_values_latest["intensity_10_25km"]
            ),
            "trend_rising": trend_rising,
            "nearby_max_intensity": nearby_max_latest,
            "frame_time": frame_time_latest,
        }
        _signal_cache["result"] = result
        _signal_cache["computed_at"] = now
        return result
    except Exception:
        logger.exception("Radar nowcast signal computation failed")
        return {"available": False}
