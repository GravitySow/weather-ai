"""Radar motion extrapolation — arrival ETA nowcast. SHADOW MODE ONLY.

Builds on radar_nowcast.py's tile-stitching (see its 2026-07-21 geometry-fix
note) to estimate how fast and in what direction nearby rain is moving, then
projects the current rain field forward along that motion vector to answer
"when (if ever) does rain reach the sensor" — the genuine advance-warning
capability the local ML model provably cannot provide. See plan.md's "Hard
constraints": restricting the rain model's test set to dry-antecedent rows
(no rain in the last 60min) drops recall to 0.000 at every horizon — it can
only ever say "rain that's already happening will continue."

Method: frozen-turbulence advection nowcasting — assume the current rain
field translates at a constant velocity without changing shape or
intensity. This is the simplest form of what pysteps/TITAN-style nowcasters
do, without their optical-flow, rotation, or growth/decay terms. Chosen
deliberately over anything fancier, per plan.md Phase 2: "numpy only to
start... pysteps/opencv heavy for a Pi 4 — only consider if the numpy
cross-correlation proves inadequate."

Motion estimation: a bounded brute-force overlap search between consecutive
frames' binary rain masks, cropped to a window around the sensor (not the
full stitched canvas — an unrelated storm on the far side of the ~300km
stitched box moving independently would corrupt a whole-canvas
correlation). This is coarser than real optical flow but matches the
precision the rest of this project's radar signal already operates at (0-4
intensity buckets, not calibrated mm/hr).

SHADOW MODE (2026-07-21): published to Home Assistant only, and logged to
forecast_log (source_layer="radar_advection") so forecast_scoring.py can
backtest it — NOT wired into any Telegram alert or notify_rain_state_change
logic. Do not wire this into Telegram until that backtest shows it beats
climatology; the original (unvalidated, geometrically-broken) radar
intensity alert should have followed this discipline from the start and
didn't — see plan.md's Phase 5 rationale.
"""

import logging
import time

import numpy as np

import radar_nowcast
from nwp_forecast import WEATHER_LAT, WEATHER_LON

logger = logging.getLogger(__name__)

# Same 5-minute cache cadence as radar_nowcast.py's own signal — no point
# recomputing faster than RainViewer's own ~10-minute frame cadence.
_CACHE_TTL_SECONDS = 5 * 60

# How many of the most recent frames to use for motion estimation. 4 frames
# = 3 consecutive pairs to average, keeping the vector stable against one
# noisy pair without needing much more history (RainViewer frames are
# ~10min apart, so 4 frames already spans ~30min).
_MOTION_FRAMES = 4

# Crop the stitched canvas to this window (px) centered on the sensor
# before searching for the motion vector. 300px =~ 357km at this
# latitude/zoom (see radar_nowcast._meters_per_pixel) — generous for
# tracking the local system without picking up unrelated weather far away.
_MOTION_WINDOW_PX = 300

# Bounded shift search range, same reasoning as radar_nowcast.py: at
# ~10min/frame and ~1.19km/px here, +-30px allows tracking storms up to
# ~213km/h — "well beyond any real storm" (plan.md Phase 2), so hitting
# this bound signals a bad correlation, not an actually-that-fast storm.
_MAX_SHIFT_PX = 30
_SHIFT_STEP_PX = 2

# How far ahead (minutes) to project the advected field looking for
# arrival, and the step size to check at.
_MAX_LOOKAHEAD_MINUTES = 120
_LOOKAHEAD_STEP_MINUTES = 5

# Only trust a source/destination pixel as "rain" at bucket >= this.
_RAIN_BUCKET_THRESHOLD = 1

_signal_cache = {"result": None, "computed_at": 0.0}


def _crop_window(array, center_x, center_y, half_size):
    h, w = array.shape[:2]
    y0, y1 = max(0, center_y - half_size), min(h, center_y + half_size)
    x0, x1 = max(0, center_x - half_size), min(w, center_x + half_size)
    return array[y0:y1, x0:x1], (x0, y0)


def _overlap_score(prev_mask, curr_mask, dx, dy):
    """Count of pixels where prev_mask, shifted by (dx,dy), agrees with
    curr_mask — i.e. how well "the rain that was at (y,x) moved to
    (y+dy,x+dx)" explains what we actually see, over only the region where
    both are defined (no wraparound)."""
    h, w = prev_mask.shape
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    if y0 >= y1 or x0 >= x1:
        return 0
    prev_region = prev_mask[y0:y1, x0:x1]
    curr_region = curr_mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return int(np.sum(prev_region & curr_region))


def _best_shift(prev_mask, curr_mask):
    """Returns ((dx, dy), no_rain). dx/dy in px, +dx=east, +dy=south.
    no_rain=True if either frame has nothing to track (skip this pair)."""
    if not prev_mask.any() or not curr_mask.any():
        return (0, 0), True

    best_score = _overlap_score(prev_mask, curr_mask, 0, 0)
    best_shift = (0, 0)
    for dy in range(-_MAX_SHIFT_PX, _MAX_SHIFT_PX + 1, _SHIFT_STEP_PX):
        for dx in range(-_MAX_SHIFT_PX, _MAX_SHIFT_PX + 1, _SHIFT_STEP_PX):
            score = _overlap_score(prev_mask, curr_mask, dx, dy)
            if score > best_score:
                best_score = score
                best_shift = (dx, dy)
    return best_shift, False


def _bearing_degrees(dx, dy):
    """Compass bearing (0=N, 90=E, ...) rain is moving toward. Image dy is
    south-positive, so "north" is -dy."""
    angle = np.degrees(np.arctan2(dx, -dy))
    return int(round(angle % 360))


def _empty_result(frame_time):
    return {
        "available": True,
        "rain_arriving": False,
        "eta_minutes": None,
        "expected_intensity": None,
        "motion_deg": None,
        "motion_kmh": None,
        "confidence": 0.0,
        "frame_time": frame_time,
    }


def get_advection_signal():
    """Main entry point. Never raises — returns {"available": False} on any
    failure, or a "rain_arriving": False result when there's nothing nearby
    to track.

    On success returns:
      {
        "available": True,
        "rain_arriving": <bool>,
        "eta_minutes": <int or None — None if the tracked motion doesn't
            bring rain to the sensor within _MAX_LOOKAHEAD_MINUTES>,
        "expected_intensity": <int 0-4, bucket at the projected arrival, or
            None if rain_arriving is False>,
        "motion_deg": <int 0-359 compass bearing, or None>,
        "motion_kmh": <float, or None>,
        "confidence": <float 0-1 — how consistent the sampled frame pairs'
            motion vectors were with each other, scaled by how many pairs
            had trackable rain at all. Coarse, not a calibrated probability>,
        "frame_time": <unix ts of latest frame used>,
      }
    """
    try:
        now = time.monotonic()
        cached = _signal_cache["result"]
        if cached is not None and (now - _signal_cache["computed_at"]) < _CACHE_TTL_SECONDS:
            return cached

        if not WEATHER_LAT or not WEATHER_LON:
            logger.warning("WEATHER_LAT/WEATHER_LON not set; skipping radar advection")
            return {"available": False}

        host, past_frames = radar_nowcast._fetch_frame_list()
        if not host or not past_frames:
            return {"available": False}

        lat = float(WEATHER_LAT)
        lon = float(WEATHER_LON)
        tile_x, tile_y, pixel_x, pixel_y = radar_nowcast._latlon_to_tile_pixel(
            lat, lon, radar_nowcast.ZOOM
        )
        half_tiles = radar_nowcast.STITCH_TILES_PER_SIDE // 2
        center_x = half_tiles * radar_nowcast.TILE_SIZE + pixel_x
        center_y = half_tiles * radar_nowcast.TILE_SIZE + pixel_y
        meters_per_pixel = radar_nowcast._meters_per_pixel(lat)

        sample_frames = past_frames[-_MOTION_FRAMES:]
        if len(sample_frames) < 2:
            return {"available": False}

        half_window = _MOTION_WINDOW_PX // 2
        bucket_windows = []
        frame_times = []
        window_origin = (0, 0)

        for frame in sample_frames:
            canvas = radar_nowcast._stitch_frame(host, frame, tile_x, tile_y)
            bucket_array = radar_nowcast._intensity_bucket_array(canvas)
            window, window_origin = _crop_window(bucket_array, center_x, center_y, half_window)
            bucket_windows.append(window)
            frame_times.append(frame["time"])

        masks = [window >= _RAIN_BUCKET_THRESHOLD for window in bucket_windows]
        latest_bucket_window = bucket_windows[-1]
        sensor_in_window = (center_x - window_origin[0], center_y - window_origin[1])

        shifts = []
        for i in range(len(masks) - 1):
            dt_minutes = (frame_times[i + 1] - frame_times[i]) / 60.0
            if dt_minutes <= 0:
                continue
            (dx, dy), no_rain = _best_shift(masks[i], masks[i + 1])
            if no_rain:
                continue
            shifts.append((dx / dt_minutes, dy / dt_minutes))  # px/minute

        if not shifts or not masks[-1].any():
            result = _empty_result(frame_times[-1])
            _signal_cache["result"] = result
            _signal_cache["computed_at"] = now
            return result

        vx_px_per_min = float(np.mean([s[0] for s in shifts]))
        vy_px_per_min = float(np.mean([s[1] for s in shifts]))
        speed_px_per_min = float(np.hypot(vx_px_per_min, vy_px_per_min))

        # Confidence: how consistent the individual pairwise vectors are
        # with each other (low spread relative to speed = high confidence),
        # scaled by how many of the sampled pairs actually had trackable
        # rain at all. Coarse by construction — see module docstring.
        if len(shifts) >= 2:
            spread = float(np.std([s[0] for s in shifts])) + float(np.std([s[1] for s in shifts]))
            consistency = max(0.0, 1.0 - spread / max(speed_px_per_min, 1e-6))
        else:
            consistency = 0.5  # only one usable pair -- can't judge consistency
        coverage = len(shifts) / (len(masks) - 1)
        confidence = round(max(0.0, min(1.0, consistency * coverage)), 2)

        motion_kmh = speed_px_per_min * meters_per_pixel / 1000.0 * 60.0
        motion_deg = _bearing_degrees(vx_px_per_min, vy_px_per_min)

        sensor_x, sensor_y = sensor_in_window
        eta_minutes = None
        expected_intensity = None

        if speed_px_per_min > 1e-6:
            window_h, window_w = latest_bucket_window.shape
            for t in range(0, _MAX_LOOKAHEAD_MINUTES + 1, _LOOKAHEAD_STEP_MINUTES):
                # Frozen-turbulence extrapolation: the field t minutes from
                # now at the sensor equals the LATEST observed field at
                # (sensor - velocity*t) — i.e. "what's currently upstream,
                # t minutes' travel-time away".
                src_x = int(round(sensor_x - vx_px_per_min * t))
                src_y = int(round(sensor_y - vy_px_per_min * t))
                if not (0 <= src_y < window_h and 0 <= src_x < window_w):
                    break  # motion carried the lookup off the edge of our window
                bucket = int(latest_bucket_window[src_y, src_x])
                if bucket >= _RAIN_BUCKET_THRESHOLD:
                    eta_minutes = t
                    expected_intensity = bucket
                    break

        result = {
            "available": True,
            "rain_arriving": eta_minutes is not None,
            "eta_minutes": eta_minutes,
            "expected_intensity": expected_intensity,
            "motion_deg": motion_deg,
            "motion_kmh": round(motion_kmh, 1),
            "confidence": confidence,
            "frame_time": frame_times[-1],
        }
        _signal_cache["result"] = result
        _signal_cache["computed_at"] = now
        return result
    except Exception:
        logger.exception("Radar advection signal computation failed")
        return {"available": False}
