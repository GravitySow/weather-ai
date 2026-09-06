"""Conservative, experimental rain-stop estimate from observed rain events.

This is intentionally not the same target as ``rain within N minutes``.  A
stop is confirmed only by the first dry sample followed by ten uninterrupted
dry minutes.  Missing observations censor the event instead of silently
claiming that rain stopped.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

import weather_db

logger = logging.getLogger(__name__)

RAIN_THRESHOLD = 0.10
DRY_CONFIRM_MINUTES = 10
MAX_GAP_SECONDS = 150
LOOKBACK_DAYS = 30


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _parse_time(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def analyze_history(rows, now=None, dry_confirm_minutes=DRY_CONFIRM_MINUTES):
    """Return a JSON-safe rain-stop status from timestamped DB-like rows."""
    now = _parse_time(now) or datetime.now(timezone.utc).replace(tzinfo=None)
    parsed = []
    for row in rows or []:
        timestamp = _parse_time(row.get("reading_time", row.get("timestamp")))
        rain = _finite(row.get("rain_flag"))
        if timestamp is not None and rain is not None:
            parsed.append((timestamp, rain > RAIN_THRESHOLD))
    parsed.sort(key=lambda pair: pair[0])
    if not parsed:
        return {"status": "unavailable", "reason": "no_observations", "source": "sensor_history"}

    # Split on gaps. A gap can hide either the onset or the confirmed stop and
    # therefore must never be treated as a dry observation.
    segments = []
    segment = [parsed[0]]
    for item in parsed[1:]:
        if (item[0] - segment[-1][0]).total_seconds() > MAX_GAP_SECONDS:
            segments.append(segment)
            segment = []
        segment.append(item)
    segments.append(segment)

    completed = []
    current = None
    pending_stop = False
    censored_events = 0
    for segment in segments:
        active_start = None
        dry_run_start = None
        dry_count = 0
        for timestamp, wet in segment:
            if wet:
                if active_start is None:
                    active_start = timestamp
                dry_run_start = None
                dry_count = 0
                continue
            if active_start is None:
                continue
            if dry_run_start is None:
                dry_run_start = timestamp
            dry_count += 1
            if dry_count >= int(dry_confirm_minutes):
                duration = max(0.0, (dry_run_start - active_start).total_seconds() / 60.0)
                completed.append({
                    "start": active_start.isoformat() + "Z",
                    "stop_confirmed_at": dry_run_start.isoformat() + "Z",
                    "duration_minutes": round(duration, 2),
                })
                active_start = None
                dry_run_start = None
                dry_count = 0
        if active_start is not None:
            censored_events += 1
            # The latest segment is the only one that can describe the live
            # state. Older open segments were cut by a data gap.
            if segment is segments[-1]:
                if segment[-1][1]:
                    current = {
                        "start": active_start,
                        "last_observation": segment[-1][0],
                    }
                else:
                    pending_stop = True

    latest_timestamp, latest_wet = parsed[-1]
    if (now - latest_timestamp).total_seconds() > MAX_GAP_SECONDS:
        return {
            "status": "unavailable",
            "reason": "latest_observation_stale",
            "last_observation": latest_timestamp.isoformat() + "Z",
            "completed_events": len(completed),
            "censored_events": censored_events,
            "source": "sensor_history",
        }
    if not latest_wet and pending_stop:
        return {
            "status": "unavailable",
            "reason": "stop_not_confirmed",
            "last_observation": latest_timestamp.isoformat() + "Z",
            "completed_events": len(completed),
            "censored_events": censored_events,
            "source": "sensor_history",
        }
    if not latest_wet or current is None:
        return {
            "status": "not_raining",
            "last_observation": latest_timestamp.isoformat() + "Z",
            "completed_events": len(completed),
            "censored_events": censored_events,
            "source": "sensor_history",
        }
    if len(completed) < 3:
        return {
            "status": "unavailable",
            "reason": "insufficient_completed_events",
            "raining_since": current["start"].isoformat() + "Z",
            "elapsed_minutes": round((latest_timestamp - current["start"]).total_seconds() / 60.0, 2),
            "completed_events": len(completed),
            "censored_events": censored_events,
            "source": "sensor_history",
        }

    durations = [event["duration_minutes"] for event in completed]
    p20 = _percentile(durations, 20)
    p50 = _percentile(durations, 50)
    p80 = _percentile(durations, 80)
    elapsed = max(0.0, (latest_timestamp - current["start"]).total_seconds() / 60.0)
    remaining_lower = max(0.0, p20 - elapsed)
    remaining_median = max(0.0, p50 - elapsed)
    remaining_upper = max(remaining_median, p80 - elapsed)
    estimate_at = latest_timestamp + timedelta(minutes=remaining_median)

    return {
        "status": "experimental",
        "raining_since": current["start"].isoformat() + "Z",
        "last_observation": latest_timestamp.isoformat() + "Z",
        "elapsed_minutes": round(elapsed, 2),
        "estimated_stop_at": estimate_at.isoformat() + "Z",
        "remaining_minutes": {
            "lower": round(remaining_lower, 1),
            "median": round(remaining_median, 1),
            "upper": round(remaining_upper, 1),
        },
        "historical_duration_minutes": {
            "p20": round(p20, 1), "median": round(p50, 1), "p80": round(p80, 1),
        },
        "coverage": {"completed_events": len(completed), "censored_events": censored_events},
        "label_definition": f"first dry sample followed by {dry_confirm_minutes} uninterrupted dry minutes",
        "source": "sensor_history",
    }


def get_rain_stop_forecast(lookback_days=LOOKBACK_DAYS):
    """Load recent observations and calculate a conservative live estimate."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        rows = weather_db.get_readings(now - timedelta(days=lookback_days), now + timedelta(seconds=1))
    except Exception:
        logger.exception("Could not load history for rain-stop estimate")
        return {"status": "unavailable", "reason": "history_query_failed", "source": "sensor_history"}
    result = analyze_history(rows, now=now)
    result["lookback_days"] = int(lookback_days)
    return result
