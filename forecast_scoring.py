"""Nightly scorer for forecast_log — see plan.md "Phase 5".

Compares each logged forecast's rain_prob against the event the source actually
predicts.  The local RF predicts whether rain occurs *any time in its next
lead-time window*, whereas radar-advection makes a point-in-time ETA claim.
The scorer keeps those semantics separate rather than treating every source as
an instantaneous forecast.

This exists specifically so future signals are measurable from day one
instead of shipping unvalidated — see plan.md's "Hard constraints" section:
the existing radar alert has never been backtested, and its geometry bug
(fixed 2026-07-21, see radar_nowcast.py) went unnoticed for weeks because
nothing was scoring it. A signal that can't beat climatology should be
pulled, not shipped; this module is what makes that call possible instead
of a guess.

Built ahead of the Phase 2 radar-advection work: any new source_layer value
written to forecast_log (e.g. "radar_advection") is picked up automatically
by score_recent_forecasts() with no code changes needed here.
"""

import logging
import threading
import time
from datetime import datetime, timedelta

import numpy as np

import weather_db

logger = logging.getLogger(__name__)

BANGKOK_OFFSET = timedelta(hours=7)

# Run once daily, after the day's readings are in and forecasts have had
# time to mature (a 30-minute-lead forecast issued at 23:45 hasn't matured
# yet at midnight).
SCORE_HOUR = 23
SCORE_MINUTE = 30

# Score forecasts issued in roughly the last day, each run — keeps every
# nightly run's cost flat regardless of how much history has accumulated.
SCORE_LOOKBACK_DAYS = 1
_MAX_SCORABLE_LEAD_MINUTES = 120

# A ground-truth reading must land within this many seconds of a forecast's
# valid_at to count as a match (readings arrive ~1/min).
_MATCH_TOLERANCE_SECONDS = 90
_WINDOW_MATCH_TOLERANCE_SECONDS = 30

# The classifier was trained with this event definition.  Do not use a
# different threshold during live verification or probability calibration.
RAIN_THRESHOLD = 0.10

# `local_rf` is trained on max(rain_flag[t+1:t+lead]), i.e. rain occurring at
# any point in the next lead-minute window.  All other layers retain the
# historic point-in-time semantics; in particular radar_advection's ETA is a
# claim about rain at its projected arrival time, not about a whole interval.
_WINDOW_EVENT_SOURCE_LAYERS = frozenset({"local_rf", "local_xgb"})

# Hourly probabilities for the same *next-N-minute* event as the local
# classifiers.  They were derived only from each horizon's chronological TRAIN
# partition in dataset/*.csv, never from calibration/validation/test rows:
#
#   df, group, features = train_weather_ai.load_features("dataset")
#   future = train_weather_ai.build_future_rain_target(group, horizon)
#   frame = df.assign(target=(future > 0.10).astype(float).where(future.notna()))
#   train = train_weather_ai.chronological_partitions(
#       frame.dropna(subset=[*features, "target"]), horizon)["train"]
#   rates = train.groupby(train.timestamp.dt.hour).target.mean()
#
# The hour keys are the **UTC issued/observation hour**, matching the timestamp
# of the feature row on which the target starts — not Bangkok valid-at hour.
# Provenance after cadence validation and label purging:
#   5m:  43,300 train rows, 2026-04-27T20:37:53.928001+00:00 through
#        2026-06-12T00:15:08.999845+00:00
#   10m: 42,920 train rows, 2026-04-27T20:37:53.928001+00:00 through
#        2026-06-11T23:15:08.894477+00:00
#   30m: 41,476 train rows, 2026-04-27T20:37:53.928001+00:00 through
#        2026-06-11T16:16:07.964213+00:00
# A horizon/hour without a measured rate is deliberately unscorable; do not
# fall back to the point-event climatology below.
_WINDOW_HOURLY_RAIN_RATE = {
    5: {
        0: 0.011284256, 1: 0.043341709, 2: 0.074825986, 3: 0.054939516,
        4: 0.056678167, 5: 0.082951654, 6: 0.071515892, 7: 0.075183374,
        8: 0.079112123, 9: 0.134380454, 10: 0.216201999, 11: 0.183925811,
        12: 0.116118770, 13: 0.113017751, 14: 0.051656373, 15: 0.038974359,
        16: 0.034197463, 17: 0.054809843, 18: 0.050245767, 19: 0.019596865,
        20: 0.061041293, 21: 0.054802956, 22: 0.017462165, 23: 0.027764006,
    },
    10: {
        0: 0.011570248, 1: 0.053265694, 2: 0.082946636, 3: 0.057099545,
        4: 0.068675889, 5: 0.095584416, 6: 0.078105781, 7: 0.085223789,
        8: 0.085204375, 9: 0.161610268, 10: 0.235325225, 11: 0.196271362,
        12: 0.126145553, 13: 0.113622844, 14: 0.051656373, 15: 0.041968912,
        16: 0.038375973, 17: 0.060572070, 18: 0.060406370, 19: 0.023917995,
        20: 0.064613527, 21: 0.056381660, 22: 0.020551967, 23: 0.018800813,
    },
    30: {
        0: 0.016345593, 1: 0.091623037, 2: 0.108468677, 3: 0.065228557,
        4: 0.119238477, 5: 0.132768362, 6: 0.102435978, 7: 0.119949495,
        8: 0.105106888, 9: 0.258559622, 10: 0.276086957, 11: 0.217185029,
        12: 0.151129126, 13: 0.114990969, 14: 0.054988662, 15: 0.051419389,
        16: 0.040276180, 17: 0.081111741, 18: 0.100564972, 19: 0.037679426,
        20: 0.039499671, 21: 0.038612565, 22: 0.000000000, 23: 0.021544929,
    },
}

# Coarse hour-of-day rain-rate climatology, Bangkok local time, from the
# 2026-07-21 pressure/rain analysis (91,846 readings, Apr-Jul dataset/*.csv).
# Static rather than recomputed live: it only needs to be roughly right to
# be a fair baseline, and it captures the real diurnal swing (0.9% at 05:00
# vs 18% at 18:00) that a single flat "7.5% overall" baseline would hide —
# scoring against the flat rate would flatter any signal that merely knows
# "it's usually not raining".
_HOURLY_RAIN_RATE = {
    0: 0.083, 1: 0.084, 2: 0.081, 3: 0.076, 4: 0.036, 5: 0.009,
    6: 0.036, 7: 0.024, 8: 0.034, 9: 0.037, 10: 0.030, 11: 0.035,
    12: 0.054, 13: 0.058, 14: 0.048, 15: 0.055, 16: 0.085, 17: 0.145,
    18: 0.180, 19: 0.162, 20: 0.137, 21: 0.114, 22: 0.100, 23: 0.097,
}


def _bangkok_now():
    return datetime.utcnow() + BANGKOK_OFFSET


def _climatology_rate(valid_at_utc):
    valid_at_local = valid_at_utc + BANGKOK_OFFSET
    return _HOURLY_RAIN_RATE[valid_at_local.hour]


def _window_climatology_rate(issued_at_utc, lead_minutes):
    """Return a measured UTC-issued-hour window rate, if one exists."""
    return _WINDOW_HOURLY_RAIN_RATE.get(int(lead_minutes), {}).get(issued_at_utc.hour)


def _build_reading_index(readings):
    """Bucket readings by minute for O(1) approximate lookup instead of a
    linear scan per forecast (a day's worth of 3-horizon-per-minute
    forecasts against a day's worth of readings is ~6M comparisons the
    naive way; this keeps it to O(n+m))."""
    index = {}
    for row in readings:
        bucket = row["reading_time"].replace(second=0, microsecond=0)
        index.setdefault(bucket, []).append(row)
    return index


def _match_ground_truth(reading_index, valid_at):
    """Return the point-event label nearest ``valid_at`` or ``None``.

    This is intentionally kept for radar ETA and any future point forecast.
    """
    base = valid_at.replace(second=0, microsecond=0)
    candidates = []
    for delta_min in (0, -1, 1):
        candidates.extend(reading_index.get(base + timedelta(minutes=delta_min), []))

    best, best_delta = None, None
    for row in candidates:
        delta = abs((row["reading_time"] - valid_at).total_seconds())
        if best_delta is None or delta < best_delta:
            best, best_delta = row, delta

    if best is None or best_delta > _MATCH_TOLERANCE_SECONDS:
        return None
    return float(best["rain_flag"]) > RAIN_THRESHOLD


def _window_readings(reading_index, issued_at, valid_at):
    """Return readings in ``(issued_at, valid_at]``, ordered by time.

    The minute-bucket index keeps this bounded to the forecast horizon rather
    than scanning a whole day's observations for every forecast.
    """
    first_bucket = issued_at.replace(second=0, microsecond=0)
    # Match the same +/-30s cadence jitter allowed by training targets. A
    # reading at +5m+10ms is still the fifth sample, not missing coverage.
    window_end = valid_at + timedelta(seconds=_WINDOW_MATCH_TOLERANCE_SECONDS)
    last_bucket = window_end.replace(second=0, microsecond=0)
    readings = []
    bucket = first_bucket
    while bucket <= last_bucket:
        readings.extend(reading_index.get(bucket, []))
        bucket += timedelta(minutes=1)
    return sorted(
        (row for row in readings if issued_at < row["reading_time"] <= window_end),
        key=lambda row: row["reading_time"],
    )


def _match_window_ground_truth(reading_index, issued_at, valid_at, lead_minutes):
    """Return the local-RF window-event label, or ``None`` if coverage is thin.

    The training target is the maximum rain flag in the next ``lead_minutes``
    one-minute readings.  For scoring, every expected minute must have a
    distinct nearby observation.  Without that guard, a missing observation
    (including the only rainy minute) could be silently scored as a dry
    forecast and make the model look better or worse by accident.
    """
    if lead_minutes <= 0 or valid_at <= issued_at:
        return None

    readings = _window_readings(reading_index, issued_at, valid_at)
    used_indexes = set()
    matched = []

    # A one-to-one match matters: one reading must not be reused to fill two
    # adjacent expected minutes when an ingestion gap occurred.
    for minute in range(1, lead_minutes + 1):
        expected_at = issued_at + timedelta(minutes=minute)
        candidates = [
            (index, row)
            for index, row in enumerate(readings)
            if index not in used_indexes
            and abs((row["reading_time"] - expected_at).total_seconds())
            <= _WINDOW_MATCH_TOLERANCE_SECONDS
        ]
        if not candidates:
            return None
        index, row = min(
            candidates,
            key=lambda item: abs(
                (item[1]["reading_time"] - expected_at).total_seconds()
            ),
        )
        used_indexes.add(index)
        matched.append(row)

    return any(float(row["rain_flag"]) > RAIN_THRESHOLD for row in matched)


def score_recent_forecasts(lookback_days=SCORE_LOOKBACK_DAYS):
    """Scores forecast_log rows whose valid_at has already passed, issued in
    the trailing `lookback_days`. Groups with a measured event-matched
    climatology are written/upserted to forecast_score_log.  A future unknown
    local horizon remains in the returned/logged report but is not persisted
    with a fabricated baseline. Never raises — logs and returns [] on failure,
    matching this project's "auxiliary jobs must never take down anything
    else" posture."""
    now_utc = datetime.utcnow()
    # Score a full issue-time day only after its longest forecast has matured.
    # Filtering immature forecasts out of [now-1day, now) loses those rows
    # forever: by tomorrow's run they are already older than the lookup window.
    window_end = now_utc - timedelta(minutes=_MAX_SCORABLE_LEAD_MINUTES,
                                     seconds=_MATCH_TOLERANCE_SECONDS)
    window_start = window_end - timedelta(days=lookback_days)

    try:
        forecasts = weather_db.get_forecast_log(window_start, window_end)
    except Exception:
        logger.exception("Failed to fetch forecast_log for scoring")
        return []

    forecasts = [f for f in forecasts if f["valid_at"] <= now_utc]
    if not forecasts:
        logger.info("forecast_scoring: no matured forecasts in the last %d day(s)", lookback_days)
        return []

    try:
        readings = weather_db.get_readings(window_start - timedelta(minutes=5), now_utc)
    except Exception:
        logger.exception("Failed to fetch weather_readings for scoring")
        return []

    if not readings:
        logger.info("forecast_scoring: no readings available to score against")
        return []

    reading_index = _build_reading_index(readings)

    groups = {}
    for forecast in forecasts:
        key = (forecast["source_layer"], forecast["lead_minutes"])
        groups.setdefault(key, []).append(forecast)

    as_of_date = _bangkok_now().date()
    results = []

    for (source_layer, lead_minutes), group in groups.items():
        is_window_event = source_layer in _WINDOW_EVENT_SOURCE_LAYERS
        errors, clim_errors, predicted, observed = [], [], [], []
        climatology_complete = True

        for forecast in group:
            if is_window_event:
                actual = _match_window_ground_truth(
                    reading_index,
                    forecast["issued_at"],
                    forecast["valid_at"],
                    int(lead_minutes),
                )
            else:
                actual = _match_ground_truth(reading_index, forecast["valid_at"])
            if actual is None:
                continue
            prob = float(forecast["rain_prob"])
            actual_val = 1.0 if actual else 0.0
            errors.append((prob - actual_val) ** 2)
            if is_window_event:
                climatology_rate = _window_climatology_rate(
                    forecast["issued_at"], lead_minutes
                )
            else:
                climatology_rate = _climatology_rate(forecast["valid_at"])
            if climatology_rate is None:
                # Keep the Brier result but never calculate skill on a subset
                # of the group with a different denominator.
                climatology_complete = False
            else:
                clim_errors.append((climatology_rate - actual_val) ** 2)
            predicted.append(prob)
            observed.append(actual_val)

        if not errors:
            continue

        brier = float(np.mean(errors))
        # Use the source-matched baseline above.  Never transform the point
        # climatology into a window probability using an independence
        # assumption; unknown horizons intentionally have no skill score.
        climatology_brier = (
            float(np.mean(clim_errors))
            if climatology_complete and len(clim_errors) == len(errors)
            else None
        )
        skill_score = (
            1.0 - (brier / climatology_brier)
            if climatology_brier is not None and climatology_brier > 0
            else None
        )

        result = {
            "source_layer": source_layer,
            "lead_minutes": lead_minutes,
            "n": len(errors),
            "brier_score": brier,
            "climatology_brier": climatology_brier,
            "skill_score": skill_score,
            "mean_predicted": float(np.mean(predicted)),
            "mean_observed": float(np.mean(observed)),
        }
        results.append(result)

        logger.info(
            "forecast_score source=%s lead=%dmin n=%d brier=%.4f climatology_brier=%s "
            "skill=%s mean_predicted=%.3f mean_observed=%.3f",
            source_layer, lead_minutes, result["n"], brier,
            (
                f"{climatology_brier:.4f}"
                if climatology_brier is not None
                else "n/a (no measured event-matched climatology)"
            ),
            f"{skill_score:.3f}" if skill_score is not None else "n/a",
            result["mean_predicted"], result["mean_observed"],
        )

        # The current DB schema requires climatology_brier to be non-null.
        # Do not store a made-up sentinel merely to satisfy it; the returned
        # result/log still reports the valid Brier score and explicitly marks
        # skill unavailable.
        if climatology_brier is None:
            logger.info(
                "forecast_score source=%s lead=%dmin not persisted: event-matched climatology unavailable",
                source_layer,
                lead_minutes,
            )
        else:
            try:
                weather_db.insert_forecast_score(as_of_date, result)
            except Exception:
                logger.exception(
                    "Failed to persist forecast_score_log row for %s/%dmin", source_layer, lead_minutes
                )

    return results


def schedule_loop():
    """Sleeps until ~SCORE_HOUR:SCORE_MINUTE Bangkok time each day and scores
    the trailing day's matured forecasts. Mirrors telegram_bot.schedule_loop's
    shape but runs independently — scoring has nothing to do with whether
    Telegram is configured."""
    while True:
        now_bkk = _bangkok_now()
        run_at = now_bkk.replace(hour=SCORE_HOUR, minute=SCORE_MINUTE, second=0, microsecond=0)
        if run_at <= now_bkk:
            run_at += timedelta(days=1)
        time.sleep(max((run_at - now_bkk).total_seconds(), 1))

        try:
            score_recent_forecasts()
        except Exception:
            logger.exception("Nightly forecast scoring failed")
