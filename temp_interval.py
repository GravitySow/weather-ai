"""Calibrated temperature prediction intervals for the NWP timeline.

The runtime only uses an interval artifact produced from out-of-sample
observed-minus-forecast residuals.  Without that artifact it returns no
interval rather than presenting an arbitrary spread as calibrated confidence.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


DEFAULT_COVERAGE = 0.80
MIN_SAMPLES = 30


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percentile(values, percentile):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * percentile / 100.0
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    fraction = rank - lower
    return float(values[lower] + fraction * (values[upper] - values[lower]))


def fit_interval_model(pairs, coverage=DEFAULT_COVERAGE, min_samples=MIN_SAMPLES):
    """Fit asymmetric residual quantiles by lead hour.

    ``pairs`` accepts mappings with ``lead_hours``, ``error_c`` and optional
    ``valid_hour`` or tuples ``(lead_hours, error_c)``.  The caller is
    responsible for chronological train/calibration splitting; this function
    does not inspect future rows.
    """
    coverage = float(coverage)
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    groups = {}
    all_errors = []
    for pair in pairs:
        if isinstance(pair, dict):
            lead = pair.get("lead_hours", pair.get("lead"))
            error = pair.get("error_c", pair.get("error"))
        else:
            try:
                lead, error = pair[:2]
            except (TypeError, ValueError, IndexError):
                continue
        try:
            lead = int(lead)
        except (TypeError, ValueError):
            continue
        error = _finite(error)
        if error is None:
            continue
        groups.setdefault(lead, []).append(error)
        all_errors.append(error)
    if len(all_errors) < int(min_samples):
        raise ValueError(f"Need at least {min_samples} valid residuals; got {len(all_errors)}")

    low_q = (1.0 - coverage) * 50.0
    high_q = 100.0 - low_q

    def summarize(values):
        return {
            "n": len(values),
            "lower_error_c": round(_percentile(values, low_q), 4),
            "upper_error_c": round(_percentile(values, high_q), 4),
            "median_error_c": round(_percentile(values, 50), 4),
        }

    usable = {str(key): summarize(values) for key, values in groups.items()
              if len(values) >= int(min_samples)}
    return {
        "schema_version": 1,
        "method": "empirical_residual_quantiles",
        "coverage_target": coverage,
        "min_samples": int(min_samples),
        "global": summarize(all_errors),
        "by_lead_hours": usable,
    }


def load_model(path=None):
    path = path or os.getenv("NWP_TEMP_INTERVAL_PATH")
    if not path:
        return None
    try:
        model = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if model.get("schema_version") != 1 or "global" not in model:
        return None
    return model


def interval_for(model, lead_hours):
    if not model:
        return None
    entry = model.get("by_lead_hours", {}).get(str(int(lead_hours))) or model.get("global")
    if not entry:
        return None
    lower = _finite(entry.get("lower_error_c"))
    upper = _finite(entry.get("upper_error_c"))
    if lower is None or upper is None or lower > upper:
        return None
    return {
        "lower_error_c": lower,
        "upper_error_c": upper,
        "coverage_target": _finite(model.get("coverage_target")),
        "samples": int(entry.get("n", 0)),
        "method": model.get("method", "unknown"),
        "center_postprocessing": model.get("center_postprocessing", "none"),
    }


def apply_to_timeline(timeline, model=None):
    """Annotate entries with lower/upper temperature bounds."""
    model = model if model is not None else load_model()
    result = []
    for entry in timeline:
        item = dict(entry)
        center = _finite(item.get("temp"))
        lead = item.get("lead_hours", 0)
        interval = interval_for(model, lead) if center is not None else None
        if interval and interval.get("center_postprocessing") == "nwp_bias_correction" \
                and not item.get("temp_bias_correction_enabled"):
            # Residuals are tied to the corrected centre; applying them to a
            # raw forecast would double-count the station bias.
            interval = None
        if interval:
            item["temp_interval"] = {
                "lower": round(center + interval["lower_error_c"], 2),
                "upper": round(center + interval["upper_error_c"], 2),
                "coverage_target": interval["coverage_target"],
                "samples": interval["samples"],
                "method": interval["method"],
            }
        else:
            item["temp_interval"] = None
        result.append(item)
    return result
