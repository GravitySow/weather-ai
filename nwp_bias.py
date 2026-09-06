"""Small, auditable MOS-style bias correction for hourly NWP temperatures.

The correction is intentionally opt-in.  A model is fitted from historical
``observed - forecast`` pairs and applied only when the API is configured with
``NWP_BIAS_CORRECTION_PATH``.  This keeps an unverified bias estimate out of
the default forecast while making the calibration path reproducible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median


MAX_CORRECTION_C = 6.0


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fit_bias_model(pairs, min_samples=50):
    """Fit robust per-lead and optional issue-hour corrections.

    ``pairs`` accepts mappings or tuples.  Mapping keys are
    ``lead_hours``, ``issue_hour`` and ``error_c`` (observed minus forecast),
    while tuples are ``(lead_hours, issue_hour, error_c)``.  Median errors are
    used so one corrupt sensor/NWP pair cannot dominate the correction.
    """
    grouped_lead = {}
    grouped_hour = {}
    all_errors = []
    for pair in pairs:
        if isinstance(pair, dict):
            lead = pair.get("lead_hours", pair.get("lead"))
            hour = pair.get("issue_hour", pair.get("hour"))
            error = pair.get("error_c", pair.get("error"))
        else:
            try:
                lead, hour, error = pair
            except (TypeError, ValueError):
                continue
        try:
            lead = int(lead)
        except (TypeError, ValueError):
            continue
        error = _finite(error)
        if error is None:
            continue
        hour = None if hour is None else int(hour) % 24
        grouped_lead.setdefault(lead, []).append(error)
        if hour is not None:
            grouped_hour.setdefault(hour, []).append(error)
        all_errors.append(error)

    if len(all_errors) < min_samples:
        raise ValueError(f"Need at least {min_samples} valid NWP error pairs; got {len(all_errors)}")

    def summarize(groups):
        result = {}
        for key, values in groups.items():
            if len(values) >= min_samples:
                result[str(key)] = {
                    "n": len(values),
                    "median_c": round(float(median(values)), 6),
                    "mean_c": round(sum(values) / len(values), 6),
                }
        return result

    return {
        "schema_version": 1,
        "method": "median_observed_minus_forecast",
        "min_samples": int(min_samples),
        "global": {"n": len(all_errors), "median_c": round(float(median(all_errors)), 6),
                   "mean_c": round(sum(all_errors) / len(all_errors), 6)},
        "by_lead_hours": summarize(grouped_lead),
        "by_issue_hour": summarize(grouped_hour),
    }


def load_bias_model(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_bias_model(model, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2), encoding="utf-8")


def correction_for(model, lead_hours, issue_hour=None):
    """Return a bounded correction in degrees C (observed minus forecast)."""
    lead = model.get("by_lead_hours", {}).get(str(int(lead_hours)))
    global_entry = model.get("global", {})
    correction = lead.get("median_c") if lead else global_entry.get("median_c", 0.0)
    # The lead estimate is the primary signal.  Hour-of-day is only used when
    # that lead has not reached the minimum sample count.
    if not lead and issue_hour is not None:
        hour = model.get("by_issue_hour", {}).get(str(int(issue_hour) % 24))
        if hour:
            correction = hour.get("median_c", correction)
    correction = _finite(correction) or 0.0
    return max(-MAX_CORRECTION_C, min(MAX_CORRECTION_C, correction))


def apply_to_timeline(timeline, model):
    """Return a copied timeline with raw and bias-corrected temperatures."""
    corrected = []
    for entry in timeline:
        item = dict(entry)
        raw = _finite(item.get("temp"))
        if raw is None:
            item["temp_raw"] = item.get("temp")
            item["temp_bias_correction"] = 0.0
            corrected.append(item)
            continue
        time_text = str(item.get("time", ""))
        try:
            hour = int(time_text[11:13])
        except (ValueError, IndexError):
            hour = None
        lead = item.get("lead_hours", 0)
        correction = correction_for(model, lead, hour)
        item["temp_raw"] = raw
        item["temp_bias_correction"] = correction
        item["temp"] = round(raw + correction, 3)
        corrected.append(item)
    return corrected
