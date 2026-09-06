"""Fit an out-of-sample NWP temperature interval artifact from MariaDB.

Example (read-only DB access, new output path):
  python train_temperature_intervals.py --output artifacts/nwp-temperature-interval-YYYYMMDD.json

The last chronological fraction is held out for a coverage/width report. The
runtime artifact contains only residual quantiles learned from the earlier
window, so it cannot leak the holdout into the interval bounds.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import temp_interval
import weather_db
import nwp_bias


def _parse_time(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def build_pairs(nwp_rows, observation_rows, tolerance_minutes=8, bias_model=None):
    nwp = pd.DataFrame(nwp_rows)
    obs = pd.DataFrame(observation_rows)
    if nwp.empty or obs.empty:
        return pd.DataFrame(columns=["valid_at", "lead_hours", "error_c"])
    nwp["valid_at"] = _parse_time(nwp["valid_at"])
    obs["reading_time"] = _parse_time(obs["reading_time"])
    nwp["temp_forecast"] = pd.to_numeric(nwp["temp_forecast"], errors="coerce")
    obs["temp"] = pd.to_numeric(obs["temp"], errors="coerce")
    nwp = nwp.dropna(subset=["valid_at", "temp_forecast"]).sort_values("valid_at")
    obs = obs.dropna(subset=["reading_time", "temp"]).sort_values("reading_time")
    joined = pd.merge_asof(
        nwp,
        obs,
        left_on="valid_at",
        right_on="reading_time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    ).dropna(subset=["temp"])
    if bias_model:
        def corrected(row):
            issued = _parse_time(row["issued_hour"])
            correction = nwp_bias.correction_for(
                bias_model, row["lead_hours"], issued.hour if pd.notna(issued) else None
            )
            return row["temp_forecast"] + correction
        joined["center_temp"] = joined.apply(corrected, axis=1)
    else:
        joined["center_temp"] = joined["temp_forecast"]
    joined["error_c"] = joined["temp"] - joined["center_temp"]
    joined["issue_hour"] = joined["issued_hour"].map(lambda value: _parse_time(value).hour
                                                       if pd.notna(_parse_time(value)) else None)
    return joined[["valid_at", "lead_hours", "issue_hour", "error_c"]].sort_values("valid_at")


def fit_and_evaluate(pairs, holdout_fraction=0.20, coverage=0.80, min_samples=30,
                     center_postprocessing="none"):
    if len(pairs) < max(min_samples * 2, 20):
        raise ValueError(f"Need at least {max(min_samples * 2, 20)} paired rows; got {len(pairs)}")
    cutoff = max(1, min(len(pairs) - 1, int(len(pairs) * (1.0 - holdout_fraction))))
    train = pairs.iloc[:cutoff]
    holdout = pairs.iloc[cutoff:]
    model = temp_interval.fit_interval_model(
        train[["lead_hours", "error_c"]].to_dict("records"),
        coverage=coverage,
        min_samples=min_samples,
    )
    lower, upper = [], []
    for row in holdout.itertuples():
        interval = temp_interval.interval_for(model, row.lead_hours)
        if interval is None:
            interval = temp_interval.interval_for(model, 0)
        if interval is None:
            continue
        lower.append(row.error_c >= interval["lower_error_c"])
        upper.append(row.error_c <= interval["upper_error_c"])
    covered = [a and b for a, b in zip(lower, upper)]
    widths = []
    for lead in holdout["lead_hours"]:
        interval = temp_interval.interval_for(model, lead) or temp_interval.interval_for(model, 0)
        if interval:
            widths.append(interval["upper_error_c"] - interval["lower_error_c"])
    model["fit_window"] = {
        "start": train.valid_at.iloc[0].isoformat(),
        "end": train.valid_at.iloc[-1].isoformat(),
        "paired_rows": int(len(train)),
    }
    model["center_postprocessing"] = center_postprocessing
    model["holdout"] = {
        "start": holdout.valid_at.iloc[0].isoformat(),
        "end": holdout.valid_at.iloc[-1].isoformat(),
        "paired_rows": int(len(holdout)),
        "interval_rows": len(covered),
        "coverage": float(np.mean(covered)) if covered else None,
        "target_coverage": float(coverage),
        "mean_width_c": float(np.mean(widths)) if widths else None,
    }
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New JSON path; refuses to overwrite")
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--tolerance-minutes", type=int, default=8)
    parser.add_argument("--bias-model", default=None,
                        help="Optional nwp_bias JSON; fit residuals around its corrected centre")
    args = parser.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing artifact: {destination}")
    bias_model = nwp_bias.load_bias_model(args.bias_model) if args.bias_model else None
    pairs = build_pairs(weather_db.get_nwp_forecast_rows(), weather_db.get_temperature_observations(),
                        tolerance_minutes=args.tolerance_minutes, bias_model=bias_model)
    model = fit_and_evaluate(
        pairs, args.holdout_fraction, args.coverage, args.min_samples,
        center_postprocessing="nwp_bias_correction" if bias_model else "none",
    )
    model["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    model["source"] = "nwp_forecast_log + weather_readings"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(destination.resolve()), "pairs": len(pairs), "holdout": model["holdout"]}, indent=2))


if __name__ == "__main__":
    main()
