"""Audit saved holdout probabilities as rain events and alert episodes, offline.

A storm onset is a wet observation (>0.10) after ``dry_minutes`` consecutive
dry minute samples in the same uninterrupted observation segment. The elapsed
dry interval must also equal that duration within 30 seconds. Rain after a
missing observation is therefore not assumed to be a new storm.

An onset is evaluable only when all N forecasts immediately before it exist
and have complete observed futures. A detection requires an above-threshold
forecast issued while dry, strictly before onset, whose next N observed
samples contain that onset. This is the training target's actual interval:
N minute samples with total duration N minutes +/-30 seconds. Lead time uses
the earliest *matching* forecast, never an earlier unrelated false warning.

Alerts separated by at most ``cooldown_minutes`` belong to one episode. An
unavailable forecast minute breaks an episode. Complete episodes also require
that much covered time before the first alert and after the last; boundary
episodes are exported as censored and excluded from episode rates. A false
episode means *none* of its alerts has observed rain in its forecast window.
Real rain that is not a new storm is counted separately, not as a false alarm.
Episodes are retrospective threshold groupings, not a replay of the notifier.

No thresholds are fitted here. Existing files are read only; the CLI writes a
new report directory and refuses to overwrite one that already exists.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_weather_ai import RAIN_THRESHOLD, build_future_rain_target, load_features


EVENT_COLUMNS = ["event_id", "onset", "evaluable", "detected", "first_matching_alert",
                 "lead_minutes", "matching_episode_ids"]
EPISODE_COLUMNS = ["episode_id", "first_alert", "last_alert", "alert_rows", "complete",
                   "left_censored", "right_censored", "observed_rain", "false_alarm",
                   "matched_event_ids"]


def _observations(history):
    frame = history[["timestamp", "rain_flag"]].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, format="mixed")
    frame["rain_flag"] = pd.to_numeric(frame.rain_flag, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().sort_values("timestamp")
    if frame.timestamp.duplicated().any():
        raise ValueError("Observation timestamps must be unique")
    frame = frame.reset_index(drop=True)
    gap = frame.timestamp.diff()
    frame["segment_id"] = (gap.lt(pd.Timedelta(seconds=30)) |
                           gap.gt(pd.Timedelta(seconds=90))).cumsum()
    frame["position"] = np.arange(len(frame))
    return frame


def _boolean_column(values, name):
    parsed = values.astype(str).str.lower().map({"true": True, "false": False,
                                               "1": True, "0": False})
    if parsed.isna().any():
        raise ValueError(f"{name} must contain booleans")
    return parsed.astype(bool)


def evaluate_events(history, predictions, horizon, threshold, dry_minutes=60,
                    cooldown_minutes=30, alarm_scope="dry_now"):
    """Return JSON-safe metrics, an onset table and an alert-episode table.

    ``history`` is the frame from ``load_features`` (timestamp/rain_flag are
    sufficient for tests). Optional saved dry flags are checked against it.
    Missing history/forecast windows are excluded, never filled with dry data.
    ``dry_60m`` restricts alerts to dry observations with 60 dry minute samples
    ending at issue time, matching the existing model's dry_60m cohort.
    """
    if horizon < 1 or int(horizon) != horizon or dry_minutes < 1 or int(dry_minutes) != dry_minutes:
        raise ValueError("horizon and dry_minutes must be positive integers")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if not np.isfinite(cooldown_minutes) or cooldown_minutes <= 0:
        raise ValueError("cooldown_minutes must be positive")
    if alarm_scope not in ("dry_now", "dry_60m"):
        raise ValueError("alarm_scope must be dry_now or dry_60m")
    horizon, dry_minutes = int(horizon), int(dry_minutes)
    frame = _observations(history)
    group = frame.groupby("segment_id", group_keys=False)
    future = build_future_rain_target(group, horizon)
    frame["future_observed"] = (future > RAIN_THRESHOLD).where(future.notna())
    frame["future_end"] = group.timestamp.shift(-horizon)
    frame["dry_now_actual"] = frame.rain_flag <= RAIN_THRESHOLD
    dry60 = group.rain_flag.transform(lambda s: s.rolling(60, min_periods=60).max())
    frame["dry_60m_actual"] = frame.dry_now_actual & (dry60 <= RAIN_THRESHOLD)
    before = group.rain_flag.transform(lambda s: s.shift(1).rolling(dry_minutes,
                                                       min_periods=dry_minutes).max())
    elapsed = frame.timestamp - group.timestamp.shift(dry_minutes)
    starts = frame.loc[(frame.rain_flag > RAIN_THRESHOLD) & (before <= RAIN_THRESHOLD) &
                       ((elapsed - pd.Timedelta(minutes=dry_minutes)).abs() <=
                        pd.Timedelta(seconds=30))]

    saved = predictions.copy()
    saved["timestamp"] = pd.to_datetime(saved.timestamp, utc=True, format="mixed")
    if saved.timestamp.isna().any() or saved.timestamp.duplicated().any():
        raise ValueError("Prediction timestamps must be present and unique")
    for name in ("probability", "observed"):
        saved[name] = pd.to_numeric(saved[name], errors="coerce")
    if not saved.probability.between(0, 1).all() or not saved.observed.isin([0, 1]).all():
        raise ValueError("Probabilities must be finite in [0, 1] and observed must be binary")
    saved = saved.sort_values("timestamp")
    joined = saved.merge(frame, on="timestamp", how="left", validate="one_to_one")
    valid = joined.future_observed.notna()
    usable = joined.loc[valid].copy().sort_values("timestamp").reset_index(drop=True)
    if not np.array_equal(usable.observed.astype(bool), usable.future_observed.astype(bool)):
        raise ValueError("Saved observed labels disagree with raw observation history")
    for name in ("dry_now", "dry_60m"):
        if name in usable and not np.array_equal(_boolean_column(usable[name], name),
                                                usable[f"{name}_actual"]):
            raise ValueError(f"Saved {name} flags disagree with raw observation history")

    # A missing prediction is an interruption even when sensor history exists.
    usable["block"] = (usable.position.diff().ne(1) |
                       usable.segment_id.diff().ne(0)).cumsum()
    usable["alert"] = usable[f"{alarm_scope}_actual"] & (usable.probability >= threshold)
    cooldown = pd.Timedelta(minutes=cooldown_minutes)
    episode_rows, alert_to_episode = [], {}
    for _, block in usable.groupby("block", sort=False):
        alerts = block.loc[block.alert]
        episode_groups = (alerts.timestamp.diff() > cooldown).cumsum()
        for _, alerts_in_episode in alerts.groupby(episode_groups, sort=False):
            first = alerts_in_episode.timestamp.iloc[0]
            last = alerts_in_episode.timestamp.iloc[-1]
            left = first - block.timestamp.iloc[0] < cooldown
            right = block.timestamp.iloc[-1] - last < cooldown
            episode_id = len(episode_rows) + 1
            rain = bool(alerts_in_episode.future_observed.astype(bool).any())
            episode_rows.append({"episode_id": episode_id, "first_alert": first.isoformat(),
                                 "last_alert": last.isoformat(), "alert_rows": len(alerts_in_episode),
                                 "complete": not (left or right), "left_censored": bool(left),
                                 "right_censored": bool(right), "observed_rain": rain,
                                 "false_alarm": not rain, "matched_event_ids": []})
            for position in alerts_in_episode.position:
                alert_to_episode[int(position)] = episode_id

    available = set(usable.position.astype(int))
    event_rows = []
    if not usable.empty:
        # Include the observable forecast tail, not arbitrary history outside
        # this holdout's span; incomplete onset windows remain visible in CSV.
        starts = starts.loc[(starts.timestamp > usable.timestamp.min()) &
                            (starts.timestamp <= usable.future_end.max())]
        for onset in starts.itertuples():
            prior = range(onset.position - horizon, onset.position)
            evaluable = all(position in available for position in prior)
            opportunities = usable.loc[usable.position.isin(prior) & usable.alert &
                                        (usable.timestamp < onset.timestamp) &
                                        (usable.future_end >= onset.timestamp) &
                                        (usable.segment_id == onset.segment_id)]
            detected = evaluable and not opportunities.empty
            matching_ids = sorted({alert_to_episode[int(position)]
                                   for position in opportunities.position}) if evaluable else []
            event_id = len(event_rows) + 1
            first = opportunities.timestamp.iloc[0] if detected else None
            event_rows.append({"event_id": event_id, "onset": onset.timestamp.isoformat(),
                               "evaluable": evaluable, "detected": detected,
                               "first_matching_alert": first.isoformat() if first is not None else None,
                               "lead_minutes": (onset.timestamp - first).total_seconds() / 60
                               if first is not None else None, "matching_episode_ids": matching_ids})
            for episode_id in matching_ids:
                episode_rows[episode_id - 1]["matched_event_ids"].append(event_id)

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    episodes = pd.DataFrame(episode_rows, columns=EPISODE_COLUMNS)
    evaluable_count = sum(row["evaluable"] for row in event_rows)
    detected_count = sum(row["detected"] for row in event_rows)
    complete = [row for row in episode_rows if row["complete"]]
    false_count = sum(row["false_alarm"] for row in complete)
    leads = [row["lead_minutes"] for row in event_rows if row["detected"]]
    summary = {
        "horizon_minutes": horizon, "threshold": float(threshold), "alarm_scope": alarm_scope,
        "dry_minutes_before_onset": dry_minutes, "episode_cooldown_minutes": cooldown_minutes,
        "input_forecast_rows": len(saved), "usable_forecast_rows": len(usable),
        "excluded_forecast_rows": int((~valid).sum()),
        "forecast_start": usable.timestamp.min().isoformat() if len(usable) else None,
        "forecast_end": usable.timestamp.max().isoformat() if len(usable) else None,
        "onsets_in_span": len(events), "evaluable_onsets": evaluable_count,
        "excluded_onsets_incomplete_coverage": len(events) - evaluable_count,
        "detected_onsets": detected_count,
        "onset_recall": detected_count / evaluable_count if evaluable_count else None,
        "alert_rows": int(usable.alert.sum()), "alert_episodes": len(episodes),
        "complete_alert_episodes": len(complete), "censored_alert_episodes": len(episodes) - len(complete),
        "false_alert_episodes": false_count,
        "false_alarm_ratio": false_count / len(complete) if complete else None,
        "rain_episodes_without_evaluable_new_onset": sum(
            row["observed_rain"] and not row["matched_event_ids"] for row in complete),
        "lead_minutes": {"n": len(leads), **{label: float(np.percentile(leads, percentile)) if leads else None
                          for label, percentile in [("min", 0), ("p25", 25), ("median", 50),
                                                    ("p75", 75), ("max", 100)]}},
    }
    return summary, events, episodes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", default="artifacts/rf-accuracy-20260906")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default=None, help="New directory only")
    parser.add_argument("--dry-minutes", type=int, default=60)
    parser.add_argument("--cooldown-minutes", type=float, default=30)
    parser.add_argument("--alarm-scope", choices=["dry_now", "dry_60m", "both"], default="both")
    args = parser.parse_args(argv)
    bundle = Path(args.bundle_dir)
    evaluation = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    kind = evaluation["model_kind"]
    history, _, _ = load_features(args.data_dir)
    outputs, results = [], []
    scopes = ["dry_now", "dry_60m"] if args.alarm_scope == "both" else [args.alarm_scope]
    for result in evaluation["results"]:
        horizon = result["horizon"]
        predictions = pd.read_csv(bundle / f"{kind}_holdout_predictions_{horizon}m.csv")
        for scope in scopes:
            summary, events, episodes = evaluate_events(
                history, predictions, horizon, result[f"{kind}_best_threshold"],
                args.dry_minutes, args.cooldown_minutes, scope)
            results.append(summary)
            outputs.extend([(f"events_{horizon}m_{scope}.csv", events),
                            (f"episodes_{horizon}m_{scope}.csv", episodes)])
    output = Path(args.output_dir or f"artifacts/event-review-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    for filename, table in outputs:
        table.to_csv(output / filename, index=False)
    report = {"schema_version": 1, "source_bundle": str(bundle.resolve()),
              "semantics": __doc__, "results": results}
    (output / "event_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Holdout rain-event review", "", __doc__.strip(), "",
             "| Horizon | Alert scope | Detected / evaluable onsets | Complete / all episodes | False episodes | Median lead (min) |",
             "|---|---|---|---|---|---|"]
    for result in results:
        median = result["lead_minutes"]["median"]
        lines.append(f"| {result['horizon_minutes']}m | {result['alarm_scope']} | "
                     f"{result['detected_onsets']} / {result['evaluable_onsets']} | "
                     f"{result['complete_alert_episodes']} / {result['alert_episodes']} | "
                     f"{result['false_alert_episodes']} | {median if median is not None else 'n/a'} |")
    lines.extend(["", "This is a diagnostic of saved held-out forecasts at their existing validation-selected thresholds. "
                  "It does not tune thresholds, demonstrate future deployment performance, or treat correlated "
                  "minute rows as independent storms. Horizon holdouts can have different time boundaries. "
                  "An empty lead distribution means no qualifying detected onset; it is not zero-minute lead."])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved event evaluation: {output.resolve()}")
    for result in results:
        print(f"{result['horizon_minutes']}m/{result['alarm_scope']}: "
              f"onsets {result['detected_onsets']}/{result['evaluable_onsets']}, "
              f"false episodes {result['false_alert_episodes']}/{result['complete_alert_episodes']}")


if __name__ == "__main__":
    main()
