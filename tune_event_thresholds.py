"""Select conservative onset thresholds using event episodes, not row F1.

The regular trainer selects thresholds for correlated minute rows.  That can
produce a low numeric threshold which looks good on F1 but raises one long
false alert episode.  This diagnostic uses the untouched validation forecasts
and :func:`weather_event_report.evaluate_events` to choose a threshold subject
to a false-episode budget.  It writes a new candidate manifest and never
changes a deployed bundle.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from train_weather_ai import load_features
from weather_event_report import evaluate_events


def _finite(value, fallback=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def tune_thresholds(
    bundle_dir: str | Path,
    data_dir: str | Path,
    max_false_episode_ratio: float = 0.20,
    cooldown_minutes: float = 30.0,
    alarm_scope: str = "dry_60m",
):
    bundle = Path(bundle_dir)
    evaluation = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    kind = evaluation["model_kind"]
    history, _, _ = load_features(data_dir)
    if not 0 <= max_false_episode_ratio <= 1:
        raise ValueError("max_false_episode_ratio must be between zero and one")

    output = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_bundle": str(bundle.resolve()), "model_kind": kind,
              "alarm_scope": alarm_scope, "cooldown_minutes": cooldown_minutes,
              "max_false_episode_ratio": max_false_episode_ratio, "horizons": []}
    tables = {}
    for result in evaluation["results"]:
        horizon = int(result["horizon"])
        predictions = pd.read_csv(bundle / f"{kind}_validation_predictions_{horizon}m.csv")
        rows = []
        # Fine resolution where onset models usually operate, coarser above
        # 0.20 to keep this diagnostic quick on long minute histories.
        thresholds = [round(value / 100, 2) for value in range(1, 21)]
        thresholds.extend([0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99])
        for threshold in thresholds:
            summary, _, _ = evaluate_events(
                history, predictions, horizon, threshold,
                dry_minutes=60, cooldown_minutes=cooldown_minutes,
                alarm_scope=alarm_scope,
            )
            rows.append(summary)
        table = pd.DataFrame(rows)
        # Zero complete episodes is valid (no alerts or only boundary-censored
        # alerts); it must not force a threshold of 1.0 when the observed
        # episode count is also zero.  The hard safety signal is the absolute
        # number of false complete episodes plus the ratio when a denominator
        # exists.
        table["safe"] = (
            table["false_alert_episodes"].eq(0)
            & table["false_alarm_ratio"].fillna(0).le(max_false_episode_ratio)
            # A threshold whose alerts occupy the entire validation boundary
            # is censored, not evidence of zero false alarms.  Accept either
            # no alerts or at least one complete episode to avoid this trap.
            & (table["alert_rows"].eq(0) | table["complete_alert_episodes"].gt(0))
        )
        safe = table.loc[table["safe"]].copy()
        nontrivial = safe.loc[safe["alert_rows"] > 0]
        if nontrivial.empty:
            selected = {
                "threshold": 1.0,
                "reason": "no_nontrivial_validation_threshold_meets_episode_budget",
            }
        else:
            chosen = nontrivial.sort_values(
                ["onset_recall", "false_alarm_ratio", "alert_rows", "threshold"],
                ascending=[False, True, True, True], kind="stable",
            ).iloc[0]
            selected = {
                "threshold": float(chosen["threshold"]),
                "reason": "max_onset_recall_under_false_episode_budget",
                "onset_recall": _finite(chosen["onset_recall"]),
                "detected_onsets": int(chosen["detected_onsets"]),
                "evaluable_onsets": int(chosen["evaluable_onsets"]),
                "false_alert_episodes": int(chosen["false_alert_episodes"]),
                "complete_alert_episodes": int(chosen["complete_alert_episodes"]),
                "false_alarm_ratio": _finite(chosen["false_alarm_ratio"]),
                "alert_rows": int(chosen["alert_rows"]),
            }
        output["horizons"].append({"horizon": horizon, **selected})
        tables[horizon] = table
    return output, tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; refuses to overwrite")
    parser.add_argument("--max-false-episode-ratio", type=float, default=0.20)
    parser.add_argument("--cooldown-minutes", type=float, default=30.0)
    parser.add_argument("--alarm-scope", choices=["dry_now", "dry_60m"], default="dry_60m")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, tables = tune_thresholds(
        args.bundle_dir, args.data_dir, args.max_false_episode_ratio,
        args.cooldown_minutes, args.alarm_scope,
    )
    (output_dir / "event_thresholds.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    for horizon, table in tables.items():
        table.to_csv(output_dir / f"threshold_metrics_{horizon}m.csv", index=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
