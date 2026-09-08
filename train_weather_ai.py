"""Train rain-window probabilities with chronological, purged evaluation.

60% train -> 10% probability calibration -> 10% model/threshold selection
-> 20% final test. The forecast horizon is purged at every boundary so a
training label cannot look into the next partition. Output is a new, complete
model directory; existing deployed artifacts are never overwritten by default.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix

from weather_features_lib import build_feature_frame
from weather_drift import build_baseline, save_baseline
from rain_probability import enforce_horizon_coherence
from weather_probability_model import PersistenceBlendClassifier, select_persistence_weight

RAIN_THRESHOLD = 0.10
PREDICTION_WINDOWS = [5, 10, 30, 60, 120]
RANDOM_STATE = 42


def load_features(data_dir="dataset"):
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    core_numeric = ["temp", "humidity", "pressure", "rain_flag"]
    optional_numeric = [
        "radar_available", "radar_point_intensity", "radar_nearby_max_intensity",
        "radar_trend_rising", "cloud_available", "cloud_cover_now",
        "cloud_cover_low_now", "cloud_trend_rising", "wind_available",
        "wind_speed", "wind_gust", "wind_direction",
        "nwp_available", "nwp_precipitation_probability", "nwp_precipitation",
        "nwp_weathercode", "nwp_wind_speed", "nwp_wind_direction",
        "nwp_cloud_cover", "nwp_lead_hours", "nwp_age_seconds",
    ]
    # Historical CSV exports may predate the optional radar/cloud columns.
    # Create them as missing values here; ``build_feature_frame`` applies the
    # explicit availability/missing semantics shared with live inference.
    for column in optional_numeric:
        if column not in df:
            df[column] = np.nan
    for column in [*core_numeric, *optional_numeric]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # Radar/cloud context is optional in historical CSVs.  Keep those rows;
    # weather_features_lib supplies explicit missingness/default features so
    # the model can learn the difference between “not available” and zero.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", *core_numeric])
    df = df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        return build_feature_frame(df.reset_index(drop=True))


def build_future_rain_target(group, window):
    """Any rain in the next N one-minute samples, with complete future coverage.

    Feature segments already reject interrupted cadence. The extra elapsed-time
    check rejects cumulative timestamp drift; no forward-fill invents dry labels.
    """
    future = group["rain_flag"].transform(
        lambda s: s.shift(-1).rolling(window, min_periods=window).max().shift(-(window - 1))
    )
    elapsed = group["timestamp"].shift(-window) - group.obj["timestamp"]
    valid = (elapsed - pd.Timedelta(minutes=window)).abs() <= pd.Timedelta(seconds=30)
    return future.where(valid)


def chronological_partitions(frame, horizon):
    """Four ordered partitions, with timestamp-based label purging at boundaries."""
    if len(frame) < 100:
        raise ValueError("Need at least 100 usable observations for chronological evaluation")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("Observations must be sorted and unique before splitting")
    cuts = [0, int(len(frame) * .6), int(len(frame) * .7), int(len(frame) * .8), len(frame)]
    parts = {}
    for i, name in enumerate(("train", "calibration", "validation", "test")):
        part = frame.iloc[cuts[i]:cuts[i + 1]]
        if i < 3:
            next_start = frame["timestamp"].iloc[cuts[i + 1]]
            # Include the 30-second observation jitter allowed by target creation.
            label_end = part["timestamp"] + pd.Timedelta(minutes=horizon, seconds=30)
            part = part.loc[label_end < next_start]
        if part.empty:
            raise ValueError(f"Empty {name} partition after purging {horizon}m labels")
        parts[name] = part
    return parts


def calculate_threshold_metrics(y_true, proba, threshold):
    pred = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold), "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "false_alarm_rate": float(fpr),
        "false_alarm_ratio": float(fp / (tp + fp)) if tp + fp else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def select_threshold(y_true, proba):
    sweep = pd.DataFrame(calculate_threshold_metrics(y_true, proba, t)
                         for t in np.round(np.arange(.01, 1.0, .01), 2))
    best = sweep.sort_values(["f1", "recall", "false_alarm_rate"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), sweep


def probability_metrics(y, p, threshold):
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": len(y), "positive_rows": int(np.sum(y)),
        "brier": float(brier_score_loss(y, p)),
        "average_precision": float(average_precision_score(y, p)) if np.sum(y) else None,
        **calculate_threshold_metrics(y, p, threshold),
    }


def reliability_table(y, proba):
    rows = pd.DataFrame({"observed": np.asarray(y), "probability": proba})
    rows["bin"] = np.minimum((rows["probability"] * 10).astype(int), 9)
    return rows.groupby("bin").agg(n=("observed", "size"),
                                    mean_predicted=("probability", "mean"),
                                    mean_observed=("observed", "mean")).reset_index()


def make_classifier(kind, trees=600, jobs=4):
    if kind in ("rf", "rf_onset"):
        return RandomForestClassifier(n_estimators=trees, max_depth=18, min_samples_leaf=2,
                                      max_features="sqrt", random_state=RANDOM_STATE,
                                      n_jobs=jobs, class_weight="balanced_subsample")
    if kind == "extra_trees":
        # Extremely randomized trees provide a diverse, CPU-friendly candidate
        # without changing the deployed RF bundle.  Keep the same feature
        # schema and calibration/evaluation path so the comparison is fair.
        return ExtraTreesClassifier(n_estimators=trees, max_depth=24, min_samples_leaf=2,
                                    max_features=0.7, random_state=RANDOM_STATE,
                                    n_jobs=jobs, class_weight="balanced")
    if kind == "xgb":
        # Optional dependency: RF training never imports XGBoost or SMOTE.
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.05,
                             subsample=.9, colsample_bytree=.9, eval_metric="logloss",
                             random_state=RANDOM_STATE, n_jobs=jobs)
    raise ValueError(f"Unknown model kind: {kind}")


def train_horizon(df, group, features, horizon, kind, output_dir, trees=600, jobs=4,
                  common_index=None, blend_persistence=False):
    future = build_future_rain_target(group, horizon)
    frame = df.copy()
    frame["target"] = (future > RAIN_THRESHOLD).astype(float).where(future.notna())
    frame = frame.dropna(subset=[*features, "target"]).copy()
    if common_index is not None:
        frame = frame.loc[common_index]
    # Use common time boundaries even for onset, so its evaluation is comparable.
    parts = chronological_partitions(frame, horizon)
    if kind == "rf_onset":
        parts = {name: part.loc[(part.rain_now <= RAIN_THRESHOLD) &
                               (part.rain_last_60m <= RAIN_THRESHOLD)] for name, part in parts.items()}
    for name, part in parts.items():
        if part.empty or (name != "test" and part.target.nunique() < 2):
            raise ValueError(f"{horizon}m/{kind}: {name} requires observed rain and dry examples")
    train, calibration, validation, test = (parts[k] for k in ("train", "calibration", "validation", "test"))
    print(f"{kind}/{horizon}m rows: " + ", ".join(f"{k}={len(v)}" for k, v in parts.items()), flush=True)
    model = make_classifier(kind, trees, jobs)
    if kind == "xgb":
        model.set_params(scale_pos_weight=float((train.target == 0).sum() / (train.target == 1).sum()))
    model.fit(train[features], train.target.astype(int))
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(calibration[features], calibration.target.astype(int))
    # Decide using validation only. The last 20% is not inspected until this is fixed.
    raw_val = model.predict_proba(validation[features])[:, 1]
    calibrated_val = calibrated.predict_proba(validation[features])[:, 1]
    raw_val_brier = float(brier_score_loss(validation.target, raw_val))
    cal_val_brier = float(brier_score_loss(validation.target, calibrated_val))
    use_calibration = cal_val_brier < raw_val_brier
    selected = calibrated if use_calibration else model
    val_p = calibrated_val if use_calibration else raw_val
    base_selected = selected
    validation_brier_before_blend = float(brier_score_loss(validation.target, val_p))
    model_weight = 1.0
    if blend_persistence:
        persistence_val = (validation.rain_now > RAIN_THRESHOLD).astype(float).to_numpy()
        model_weight = select_persistence_weight(validation.target, val_p, persistence_val)
        if model_weight < 1:
            selected = PersistenceBlendClassifier(selected, model_weight=model_weight)
            val_p = selected.predict_proba(validation[features])[:, 1]
    threshold, sweep = select_threshold(validation.target, val_p)
    raw_threshold, _ = select_threshold(validation.target, raw_val)
    p = selected.predict_proba(test[features])[:, 1]
    raw_p = model.predict_proba(test[features])[:, 1]
    base_p = base_selected.predict_proba(test[features])[:, 1]
    raw_metrics = probability_metrics(test.target, raw_p, raw_threshold)
    selected_metrics = probability_metrics(test.target, p, threshold)
    # Baselines use the SAME next-N-minute event, fitted on training data only.
    hourly_rate = train.groupby(train.timestamp.dt.hour).target.mean()
    climate = test.timestamp.dt.hour.map(hourly_rate).fillna(train.target.mean()).to_numpy()
    climate_brier = float(brier_score_loss(test.target, climate))
    selected_metrics["climatology_brier"] = climate_brier
    selected_metrics["brier_skill"] = 1 - selected_metrics["brier"] / climate_brier if climate_brier else None
    selected_metrics["persistence_brier"] = float(brier_score_loss(test.target, (test.rain_now > RAIN_THRESHOLD).astype(int)))
    dry_now = test.rain_now <= RAIN_THRESHOLD
    dry_60 = dry_now & (test.rain_last_60m <= RAIN_THRESHOLD)
    model_name = f"weather_{kind}_model_{horizon}m.joblib"
    joblib.dump(selected, output_dir / model_name, compress=3)
    sweep.to_csv(output_dir / f"{kind}_threshold_metrics_{horizon}m.csv", index=False)
    reliability_table(test.target, p).to_csv(output_dir / f"{kind}_reliability_{horizon}m.csv", index=False)
    pd.DataFrame({"timestamp": test.timestamp, "observed": test.target, "raw_probability": raw_p,
                  "probability": p, "base_probability": base_p, "dry_now": dry_now, "dry_60m": dry_60}).to_csv(
                      output_dir / f"{kind}_holdout_predictions_{horizon}m.csv", index=False)
    pd.DataFrame({"timestamp": validation.timestamp, "observed": validation.target,
                  "probability": val_p}).to_csv(
                      output_dir / f"{kind}_validation_predictions_{horizon}m.csv", index=False)
    result = {
        "horizon": horizon, "model_kind": kind, "calibration": "sigmoid" if use_calibration else "none",
        f"{kind}_best_threshold": threshold, f"{kind}_model_file": model_name,
        "validation_brier_raw": raw_val_brier, "validation_brier_calibrated": cal_val_brier,
        "persistence_blend_model_weight": model_weight,
        "validation_brier_before_blend": validation_brier_before_blend,
        "validation_brier_after_blend": float(brier_score_loss(validation.target, val_p)),
        "test_brier_before_blend": float(brier_score_loss(test.target, base_p)),
        "raw_test": raw_metrics, "test": selected_metrics,
        "dry_now_test": probability_metrics(test.target[dry_now], p[dry_now], threshold),
        "dry_60m_test": probability_metrics(test.target[dry_60], p[dry_60], threshold),
        "partitions": {name: {"n": len(part), "start": str(part.timestamp.iloc[0]),
                               "end": str(part.timestamp.iloc[-1])} for name, part in parts.items()},
    }
    print(f"  calibration={result['calibration']} threshold={threshold:.2f} "
          f"test Brier raw={raw_metrics['brier']:.5f} selected={selected_metrics['brier']:.5f} "
          f"F1={selected_metrics['f1']:.4f}", flush=True)
    return result


def apply_bundle_coherence(output_dir, results, kind):
    """Project nested horizons jointly and retune thresholds on validation only.

    All models use shared split timestamps. Both predictions and labels must be
    aligned, so no horizon contributes future validation labels to another's
    earlier test rows. Reports and runtime use the same projection function.
    """
    results.sort(key=lambda r: r["horizon"])
    files = {}
    for split, stem in (("validation", "validation"), ("test", "holdout")):
        frames = [pd.read_csv(output_dir / f"{kind}_{stem}_predictions_{r['horizon']}m.csv") for r in results]
        common_times = set(frames[0].timestamp)
        for frame in frames[1:]:
            common_times &= set(frame.timestamp)
        # Purging uses each horizon's length, so validation tails differ. Use
        # their common validation rows for threshold selection; test rows must
        # already be identical by construction.
        aligned = [frame.loc[frame.timestamp.isin(common_times)].reset_index(drop=True) for frame in frames]
        if not common_times or any(not frame.timestamp.equals(aligned[0].timestamp) for frame in aligned):
            raise ValueError("Joint horizon evaluation requires aligned chronological observations")
        if split == "test" and any(len(f) != len(a) for f, a in zip(frames, aligned)):
            raise ValueError("All horizons must have identical final test observations")
        raw = np.column_stack([frame.probability.to_numpy() for frame in aligned])
        labels = np.column_stack([frame.observed.to_numpy() for frame in aligned])
        if np.any(np.diff(labels, axis=1) < 0):
            raise ValueError("Rain-window labels must be nested across horizons")
        horizons = [f"{r['horizon']}m" for r in results]
        projected = np.array([list(enforce_horizon_coherence(dict(zip(horizons, row))).values()) for row in raw])
        for i, frame in enumerate(aligned):
            frame["unadjusted_probability"] = frame.probability
            frame["probability"] = projected[:, i]
        files[split] = aligned

    for i, result in enumerate(results):
        horizon = result["horizon"]
        validation, test = files["validation"][i], files["test"][i]
        threshold, sweep = select_threshold(validation.observed, validation.probability)
        result["unprocessed_test"] = result["test"].copy()
        result[f"{kind}_best_threshold"] = threshold
        result["probability_postprocessing"] = "isotonic_horizons"
        result["coherence_validation_rows"] = len(validation)
        result["validation_brier_after_coherence"] = float(brier_score_loss(validation.observed, validation.probability))
        result["test"].update(probability_metrics(test.observed, test.probability, threshold))
        climatology_brier = result["test"]["climatology_brier"]
        result["test"]["brier_skill"] = 1 - result["test"]["brier"] / climatology_brier if climatology_brier else None
        for name, column in (("dry_now_test", "dry_now"), ("dry_60m_test", "dry_60m")):
            mask = test[column].astype(bool)
            result[name] = probability_metrics(test.observed[mask], test.probability[mask], threshold)
        sweep.to_csv(output_dir / f"{kind}_threshold_metrics_{horizon}m.csv", index=False)
        reliability_table(test.observed, test.probability).to_csv(output_dir / f"{kind}_reliability_{horizon}m.csv", index=False)
        validation.to_csv(output_dir / f"{kind}_validation_predictions_{horizon}m.csv", index=False)
        test.to_csv(output_dir / f"{kind}_holdout_predictions_{horizon}m.csv", index=False)
    return results


def compare_existing_bundle(bundle_dir, reference_dir, start, data_dir="dataset"):
    """Diagnostic comparison on identical rows; never changes selected models.

    The caller must choose a start later than the reference model's training
    period. Existing legacy thresholds were already tuned on their old test
    set, so this comparison is diagnostic, not a new untouched validation set.
    """
    bundle = Path(bundle_dir)
    report = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    kind = report["model_kind"]
    frame, _, _ = load_features(data_dir)
    reference_features = joblib.load(Path(reference_dir) / "weather_features.joblib")
    thresholds_name = "weather_onset_thresholds.joblib" if kind == "rf_onset" else "weather_thresholds.joblib"
    reference_thresholds = joblib.load(Path(reference_dir) / thresholds_name)
    reference_postprocessing = reference_thresholds.get("probability_postprocessing", "none")
    if reference_postprocessing not in ("none", "isotonic_horizons"):
        raise ValueError("Unsupported reference probability postprocessing")
    results = []
    for result in report["results"]:
        horizon = result["horizon"]
        heldout = pd.read_csv(bundle / f"{kind}_holdout_predictions_{horizon}m.csv")
        heldout["timestamp"] = pd.to_datetime(heldout.timestamp, utc=True, format="mixed")
        heldout = heldout.loc[heldout.timestamp >= pd.to_datetime(start, utc=True)]
        rows = frame.merge(heldout, on="timestamp", validate="one_to_one")
        if rows.empty:
            raise ValueError("No common test rows on or after the requested comparison start")
        reference = joblib.load(Path(reference_dir) / f"weather_{kind}_model_{horizon}m.joblib")
        p = reference.predict_proba(rows[reference_features])[:, 1]
        if reference_postprocessing == "isotonic_horizons":
            reference_probabilities = {}
            for entry in reference_thresholds["horizons"]:
                lead = entry["horizon"]
                estimator = joblib.load(Path(reference_dir) / f"weather_{kind}_model_{lead}m.joblib")
                reference_probabilities[f"{lead}m"] = estimator.predict_proba(rows[reference_features])[:, 1]
            p = np.array([enforce_horizon_coherence(dict(zip(reference_probabilities, values)))[f"{horizon}m"]
                          for values in zip(*reference_probabilities.values())])
        old_brier = float(brier_score_loss(rows.observed, p))
        new_brier = float(brier_score_loss(rows.observed, rows.probability))
        results.append({"horizon": horizon, "n": len(rows), "start": str(rows.timestamp.min()),
                        "end": str(rows.timestamp.max()), "reference_brier": old_brier,
                        "candidate_brier": new_brier,
                        "brier_improvement_pct": 100 * (old_brier - new_brier) / old_brier if old_brier else None})
    comparison = {"reference_dir": str(Path(reference_dir).resolve()),
                  "caveat": "Legacy test data was previously used for threshold/model selection; diagnostic only.",
                  "results": results}
    (bundle / "legacy_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main(argv=None, default_kind="rf"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default=None, help="New directory; refuses to overwrite existing artifacts")
    parser.add_argument("--model", choices=["rf", "xgb", "rf_onset", "extra_trees"], default=default_kind)
    parser.add_argument("--horizons", nargs="+", type=int, choices=PREDICTION_WINDOWS, default=PREDICTION_WINDOWS)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--blend-persistence", action=argparse.BooleanOptionalAction, default=True,
                        help="Select a convex RF/current-rain blend on validation Brier")
    parser.add_argument("--coherent-probabilities", action=argparse.BooleanOptionalAction, default=True,
                        help="Enforce nested probabilities and retune thresholds on joint validation rows")
    args = parser.parse_args(argv)
    if args.trees < 1 or args.jobs == 0:
        parser.error("--trees must be positive and --jobs must be nonzero")
    output_dir = Path(args.output_dir or f"artifacts/{args.model}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=False)
    df, group, features = load_features(args.data_dir)
    horizons = sorted(set(args.horizons))
    valid_rows = df[features].notna().all(axis=1)
    for horizon in horizons:
        valid_rows &= build_future_rain_target(group, horizon).notna()
    common_index = df.index[valid_rows]
    print(f"Loaded {len(df)} rows, {len(features)} features, {df.timestamp.min()} to {df.timestamp.max()}", flush=True)
    results = [train_horizon(df, group, features, h, args.model, output_dir, args.trees, args.jobs,
                             common_index=common_index, blend_persistence=args.blend_persistence)
               for h in horizons]
    postprocessing = "isotonic_horizons" if args.coherent_probabilities else "none"
    if args.coherent_probabilities:
        apply_bundle_coherence(output_dir, results, args.model)
    joblib.dump(features, output_dir / "weather_features.joblib")
    drift_baseline = build_baseline(df, features)
    save_baseline(drift_baseline, output_dir / "drift_baseline.json")
    threshold_file = "weather_onset_thresholds.joblib" if args.model == "rf_onset" else "weather_thresholds.joblib"
    joblib.dump({"goal": "f1", "selected_on": "chronological_validation",
                 "probability_postprocessing": postprocessing, "horizons": results}, output_dir / threshold_file)
    feature_digest = hashlib.sha256(
        json.dumps(features, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact_files = {}
    for artifact in sorted(output_dir.iterdir()):
        if artifact.name == "evaluation.json" or not artifact.is_file():
            continue
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact_files[artifact.name] = {"bytes": artifact.stat().st_size, "sha256": digest}
    report = {"schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_kind": args.model, "feature_cadence": "60s +/-30s",
              "feature_count": len(features), "feature_names": features,
              "feature_sha256": feature_digest,
              "target": "any rain >0.10 in next N minutes", "trees": args.trees,
              "split": "60/10/10/20 chronological; horizon + 30s purge",
              "shared_horizon_splits": True, "blend_persistence": args.blend_persistence,
              "probability_postprocessing": postprocessing,
              "required_runtime_horizons": horizons,
              "drift_baseline_file": "drift_baseline.json",
              "artifact_files": artifact_files,
              "results": results}
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model bundle and held-out evaluation: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
