import argparse
import glob
import json
import logging
import os
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import pandas as pd
from pandas.errors import PerformanceWarning

from weather_features_lib import build_feature_frame, heat_index_celsius, comfort_level
from rain_probability import enforce_horizon_coherence
import cloud_nowcast
import radar_advection
import radar_nowcast
import weather_db

logger = logging.getLogger(__name__)

RAIN_THRESHOLD = 0.10
PREDICTION_WINDOWS = [5, 10, 30, 60, 120]
# Temperature forecast only beats the persistence baseline at 30m (5m/10m
# models were dropped: min_samples_leaf=2 trees ballooned to ~600MB each in
# RAM for zero accuracy gain over persistence).
TEMP_FORECAST_WINDOWS = [30]
MODEL_KIND = "rf"  # Options: "xgb", "rf", "extra_trees"
DATA_DIR = os.getenv("WEATHER_DATA_DIR", "dataset")
MAX_OBSERVATION_AGE_SECONDS = 180

# Below this magnitude (hPa over the last hour) the pressure is called "steady"
# rather than rising/falling — arbitrary but keeps the arrow from flickering
# on noise.
PRESSURE_TREND_STEADY_BAND = 0.5


def trend_arrow_label(pressure_trend_hpa_per_hour):
    if pressure_trend_hpa_per_hour > PRESSURE_TREND_STEADY_BAND:
        return "ขึ้น ↑"
    if pressure_trend_hpa_per_hour < -PRESSURE_TREND_STEADY_BAND:
        return "ลง ↓"
    return "คงที่ →"

warnings.simplefilter(action="ignore", category=PerformanceWarning)

# predict() is called on every /reading POST (~once/minute) and must not
# re-read these from disk each time — joblib.load() on a 600-tree RF is not
# cheap. Safe because a running process never needs a different model file
# for the same path; retraining requires a process restart to pick up.
_MODEL_CACHE = {}
_MODEL_METADATA_CACHE = {}
_HISTORY_COLUMNS = ["timestamp", "temp", "humidity", "pressure", "rain_flag", "light"]


def _cached_joblib_load(path):
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = joblib.load(path)
    return _MODEL_CACHE[path]


def _load_bundle_metadata(model_dir):
    """Read optional bundle provenance once without breaking legacy bundles."""
    key = str(Path(model_dir).resolve()) if model_dir else "<legacy-cwd>"
    if key in _MODEL_METADATA_CACHE:
        return _MODEL_METADATA_CACHE[key]

    metadata = {}
    bundle_root = Path(model_dir or ".")
    manifest_path = bundle_root / "model_manifest.json"
    evaluation_path = bundle_root / "evaluation.json"
    try:
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        elif evaluation_path.exists():
            metadata = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        logger.warning("Could not read model bundle metadata from %s", manifest_path)

    bundle_name = Path(model_dir).name if model_dir else "legacy-cwd"
    metadata = {
        "schema_version": metadata.get("schema_version"),
        "created_at_utc": metadata.get("created_at_utc"),
        "model_version": metadata.get("model_version") or (
            f"{bundle_name}:{metadata.get('created_at_utc')}"
            if metadata.get("created_at_utc") else f"legacy:{bundle_name}"
        ),
        "feature_count": metadata.get("feature_count"),
        "feature_sha256": metadata.get("feature_sha256"),
        "status": metadata.get("status", "legacy" if not manifest_path.exists() else "candidate"),
        "required_runtime_horizons": metadata.get("required_runtime_horizons"),
    }
    _MODEL_METADATA_CACHE[key] = metadata
    return metadata


def _feature_coverage(df):
    """Fraction of the 120-minute continuous context available at the tail."""
    if df.empty:
        return 0.0
    if "segment_id" in df:
        latest_segment = df.loc[df["segment_id"] == df["segment_id"].iloc[-1]]
        return float(min(1.0, len(latest_segment) / 120.0))
    return 1.0


def _confidence_fields(df, newest_timestamp, live):
    coverage = _feature_coverage(df)
    age_seconds = None
    age_factor = 1.0
    if live:
        age_seconds = float((pd.Timestamp.now(tz="UTC") - newest_timestamp).total_seconds())
        age_factor = max(0.0, min(1.0, 1.0 - max(age_seconds, 0.0) / MAX_OBSERVATION_AGE_SECONDS))
    confidence = float(max(0.0, min(1.0, coverage * age_factor)))
    if confidence >= 0.85:
        level = "high"
    elif confidence >= 0.55:
        level = "medium"
    else:
        level = "low"
    reason = "continuous sensor context"
    if coverage < 1.0:
        reason = "short or interrupted sensor context"
    elif live and age_factor < 1.0:
        reason = "latest sensor observation is aging"
    return {
        "data_age_seconds": age_seconds,
        "feature_coverage": coverage,
        "data_confidence": confidence,
        "confidence_level": level,
        "confidence_reason": reason,
    }


def _empty_history_frame():
    return pd.DataFrame(columns=_HISTORY_COLUMNS)


def _merge_weather_history(*frames):
    """Combine usable history frames into one UTC, time-sorted stream.

    CSV is supplied before DB by the caller, so the DB's more recently
    persisted copy wins when both sources contain the same timestamp.
    """
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return _empty_history_frame()

    df = pd.concat(non_empty, ignore_index=True)
    # Both loaders already normalize timestamps, but keeping the merge boundary
    # defensive prevents an injected/mixed source from reintroducing a
    # tz-naive/tz-aware sort failure.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["timestamp"])
    return (
        df.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def load_weather_data(path=None, max_files=None):
    if path is not None:
        csv_files = [path]
    else:
        csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
        if max_files:
            csv_files = csv_files[-max_files:]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found. Put files in {DATA_DIR}/*.csv or pass a CSV path."
        )

    df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    df = df[df["timestamp"].astype(str).str.lower() != "timestamp"]

    # CSV history can contain a mix of old naive values and current ISO-8601
    # UTC values.  Normalize both forms before sorting or merging with MariaDB.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["timestamp"])

    for col in ["temp", "humidity", "pressure", "rain_flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "light" in df:
        df["light"] = pd.to_numeric(df["light"], errors="coerce")
    df = df.dropna(subset=["temp", "humidity", "pressure", "rain_flag"])

    return _merge_weather_history(df)


def load_recent_from_db(lookback_minutes=130):
    """Supplement local CSV history with the same recent window from
    MariaDB — CSV can be thin or absent right after an add-on reinstall
    (fresh /data volume) while the DB (a separate, longer-lived service)
    still has it. Returns an empty DataFrame on any DB failure; CSV-only
    remains the safe fallback, matching this project's existing DB-failure
    posture."""
    try:
        end = datetime.now(timezone.utc) + timedelta(minutes=1)
        start = end - timedelta(minutes=lookback_minutes)
        rows = weather_db.get_readings(start, end)
        if not rows:
            return _empty_history_frame()
        db_df = pd.DataFrame(rows).rename(columns={"reading_time": "timestamp"})
        # MariaDB normally returns naive UTC DATETIME values, but migrations or
        # mocks may mix those with ISO values that carry an offset.  ``utc=True``
        # handles both without assuming every value is tz-naive.
        db_df["timestamp"] = pd.to_datetime(db_df["timestamp"], errors="coerce", utc=True, format="mixed")
        for col in [
            "temp", "humidity", "pressure", "rain_flag",
            "wind_available", "wind_speed", "wind_gust", "wind_direction",
            "light",
        ]:
            if col in db_df:
                db_df[col] = pd.to_numeric(db_df[col], errors="coerce")
        db_df = db_df.dropna(subset=["timestamp", "temp", "humidity", "pressure", "rain_flag"])
        return _attach_nwp_features(db_df)
    except Exception:
        logger.exception("Could not load recent readings from DB; continuing with CSV only")
        return _empty_history_frame()


def _attach_nwp_features(db_df):
    """Attach an as-of NWP row to each observed DB timestamp.

    A forecast is eligible only when both its issue time and valid time are
    no later than the observation.  This prevents a later provider revision
    from leaking into a historical feature row while still allowing the
    latest hourly forecast to cover all minute observations in that hour.
    NWP is optional: a DB/API failure leaves the sensor frame untouched.
    """
    if db_df is None or db_df.empty:
        return db_df
    try:
        start = db_df["timestamp"].min() - timedelta(hours=6)
        end = db_df["timestamp"].max() + timedelta(hours=1)
        rows = weather_db.get_nwp_forecast_rows(start, end)
        if not rows:
            return db_df
        nwp = pd.DataFrame(rows)
        nwp = nwp.rename(columns={"issued_hour": "nwp_issued_at", "valid_at": "nwp_valid_at"})
        nwp["nwp_issued_at"] = pd.to_datetime(nwp["nwp_issued_at"], errors="coerce", utc=True, format="mixed")
        nwp["nwp_valid_at"] = pd.to_datetime(nwp["nwp_valid_at"], errors="coerce", utc=True, format="mixed")
        nwp = nwp.dropna(subset=["nwp_issued_at", "nwp_valid_at"]).sort_values(
            ["nwp_valid_at", "nwp_issued_at"], kind="stable"
        )
        if nwp.empty:
            return db_df

        additions = {}
        for index, timestamp in db_df["timestamp"].items():
            eligible = nwp.loc[
                (nwp["nwp_issued_at"] <= timestamp)
                & (nwp["nwp_valid_at"] <= timestamp)
            ]
            if eligible.empty:
                continue
            valid_at = eligible["nwp_valid_at"].max()
            row = eligible.loc[eligible["nwp_valid_at"] == valid_at].iloc[-1]
            additions[index] = {
                "nwp_available": 1.0,
                "nwp_precipitation_probability": row.get("precipitation_probability"),
                "nwp_precipitation": row.get("precipitation"),
                "nwp_weathercode": row.get("weathercode"),
                "nwp_wind_speed": row.get("wind_speed"),
                "nwp_wind_direction": row.get("wind_direction"),
                "nwp_cloud_cover": row.get("cloud_cover"),
                "nwp_lead_hours": row.get("lead_hours"),
                "nwp_age_seconds": max(0.0, (timestamp - row["nwp_issued_at"]).total_seconds()),
            }
        if additions:
            extra = pd.DataFrame.from_dict(additions, orient="index")
            for column in extra.columns:
                db_df.loc[extra.index, column] = extra[column]
        return db_df
    except Exception:
        logger.exception("Could not attach as-of NWP features; continuing with sensor history")
        return db_df


def load_threshold(thresholds, model_kind, horizon):
    horizon_info = next(
        item for item in thresholds["horizons"] if item["horizon"] == horizon
    )
    return horizon_info[f"{model_kind}_best_threshold"]


def predict(path=None, model_kind=MODEL_KIND, model_dir=None):
    model_dir = model_dir or os.getenv("WEATHER_MODEL_DIR")

    def model_path(name):
        return os.path.join(model_dir, name) if model_dir else name

    trained_features = _cached_joblib_load(model_path("weather_features.joblib"))
    thresholds = _cached_joblib_load(model_path("weather_thresholds.joblib"))
    # Rain bundles may evolve with new trend features while the optional
    # temperature regressor remains on its older, smaller feature schema.
    # Keep the two manifests independent so promoting a rain candidate cannot
    # break the 30-minute temperature forecast.
    temp_features_path = model_path("weather_regression_features.joblib")
    if os.path.exists(temp_features_path):
        temp_features = _cached_joblib_load(temp_features_path)
    else:
        temp_features = trained_features
    bundle_metadata = _load_bundle_metadata(model_dir)

    # An explicit CSV is a historical replay.  It must be isolated from the
    # live DB; otherwise it silently incorporates present-day observations and
    # invalidates backtests.  Live inference instead uses whichever of the
    # latest CSV files and recent DB history is available.
    if path is not None:
        df = load_weather_data(path, max_files=2)
    else:
        try:
            csv_df = load_weather_data(max_files=2)
        except FileNotFoundError:
            csv_df = _empty_history_frame()

        db_df = load_recent_from_db()
        df = _merge_weather_history(csv_df, db_df)
        if df.empty:
            raise FileNotFoundError(
                f"No usable weather history in {DATA_DIR}/*.csv or the recent database window."
            )

    if path is None:
        newest = df["timestamp"].iloc[-1]
        age_seconds = (pd.Timestamp.now(tz="UTC") - newest).total_seconds()
        if age_seconds > MAX_OBSERVATION_AGE_SECONDS or age_seconds < -60:
            raise ValueError(f"Latest observation is not current ({newest}); cannot issue a live forecast")

    df, _group, built_features = build_feature_frame(df)

    missing_features = sorted(set(trained_features) - set(built_features))
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    usable_df = df.dropna(subset=trained_features).copy()
    if usable_df.empty:
        raise ValueError(
            "Not enough recent rows to build features. Need at least about 120 minutes "
            "of continuous data."
        )

    latest = usable_df.tail(1)
    newest_timestamp = df["timestamp"].iloc[-1]
    latest_usable_timestamp = latest["timestamp"].iloc[0]
    if latest_usable_timestamp != newest_timestamp:
        raise ValueError(
            "Cannot predict from stale history: the newest observation "
            f"({newest_timestamp}) lacks the continuous history required to build all features. "
            "Need at least about 120 minutes of continuous data ending at the newest observation."
        )
    X_latest = latest[trained_features]
    timestamp = latest["timestamp"].iloc[0]
    rain_now_value = float(latest["rain_flag"].iloc[0])
    is_raining_now = rain_now_value > RAIN_THRESHOLD

    temp_now = float(latest["temp"].iloc[0])
    humidity_now = float(latest["humidity"].iloc[0])
    heat_index_now = float(heat_index_celsius(temp_now, humidity_now))
    dew_point_now = float(latest["dew_point"].iloc[0])

    pressure_trend_raw = latest["pressure_trend_60m"].iloc[0]
    pressure_trend_1h = 0.0 if pd.isna(pressure_trend_raw) else float(pressure_trend_raw)

    result = {
        "timestamp": str(timestamp),
        "model": model_kind,
        "model_version": bundle_metadata["model_version"],
        "bundle_schema_version": bundle_metadata["schema_version"],
        "bundle_created_at_utc": bundle_metadata["created_at_utc"],
        "feature_count": len(trained_features),
        "feature_sha256": bundle_metadata.get("feature_sha256"),
        "model_status": bundle_metadata.get("status"),
        "required_runtime_horizons": bundle_metadata["required_runtime_horizons"]
        or PREDICTION_WINDOWS,
        "temp": temp_now,
        "humidity": humidity_now,
        "pressure": float(latest["pressure"].iloc[0]),
        "heat_index": heat_index_now,
        "comfort_level": comfort_level(heat_index_now),
        "dew_point": dew_point_now,
        "pressure_trend": pressure_trend_1h,
        "trend_arrow": trend_arrow_label(pressure_trend_1h),
        "rain_now": rain_now_value,
        "is_raining_now": bool(is_raining_now),
        "predictions": {},
        "temp_forecast": {},
    }
    result.update(_confidence_fields(df, newest_timestamp, live=path is None))

    configured_horizons = sorted({int(item["horizon"]) for item in thresholds.get("horizons", [])})
    horizons = [horizon for horizon in PREDICTION_WINDOWS if horizon in configured_horizons]
    if not horizons:
        raise ValueError("Model bundle has no supported prediction horizons")
    raw_probabilities = {}
    for horizon in horizons:
        model = _cached_joblib_load(model_path(f"weather_{model_kind}_model_{horizon}m.joblib"))
        raw_probabilities[f"{horizon}m"] = float(model.predict_proba(X_latest)[0][1])
    postprocessing = thresholds.get("probability_postprocessing", "none")
    if postprocessing == "isotonic_horizons":
        probabilities = enforce_horizon_coherence(raw_probabilities)
    elif postprocessing == "none":
        probabilities = raw_probabilities
    else:
        raise ValueError(f"Unsupported probability postprocessing: {postprocessing}")
    result["probability_postprocessing"] = postprocessing

    for horizon in horizons:
        threshold = load_threshold(thresholds, model_kind, horizon)
        proba = probabilities[f"{horizon}m"]
        will_rain = int(proba >= threshold)
        advance_alert = bool(will_rain and not is_raining_now)

        result["predictions"][f"{horizon}m"] = {
            "rain_alert": advance_alert,
            "model_rain_signal": bool(will_rain),
            "probability": float(proba),
            "threshold": float(threshold),
            "suppressed_by_rain_now": bool(will_rain and is_raining_now),
        }
        if postprocessing != "none":
            result["predictions"][f"{horizon}m"]["unadjusted_probability"] = raw_probabilities[f"{horizon}m"]

    for horizon in TEMP_FORECAST_WINDOWS:
        temp_model_path = model_path(f"weather_temp_model_{horizon}m.joblib")
        if not os.path.exists(temp_model_path):
            continue
        temp_model = _cached_joblib_load(temp_model_path)
        temp_input = latest[temp_features]
        forecast_temp = float(temp_model.predict(temp_input)[0])
        result["temp_forecast"][f"{horizon}m"] = {
            "temp": forecast_temp,
            "delta": forecast_temp - temp_now,
        }

    any_rain_alert = any(pred["rain_alert"] for pred in result["predictions"].values())
    next_horizon = next(
        (h for h, pred in result["predictions"].items() if pred["rain_alert"]),
        None,
    )
    result["any_rain_alert"] = bool(any_rain_alert)
    result["next_rain_alert_horizon"] = (
        f"{next_horizon}" if next_horizon is not None else None
    )
    result["alert_message"] = (
        f"Rain alert in {result['next_rain_alert_horizon']}"
        if any_rain_alert
        else (
            "Raining now; no advance rain alert"
            if is_raining_now
            else "No rain alert"
        )
    )

    # Shadow-mode diagnostic only (see radar_nowcast.py / cloud_nowcast.py /
    # radar_advection.py) — purely additive, does not feed into
    # any_rain_alert / next_rain_alert_horizon / per-horizon rain_alert
    # above. None of these calls ever raise; each degrades to
    # {"available": False}.
    if path is not None:
        # Present-day radar/cloud must not be attached to a historical replay.
        for source in ("radar", "cloud", "radar_advection"):
            result[source] = {"available": False, "reason": "historical_replay"}
    else:
        result["radar"] = radar_nowcast.get_radar_signal()
        result["cloud"] = cloud_nowcast.get_cloud_signal()
        result["radar_advection"] = radar_advection.get_advection_signal()

    return result


def print_human_result(result):
    print("=" * 50)
    print("Weather AI Prediction")
    print("=" * 50)
    print(f"timestamp: {result['timestamp']}")
    print(f"model: {result['model'].upper()}")
    print(f"temp: {result['temp']:.2f}")
    print(f"humidity: {result['humidity']:.2f}")
    print(f"pressure: {result['pressure']:.2f}")
    print(f"heat_index: {result['heat_index']:.2f} ({result['comfort_level']})")
    print(f"dew_point: {result['dew_point']:.2f}")
    print(f"pressure_trend: {result['pressure_trend']:+.2f} hPa/h ({result['trend_arrow']})")
    if result["temp_forecast"]:
        forecasts = ", ".join(
            f"{h}={info['temp']:.2f} ({info['delta']:+.2f})"
            for h, info in result["temp_forecast"].items()
        )
        print(f"temp_forecast: {forecasts}")
    print(f"rain_now: {result['rain_now']:.3f}")
    print(f"is_raining_now: {result['is_raining_now']}")
    print(f"any_rain_alert: {result['any_rain_alert']}")
    print(f"next_rain_alert_horizon: {result['next_rain_alert_horizon']}")
    print(f"alert_message: {result['alert_message']}")
    radar = result.get("radar", {})
    if radar.get("available"):
        print(
            f"radar: point={radar['point_intensity']} nearby_max={radar['nearby_max_intensity']} "
            f"trend_rising={radar['trend_rising']} (shadow mode, diagnostic only)"
        )
    else:
        print("radar: unavailable (shadow mode, diagnostic only)")
    cloud = result.get("cloud", {})
    if cloud.get("available"):
        print(
            f"cloud: now={cloud['cloud_cover_now']}% low={cloud['cloud_cover_low_now']}% "
            f"next_hour={cloud['cloud_cover_next_hour']}% trend_rising={cloud['trend_rising']} "
            "(shadow mode, diagnostic only)"
        )
    else:
        print("cloud: unavailable (shadow mode, diagnostic only)")

    for horizon, prediction in result["predictions"].items():
        status = "RAIN ALERT" if prediction["rain_alert"] else "no rain alert"
        print("\n" + "-" * 50)
        print(f"{horizon}: {status}")
        print(f"model_rain_signal: {prediction['model_rain_signal']}")
        print(f"suppressed_by_rain_now: {prediction['suppressed_by_rain_now']}")
        print(f"probability: {prediction['probability']:.3f}")
        print(f"threshold: {prediction['threshold']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict rain alerts from saved weather models.")
    parser.add_argument("--csv", default=None, help="Optional CSV path. Defaults to dataset/*.csv.")
    parser.add_argument("--model-dir", default=None, help="Complete model bundle directory (or WEATHER_MODEL_DIR)")
    parser.add_argument(
        "--model",
        default=MODEL_KIND,
        choices=["xgb", "rf", "extra_trees"],
        help="Model kind to use.",
    )
    parser.add_argument("--json", action="store_true", help="Print one-line JSON for Node-RED.")
    args = parser.parse_args()

    prediction_result = predict(args.csv, args.model, args.model_dir)
    if args.json:
        print(json.dumps(prediction_result, ensure_ascii=False))
    else:
        print_human_result(prediction_result)
