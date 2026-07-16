import argparse
import glob
import json
import logging
import os
import warnings
from datetime import datetime, timezone, timedelta

import joblib
import pandas as pd
from pandas.errors import PerformanceWarning

from weather_features_lib import build_feature_frame, heat_index_celsius, comfort_level
import cloud_nowcast
import radar_nowcast
import weather_db

logger = logging.getLogger(__name__)

RAIN_THRESHOLD = 0.10
PREDICTION_WINDOWS = [5, 10, 30]
# Temperature forecast only beats the persistence baseline at 30m (5m/10m
# models were dropped: min_samples_leaf=2 trees ballooned to ~600MB each in
# RAM for zero accuracy gain over persistence).
TEMP_FORECAST_WINDOWS = [30]
MODEL_KIND = "rf"  # Options: "xgb", "rf"
DATA_DIR = os.getenv("WEATHER_DATA_DIR", "dataset")

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


def _cached_joblib_load(path):
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = joblib.load(path)
    return _MODEL_CACHE[path]


def load_weather_data(path=None, max_files=None):
    if path:
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

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for col in ["temp", "humidity", "pressure", "rain_flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["temp", "humidity", "pressure", "rain_flag"])

    return (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def load_recent_from_db(lookback_minutes=130):
    """Supplement local CSV history with the same recent window from
    MariaDB — CSV can be thin right after an add-on reinstall (fresh /data
    volume) while the DB (a separate, longer-lived service) still has it.
    Returns an empty DataFrame on any DB failure; CSV-only remains the
    safe fallback, matching this project's existing DB-failure posture."""
    try:
        end = datetime.now(timezone.utc) + timedelta(minutes=1)
        start = end - timedelta(minutes=lookback_minutes)
        rows = weather_db.get_readings(start, end)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "temp", "humidity", "pressure", "rain_flag"])
        db_df = pd.DataFrame(rows).rename(columns={"reading_time": "timestamp"})
        # weather_db stores reading_time as naive UTC (see
        # _normalize_reading_time); pandas parses it back tz-naive, while the
        # CSV-sourced timestamps are tz-aware (ISO strings always carry
        # +00:00). Concat + sort_values on mixed tz-aware/naive columns raises
        # "Cannot compare tz-naive and tz-aware timestamps" — localize here so
        # both sides match before load_weather_data's data is combined with this.
        db_df["timestamp"] = pd.to_datetime(db_df["timestamp"], errors="coerce").dt.tz_localize("UTC")
        for col in ["temp", "humidity", "pressure", "rain_flag"]:
            db_df[col] = pd.to_numeric(db_df[col], errors="coerce")
        return db_df.dropna(subset=["timestamp", "temp", "humidity", "pressure", "rain_flag"])
    except Exception:
        logger.exception("Could not load recent readings from DB; continuing with CSV only")
        return pd.DataFrame(columns=["timestamp", "temp", "humidity", "pressure", "rain_flag"])


def load_threshold(thresholds, model_kind, horizon):
    horizon_info = next(
        item for item in thresholds["horizons"] if item["horizon"] == horizon
    )
    return horizon_info[f"{model_kind}_best_threshold"]


def predict(path=None, model_kind=MODEL_KIND):
    trained_features = _cached_joblib_load("weather_features.joblib")
    thresholds = _cached_joblib_load("weather_thresholds.joblib")

    # Feature engineering only looks back 120 minutes at most (see
    # weather_features_lib.py) — loading the whole multi-day history here was
    # pure waste that grew worse every day as dataset/*.csv accumulated.
    # 2 daily files always covers the lookback, even right after midnight.
    df = load_weather_data(path, max_files=2)

    db_df = load_recent_from_db()
    if not db_df.empty:
        df = pd.concat([df, db_df], ignore_index=True)
        df = (
            df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

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

    for horizon in PREDICTION_WINDOWS:
        model = _cached_joblib_load(f"weather_{model_kind}_model_{horizon}m.joblib")
        threshold = load_threshold(thresholds, model_kind, horizon)
        proba = model.predict_proba(X_latest)[0][1]
        will_rain = int(proba >= threshold)
        advance_alert = bool(will_rain and not is_raining_now)

        result["predictions"][f"{horizon}m"] = {
            "rain_alert": advance_alert,
            "model_rain_signal": bool(will_rain),
            "probability": float(proba),
            "threshold": float(threshold),
            "suppressed_by_rain_now": bool(will_rain and is_raining_now),
        }

    for horizon in TEMP_FORECAST_WINDOWS:
        temp_model_path = f"weather_temp_model_{horizon}m.joblib"
        if not os.path.exists(temp_model_path):
            continue
        temp_model = _cached_joblib_load(temp_model_path)
        forecast_temp = float(temp_model.predict(X_latest)[0])
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

    # Shadow-mode diagnostic only (see radar_nowcast.py / cloud_nowcast.py) —
    # purely additive, does not feed into any_rain_alert / next_rain_alert_horizon
    # / per-horizon rain_alert above. Neither call ever raises; each degrades to
    # {"available": False}.
    result["radar"] = radar_nowcast.get_radar_signal()
    result["cloud"] = cloud_nowcast.get_cloud_signal()

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
    parser.add_argument(
        "--model",
        default=MODEL_KIND,
        choices=["xgb", "rf"],
        help="Model kind to use.",
    )
    parser.add_argument("--json", action="store_true", help="Print one-line JSON for Node-RED.")
    args = parser.parse_args()

    prediction_result = predict(args.csv, args.model)
    if args.json:
        print(json.dumps(prediction_result, ensure_ascii=False))
    else:
        print_human_result(prediction_result)
