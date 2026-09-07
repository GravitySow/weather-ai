"""Shared feature engineering for the rain-prediction model.

Both train_weather_ai.py and predict_weather_ai.py import build_feature_frame
from here so the training features and the inference features can never drift.
Any feature change must be made ONLY in this file.
"""

import numpy as np
import pandas as pd


def add_lag_features(df, group, column, windows, features):
    for window in windows:
        lag_col = f"{column}_{window}m_ago"
        delta_col = f"{column}_delta_{window}m"
        trend_col = f"{column}_trend_{window}m"

        df[lag_col] = group[column].shift(window)
        df[delta_col] = df[column] - df[lag_col]
        df[trend_col] = df[delta_col]

        features.extend([lag_col, delta_col, trend_col])


def add_rolling_features(df, group, column, windows, stats, features):
    for window in windows:
        rolling = group[column].rolling(window=window, min_periods=window)

        if "mean" in stats:
            col = f"{column}_avg_{window}m"
            df[col] = rolling.mean().reset_index(level=0, drop=True)
            features.append(col)

        if "std" in stats:
            col = f"{column}_std_{window}m"
            df[col] = rolling.std().reset_index(level=0, drop=True)
            features.append(col)

        if "min" in stats:
            col = f"{column}_min_{window}m"
            df[col] = rolling.min().reset_index(level=0, drop=True)
            features.append(col)

        if "max" in stats:
            col = f"{column}_max_{window}m"
            df[col] = rolling.max().reset_index(level=0, drop=True)
            features.append(col)


def add_time_rate_features(df, group, columns, windows, features):
    """Add timestamp-aware rates of change for short weather trends.

    Existing ``*_trend_*`` columns intentionally preserve the raw difference
    for backwards-compatible model bundles. These new ``*_slope_*`` columns
    divide by the measured elapsed minutes, so timestamp jitter does not turn
    a five-minute trend into a row-count artefact.
    """
    elapsed_by_window = {}
    for window in windows:
        lag_timestamp = group["timestamp"].shift(window)
        elapsed = (df["timestamp"] - lag_timestamp).dt.total_seconds() / 60.0
        elapsed_by_window[window] = elapsed.where(elapsed > 0)
    for column in columns:
        for window in windows:
            lag = group[column].shift(window)
            col = f"{column}_slope_{window}m"
            df[col] = (df[column] - lag) / elapsed_by_window[window]
            features.append(col)


def add_rain_fraction_features(df, group, windows, features):
    """Add short rain-duty-cycle and rain-change features."""
    for window in windows:
        current = group["rain_flag"].rolling(window=window, min_periods=window).mean()
        current = current.reset_index(level=0, drop=True)
        previous = current.groupby(df["segment_id"], group_keys=False).shift(window)
        fraction_col = f"rain_frac_{window}m"
        slope_col = f"rain_slope_{window}m"
        df[fraction_col] = current
        df[slope_col] = current - previous
        features.extend([fraction_col, slope_col])


def unique(items):
    return list(dict.fromkeys(items))


def add_external_nowcast_features(df, features):
    """Add radar/cloud shadow features with explicit missing-value semantics.

    Older CSV histories do not contain these columns while MariaDB readings
    do.  Always creating the columns keeps one feature schema for both paths;
    the ``*_available`` and ``*_missing`` indicators prevent zero-filled
    legacy rows from being mistaken for a measured clear sky.
    """
    specs = {
        "radar_available": 0.0,
        "radar_point_intensity": 0.0,
        "radar_nearby_max_intensity": 0.0,
        "radar_trend_rising": 0.0,
        "cloud_available": 0.0,
        "cloud_cover_now": 0.0,
        "cloud_cover_low_now": 0.0,
        "cloud_trend_rising": 0.0,
    }
    for column, default in specs.items():
        if column not in df:
            df[column] = default
            continue
        values = df[column]
        if values.dtype == object or str(values.dtype) == "boolean":
            values = values.astype(str).str.lower().map(
                {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0}
            )
        df[column] = pd.to_numeric(values, errors="coerce").fillna(default)

    for source in ("radar", "cloud"):
        available = df[f"{source}_available"].fillna(0.0).clip(0.0, 1.0)
        df[f"{source}_available"] = available
        df[f"{source}_missing"] = (available <= 0.0).astype(float)
        features.extend([
            f"{source}_available",
            f"{source}_missing",
        ])

    features.extend([
        "radar_point_intensity", "radar_nearby_max_intensity", "radar_trend_rising",
        "cloud_cover_now", "cloud_cover_low_now", "cloud_trend_rising",
    ])


def heat_index_celsius(temp_c, rh):
    """NOAA Rothfusz heat index, coefficients adapted for Celsius / %RH.

    Valid roughly for temp >= 27 C and RH >= 40%. Outside that envelope the
    apparent temperature is ~ the air temperature, so we return temp_c there.
    Accepts scalars or numpy arrays.
    """
    t = np.asarray(temp_c, dtype=float)
    r = np.asarray(rh, dtype=float)
    hi = (
        -8.78469475556
        + 1.61139411 * t
        + 2.33854883889 * r
        - 0.14611605 * t * r
        - 0.012308094 * t**2
        - 0.0164248277778 * r**2
        + 0.002211732 * t**2 * r
        + 0.00072546 * t * r**2
        - 0.000003582 * t**2 * r**2
    )
    return np.where((t >= 27.0) & (r >= 40.0), hi, t)


def comfort_level(heat_index_c):
    """Map a heat index (Celsius) to a NOAA-style comfort category."""
    hi = float(heat_index_c)
    if hi < 27:
        return "comfortable"
    if hi < 32:
        return "caution"
    if hi < 41:
        return "extreme_caution"
    if hi < 54:
        return "danger"
    return "extreme_danger"


def build_feature_frame(df):
    """Add all model features to df in place and return (df, group, features).

    Expects df with cleaned, numeric, de-duplicated, time-sorted columns:
    timestamp, temp, humidity, pressure, rain_flag.
    """
    # Row-based lags assume one observation per minute. Allow timestamp jitter,
    # but restart after missing or duplicated samples instead of silently making
    # a "30m" feature span 31+ minutes (or less than 30 minutes).
    time_gap = df["timestamp"].diff()
    irregular = time_gap.lt(pd.Timedelta(seconds=30)) | time_gap.gt(pd.Timedelta(seconds=90))
    df["segment_id"] = irregular.cumsum()
    group = df.groupby("segment_id", group_keys=False)

    features = ["temp", "humidity", "pressure"]

    # --- Baseline lag / trend features ---
    lag_windows = [5, 10, 15, 30, 60]
    rolling_windows = [5, 15, 30, 60]

    for column in ["pressure", "humidity", "temp"]:
        add_lag_features(df, group, column, lag_windows, features)

    for column in ["pressure", "humidity", "temp"]:
        add_rolling_features(
            df, group, column, rolling_windows,
            stats=["mean", "std", "min", "max"], features=features,
        )

    df["dew_point"] = df["temp"] - ((100 - df["humidity"]) / 5)
    df["temp_dewpoint_gap"] = df["temp"] - df["dew_point"]
    features.extend(["dew_point", "temp_dewpoint_gap"])

    df["rain_now"] = df["rain_flag"]
    features.append("rain_now")

    for window in [15, 30, 60]:
        col = f"rain_last_{window}m"
        df[col] = (
            group["rain_flag"].rolling(window=window, min_periods=window)
            .max().reset_index(level=0, drop=True)
        )
        features.append(col)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    hour_radian = 2 * np.pi * df["hour"] / 24
    day_radian = 2 * np.pi * df["day_of_week"] / 7
    df["hour_sin"] = np.sin(hour_radian)
    df["hour_cos"] = np.cos(hour_radian)
    df["day_sin"] = np.sin(day_radian)
    df["day_cos"] = np.cos(day_radian)
    features.extend(["hour_sin", "hour_cos", "day_sin", "day_cos"])

    # --- External nowcast context (optional in legacy CSV, persisted in DB) ---
    add_external_nowcast_features(df, features)

    # --- Augmented physical features (improve longer lead time, esp. 30m) ---
    # Short, timestamp-aware rates expose the direction and speed of local
    # changes without replacing the legacy raw-delta features above.
    add_time_rate_features(
        df, group, ["pressure", "humidity", "temp"], [5, 10, 30], features
    )

    # Short rain history helps distinguish a dry-to-wet transition from a
    # single noisy rain sensor sample. Keep the existing 60m fraction below
    # for compatibility and add these separately for candidate models.
    add_rain_fraction_features(df, group, [5, 10, 30], features)

    # Longer pressure / humidity tendency: slow synoptic-scale trend signal.
    add_lag_features(df, group, "pressure", [90, 120], features)
    add_lag_features(df, group, "humidity", [90, 120], features)

    # Pressure acceleration: how the 30m pressure delta itself is changing.
    df["pressure_accel_30m"] = (
        df["pressure_delta_30m"] - group["pressure_delta_30m"].shift(30)
    )
    features.append("pressure_accel_30m")

    # Signed rate of change per minute over several spans.
    for column in ["pressure", "humidity", "temp"]:
        for window in [15, 30, 60]:
            col = f"{column}_roc_{window}m"
            df[col] = (df[column] - group[column].shift(window)) / window
            features.append(col)

    # Longer rolling stats (90 / 120 min) for slow build-up.
    for column in ["pressure", "humidity"]:
        add_rolling_features(
            df, group, column, [90, 120], stats=["mean", "std"], features=features,
        )

    # Dew-point dynamics: gap trend and recent dew-point change.
    df["dewpoint_gap_delta_30m"] = (
        df["temp_dewpoint_gap"] - group["temp_dewpoint_gap"].shift(30)
    )
    df["dew_point_delta_30m"] = df["dew_point"] - group["dew_point"].shift(30)
    features.extend(["dewpoint_gap_delta_30m", "dew_point_delta_30m"])

    # Rain recency / intensity over a longer memory.
    df["rain_last_120m"] = (
        group["rain_flag"].rolling(window=120, min_periods=120).max()
        .reset_index(level=0, drop=True)
    )
    df["rain_frac_60m"] = (
        group["rain_flag"].rolling(window=60, min_periods=60).mean()
        .reset_index(level=0, drop=True)
    )
    features.extend(["rain_last_120m", "rain_frac_60m"])

    return df, group, unique(features)
