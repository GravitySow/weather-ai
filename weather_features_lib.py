"""Shared feature engineering for the rain-prediction model.

Both train_weather_ai.py and predict_weather_ai.py import build_feature_frame
from here so the training features and the inference features can never drift.
Any feature change must be made ONLY in this file.
"""

import numpy as np
import pandas as pd


# Station coordinates used for the clear-sky illuminance proxy.  The feature is
# deliberately deterministic and local; it is only a fallback reference curve
# and never a replacement for a calibrated lux sensor.
DEFAULT_LATITUDE = 13.8692387
DEFAULT_LONGITUDE = 100.5180519
SHORT_GAP_MIN_SECONDS = 90.0
SHORT_GAP_MAX_SECONDS = 630.0
# A second API delivery can arrive as a short burst after a delayed poll.  A
# 25-second burst is still close enough to the normal one-minute cadence to
# keep the same feature segment; only very short intervals (likely a true
# duplicate/out-of-order sample) restart the row-based history.
MIN_VALID_INTERVAL_SECONDS = 20.0
NOMINAL_SAMPLE_SECONDS = 60.0


def _repair_short_gaps(df):
    """Insert synthetic minute rows for a short sensor-history gap.

    The feature matrix is row-based (``shift(5)`` means five minutes), so a
    timestamp jump would otherwise make every later lag too old and force a
    full 120-minute warm-up. Gaps up to five minutes (with a small timestamp
    jitter allowance) are repaired with one interpolated row per missing
    minute. Gaps up to ten minutes (plus a small timestamp jitter allowance)
    are filled; longer gaps still become a new segment and are never filled.
    """
    frame = df.reset_index(drop=True).copy()
    frame["observation_gap_filled"] = 0.0
    if len(frame) < 2:
        return frame

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True, format="mixed")
    numeric_columns = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"observation_gap_filled", "segment_id"}
    ]
    repaired_rows = []
    repaired_any = False

    for index in range(len(frame) - 1):
        previous = frame.iloc[index].copy()
        following = frame.iloc[index + 1].copy()
        repaired_rows.append(previous)
        left_ts = timestamps.iloc[index]
        right_ts = timestamps.iloc[index + 1]
        gap_seconds = (right_ts - left_ts).total_seconds()
        if not (
            gap_seconds > SHORT_GAP_MIN_SECONDS
            and gap_seconds <= SHORT_GAP_MAX_SECONDS
        ):
            continue

        # Round to the expected one-minute cadence, then insert every missing
        # slot rather than a single midpoint. This preserves row-based lags
        # for a gap of two to ten minutes.
        missing_count = max(1, int(round(gap_seconds / NOMINAL_SAMPLE_SECONDS)) - 1)
        repaired_any = True
        for slot in range(1, missing_count + 1):
            fraction = slot / (missing_count + 1)
            inserted = previous.copy()
            inserted["timestamp"] = left_ts + (right_ts - left_ts) * fraction

            for column in numeric_columns:
                left_value = pd.to_numeric(previous[column], errors="coerce")
                right_value = pd.to_numeric(following[column], errors="coerce")
                if pd.notna(left_value) and pd.notna(right_value):
                    inserted[column] = left_value + (right_value - left_value) * fraction
                elif pd.notna(left_value):
                    inserted[column] = left_value
                else:
                    inserted[column] = right_value

            # Preserve non-numeric context from the nearest valid observation.
            for column in frame.columns:
                if column not in numeric_columns and column not in {
                    "timestamp", "observation_gap_filled", "segment_id",
                }:
                    inserted[column] = (
                        previous[column] if pd.notna(previous[column]) else following[column]
                    )

            # Do not invent a dry-to-wet transition. If either side says it is
            # raining, preserve that conservative state in every bridge row.
            for column in ("rain_flag", "rain_sensor"):
                if column in frame.columns:
                    left_value = pd.to_numeric(previous[column], errors="coerce")
                    right_value = pd.to_numeric(following[column], errors="coerce")
                    inserted[column] = np.nanmax([left_value, right_value])
            if "rain" in frame.columns:
                inserted["rain"] = bool(previous["rain"]) or bool(following["rain"])
            inserted["observation_gap_filled"] = 1.0
            repaired_rows.append(inserted)

    repaired_rows.append(frame.iloc[-1].copy())
    if not repaired_any:
        return frame
    return pd.DataFrame(repaired_rows, columns=frame.columns).reset_index(drop=True)


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


def _solar_elevation_degrees(timestamps, latitude=DEFAULT_LATITUDE,
                             longitude=DEFAULT_LONGITUDE):
    """Approximate solar elevation for UTC timestamps.

    NOAA's compact fractional-year approximation is accurate enough for a
    daylight availability gate.  It avoids adding a runtime dependency such as
    pvlib to the Home Assistant add-on and is not used as a weather forecast.
    """
    ts = pd.to_datetime(timestamps, errors="coerce", utc=True)
    # Keep NaT safe: the resulting rows are marked unavailable below.
    day = ts.dt.dayofyear.to_numpy(dtype=float)
    hour = (
        ts.dt.hour.to_numpy(dtype=float)
        + ts.dt.minute.to_numpy(dtype=float) / 60.0
        + ts.dt.second.to_numpy(dtype=float) / 3600.0
    )
    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    equation_minutes = (
        229.18
        * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2.0 * gamma)
            - 0.040849 * np.sin(2.0 * gamma)
        )
    )
    true_solar_minutes = hour * 60.0 + equation_minutes + 4.0 * longitude
    hour_angle = np.deg2rad((true_solar_minutes / 4.0) - 180.0)
    latitude_rad = np.deg2rad(latitude)
    sin_elevation = (
        np.sin(latitude_rad) * np.sin(decl)
        + np.cos(latitude_rad) * np.cos(decl) * np.cos(hour_angle)
    )
    elevation = np.rad2deg(np.arcsin(np.clip(sin_elevation, -1.0, 1.0)))
    return pd.Series(elevation, index=ts.index)


def add_illuminance_features(df, group, features):
    """Add optional illuminance/cloud-shadow features.

    A valid zero-lux reading is different from an absent sensor value.  During
    night or very low solar elevation the clear-sky comparison is explicitly
    unavailable, so a dark room/night cannot masquerade as a rain signal.
    """
    raw = pd.to_numeric(df.get("light", pd.Series(np.nan, index=df.index)), errors="coerce")
    available = raw.notna().astype(float)
    light = raw.clip(lower=0.0).fillna(0.0)
    elevation = _solar_elevation_degrees(df["timestamp"])
    daylight = available * (elevation > 5.0).astype(float)
    # A smooth clear-sky proxy is sufficient for relative drops.  The absolute
    # lux value is intentionally not treated as physically calibrated.
    sun = np.clip(np.sin(np.deg2rad(elevation)), 0.0, None)
    expected = 105000.0 * np.power(sun, 1.15)
    expected = pd.Series(expected, index=df.index)
    ratio = (light / expected.replace(0.0, np.nan)).clip(0.0, 1.5)
    # Keep the matrix numeric for tree models; availability/daylight bits carry
    # the semantics, so an unavailable ratio is a neutral zero rather than a
    # dropped observation.
    ratio = ratio.where(daylight > 0.0).fillna(0.0)

    df["light"] = light
    df["light_available"] = available
    df["light_missing"] = (available <= 0.0).astype(float)
    df["solar_elevation_deg"] = elevation
    df["daylight_available"] = daylight
    df["clear_sky_light_ratio"] = ratio
    features.extend([
        "light_available", "light_missing", "light", "solar_elevation_deg",
        "daylight_available", "clear_sky_light_ratio",
    ])

    for window in [5, 10, 30]:
        # GroupBy objects may have been created before these derived columns
        # existed; use the frame directly so the function also works for a
        # caller that reuses a precomputed group.
        lag_ratio = df["clear_sky_light_ratio"].groupby(df["segment_id"], group_keys=False).shift(window)
        lag_light = df["light"].groupby(df["segment_id"], group_keys=False).shift(window)
        ratio_delta = ratio - lag_ratio
        light_slope = (light - lag_light) / float(window)
        ratio_delta = ratio_delta.where(daylight > 0.0).fillna(0.0)
        light_slope = light_slope.where(daylight > 0.0).fillna(0.0)
        df[f"light_drop_vs_expected_{window}m"] = ratio_delta
        df[f"light_slope_{window}m"] = light_slope
        features.extend([f"light_drop_vs_expected_{window}m", f"light_slope_{window}m"])


def unique(items):
    return list(dict.fromkeys(items))


def add_external_nowcast_features(df, features):
    """Add radar/cloud shadow features with explicit missing-value semantics.

    Older CSV histories do not contain these columns while MariaDB readings
    do.  Always creating the columns keeps one feature schema for both paths;
    the ``*_available`` and ``*_missing`` indicators prevent zero-filled
    legacy rows from being mistaken for a measured clear sky.
    """
    raw_wind_available = (
        pd.to_numeric(df["wind_available"], errors="coerce")
        if "wind_available" in df else None
    )
    raw_nwp_available = (
        pd.to_numeric(df["nwp_available"], errors="coerce")
        if "nwp_available" in df else None
    )
    wind_available_present = raw_wind_available is not None and raw_wind_available.notna().any()
    nwp_available_present = raw_nwp_available is not None and raw_nwp_available.notna().any()
    nwp_probability_valid = (
        pd.to_numeric(df["nwp_precipitation_probability"], errors="coerce").notna()
        if "nwp_precipitation_probability" in df else pd.Series(False, index=df.index)
    )
    wind_speed_valid = (
        pd.to_numeric(df["wind_speed"], errors="coerce").notna()
        if "wind_speed" in df else pd.Series(False, index=df.index)
    )
    specs = {
        "radar_available": 0.0,
        "radar_point_intensity": 0.0,
        "radar_nearby_max_intensity": 0.0,
        "radar_trend_rising": 0.0,
        "cloud_available": 0.0,
        "cloud_cover_now": 0.0,
        "cloud_cover_low_now": 0.0,
        "cloud_trend_rising": 0.0,
        "wind_available": 0.0,
        "wind_speed": 0.0,
        "wind_gust": 0.0,
        "wind_direction": 0.0,
        "nwp_available": 0.0,
        "nwp_precipitation_probability": 0.0,
        "nwp_precipitation": 0.0,
        "nwp_weathercode": 0.0,
        "nwp_wind_speed": 0.0,
        "nwp_wind_direction": 0.0,
        "nwp_cloud_cover": 0.0,
        "nwp_lead_hours": 0.0,
        "nwp_age_seconds": 999999.0,
    }
    for column, default in specs.items():
        if column not in df:
            df[column] = default
            continue
        values = df[column]
        if values.dtype == object or str(values.dtype) == "boolean":
            normalized = values.astype(str).str.strip().str.lower()
            direction_map = {
                "n": 0.0, "nne": 22.5, "ne": 45.0, "ene": 67.5,
                "e": 90.0, "ese": 112.5, "se": 135.0, "sse": 157.5,
                "s": 180.0, "ssw": 202.5, "sw": 225.0, "wsw": 247.5,
                "w": 270.0, "wnw": 292.5, "nw": 315.0, "nnw": 337.5,
            }
            if column == "wind_direction":
                values = normalized.map(direction_map).fillna(normalized)
            else:
                values = normalized.map(
                    {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0}
                )
        df[column] = pd.to_numeric(values, errors="coerce").fillna(default)

    if not wind_available_present:
        # Infer availability for exports that carry wind values but predate
        # the explicit availability column.
        df["wind_available"] = wind_speed_valid.astype(float)
    elif raw_wind_available is not None and raw_wind_available.isna().any():
        df["wind_available"] = raw_wind_available.where(
            raw_wind_available.notna(), wind_speed_valid.astype(float)
        )
    if not nwp_available_present:
        # Forecast exports that predate the explicit availability bit can
        # still be used safely when they contain a probability value.
        df["nwp_available"] = nwp_probability_valid.astype(float)
    elif raw_nwp_available is not None and raw_nwp_available.isna().any():
        df["nwp_available"] = raw_nwp_available.where(
            raw_nwp_available.notna(), nwp_probability_valid.astype(float)
        )

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

    # Wind is optional and may be absent from older CSV/DB snapshots. Keep an
    # explicit availability bit so zero-filled legacy rows are not mistaken
    # for calm air. Meteorological direction is converted to u/v components
    # (direction the wind comes from), which avoids the 359°/0° discontinuity.
    wind_available = df["wind_available"].clip(0.0, 1.0)
    speed = df["wind_speed"].clip(lower=0.0)
    gust = df["wind_gust"].clip(lower=0.0)
    speed = speed.where(wind_available > 0.0, 0.0)
    gust = gust.where(wind_available > 0.0, 0.0)
    direction = df["wind_direction"].mod(360.0)
    radians = np.deg2rad(direction)
    df["wind_available"] = wind_available
    df["wind_missing"] = (wind_available <= 0.0).astype(float)
    df["wind_speed_now"] = speed
    df["wind_gust_now"] = gust
    df["wind_u_now"] = -speed * np.sin(radians)
    df["wind_v_now"] = -speed * np.cos(radians)
    df["wind_direction_sin"] = np.sin(radians).where(wind_available > 0.0, 0.0)
    df["wind_direction_cos"] = np.cos(radians).where(wind_available > 0.0, 0.0)
    features.extend([
        "wind_available", "wind_missing", "wind_speed_now", "wind_gust_now",
        "wind_u_now", "wind_v_now", "wind_direction_sin", "wind_direction_cos",
    ])

    nwp_available = df["nwp_available"].clip(0.0, 1.0)
    nwp_probability = df["nwp_precipitation_probability"].clip(0.0, 100.0)
    nwp_precipitation = df["nwp_precipitation"].clip(lower=0.0)
    nwp_wind_speed = df["nwp_wind_speed"].clip(lower=0.0)
    nwp_wind_direction = df["nwp_wind_direction"].mod(360.0)
    nwp_radians = np.deg2rad(nwp_wind_direction)
    # A stale/missing forecast is never represented as a valid zero-valued
    # forecast.  Keep its values numerically bounded but expose both bits and
    # age so a candidate model can learn a safe fallback.
    nwp_probability = nwp_probability.where(nwp_available > 0.0, 0.0)
    nwp_precipitation = nwp_precipitation.where(nwp_available > 0.0, 0.0)
    nwp_wind_speed = nwp_wind_speed.where(nwp_available > 0.0, 0.0)
    nwp_age = df["nwp_age_seconds"].clip(lower=0.0)
    df["nwp_available"] = nwp_available
    df["nwp_missing"] = (nwp_available <= 0.0).astype(float)
    df["nwp_precipitation_probability_now"] = nwp_probability
    df["nwp_precipitation_now"] = nwp_precipitation
    df["nwp_weathercode_now"] = df["nwp_weathercode"].clip(lower=0.0).where(nwp_available > 0.0, 0.0)
    df["nwp_wind_speed_now"] = nwp_wind_speed
    df["nwp_wind_u_now"] = -nwp_wind_speed * np.sin(nwp_radians)
    df["nwp_wind_v_now"] = -nwp_wind_speed * np.cos(nwp_radians)
    df["nwp_cloud_cover_now"] = df["nwp_cloud_cover"].clip(0.0, 100.0).where(nwp_available > 0.0, 0.0)
    df["nwp_lead_hours_now"] = df["nwp_lead_hours"].clip(lower=0.0).where(nwp_available > 0.0, 0.0)
    df["nwp_age_seconds_now"] = nwp_age.where(nwp_available > 0.0, 999999.0)
    features.extend([
        "nwp_available", "nwp_missing", "nwp_precipitation_probability_now",
        "nwp_precipitation_now", "nwp_weathercode_now", "nwp_wind_speed_now",
        "nwp_wind_u_now", "nwp_wind_v_now", "nwp_cloud_cover_now",
        "nwp_lead_hours_now", "nwp_age_seconds_now",
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
    # Row-based lags assume one observation per minute. Repair one missed
    # sample (the common ~2-minute jump from a 60-second poll) before segmenting
    # so a transient transport hiccup does not suppress predictions for 120m.
    # Duplicates, clock reversals, and longer gaps still restart the segment.
    df = _repair_short_gaps(df)
    time_gap = df["timestamp"].diff()
    irregular = time_gap.lt(pd.Timedelta(seconds=MIN_VALID_INTERVAL_SECONDS)) | time_gap.gt(
        pd.Timedelta(seconds=SHORT_GAP_MAX_SECONDS)
    )
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

    # --- Optional illuminance / cloud-shadow context ---
    add_illuminance_features(df, group, features)

    # --- Augmented physical features (improve longer lead time, esp. 30m) ---
    # Short, timestamp-aware rates expose the direction and speed of local
    # changes without replacing the legacy raw-delta features above.
    add_time_rate_features(
        df, group, ["pressure", "humidity", "temp"], [5, 10, 30], features
    )
    add_time_rate_features(
        df, group, ["wind_speed", "wind_gust"], [10, 30, 60], features
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
