"""MariaDB persistence for weather readings and rain-alert state.

Additive to the existing CSV storage in weather_api.py — CSV remains the
source of truth for train_weather_ai.py / predict_weather_ai.py. This module
lets readings also land in a queryable DB and lets the API remember the last
notified rain state across restarts.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import pymysql
from pymysql.cursors import DictCursor

logger = logging.getLogger(__name__)

# `os.getenv(name, default)` only falls back when the var is unset — but
# bashio::config (Home Assistant) always exports a value, empty string
# included, for unset optional options. `or` catches both cases.
DB_HOST = os.getenv("DB_HOST") or "localhost"
DB_PORT = int(os.getenv("DB_PORT") or 3306)
DB_USER = os.getenv("DB_USER") or "weather"
DB_PASSWORD = os.getenv("DB_PASSWORD") or "weather"
DB_NAME = os.getenv("DB_NAME") or "weather_ai"


def _normalize_reading_time(value):
    """Return a naive 'YYYY-MM-DD HH:MM:SS.ffffff' string for MariaDB DATETIME(6).

    Callers pass either a pandas/py datetime or an ISO string that may carry a
    UTC offset (e.g. from timestamp.isoformat()) — MariaDB's DATETIME columns
    have no timezone concept and reject/mangle the offset suffix, so convert
    to naive UTC before it ever reaches SQL.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


@contextmanager
def get_connection():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=DictCursor,
        autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS weather_readings (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    reading_time DATETIME(6) NOT NULL,
                    temp DOUBLE NOT NULL,
                    humidity DOUBLE NOT NULL,
                    pressure DOUBLE NOT NULL,
                    rain_flag DOUBLE NOT NULL,
                    rain VARCHAR(16),
                    rain_sensor DOUBLE,
                    light VARCHAR(64),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_reading_time (reading_time)
                ) ENGINE=InnoDB
                """
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS rain_sensor DOUBLE"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS light VARCHAR(64)"
            )
            # Radar (RainViewer) / cloud (Open-Meteo) shadow signals, persisted
            # per-reading so there's a real history to backtest against and,
            # eventually, retrain the rain-onset model with as features —
            # previously these were only published live to HA, never stored.
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS radar_available BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS radar_point_intensity TINYINT"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS radar_nearby_max_intensity TINYINT"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS radar_trend_rising BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS cloud_available BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS cloud_cover_now TINYINT"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS cloud_cover_low_now TINYINT"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS cloud_trend_rising BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS wind_available BOOLEAN"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS wind_speed DOUBLE"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS wind_gust DOUBLE"
            )
            cursor.execute(
                "ALTER TABLE weather_readings ADD COLUMN IF NOT EXISTS wind_direction DOUBLE"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_state (
                    id TINYINT PRIMARY KEY,
                    is_raining_now BOOLEAN NOT NULL,
                    any_rain_alert BOOLEAN NOT NULL,
                    next_rain_alert_horizon VARCHAR(8),
                    consecutive_alert_count INT NOT NULL DEFAULT 0,
                    radar_alert_count INT NOT NULL DEFAULT 0,
                    last_notification_at DATETIME(6),
                    last_notification_kind VARCHAR(32),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
                """
            )
            cursor.execute(
                "ALTER TABLE alert_state ADD COLUMN IF NOT EXISTS consecutive_alert_count INT NOT NULL DEFAULT 0"
            )
            cursor.execute(
                "ALTER TABLE alert_state ADD COLUMN IF NOT EXISTS radar_alert_count INT NOT NULL DEFAULT 0"
            )
            cursor.execute(
                "ALTER TABLE alert_state ADD COLUMN IF NOT EXISTS last_notification_at DATETIME(6)"
            )
            cursor.execute(
                "ALTER TABLE alert_state ADD COLUMN IF NOT EXISTS last_notification_kind VARCHAR(32)"
            )
            # Tracks the last calendar date (Bangkok time) each scheduled
            # Telegram push (hour=6 or 21) actually went out, so a restart
            # that lands after a slot's time can tell "already sent today"
            # apart from "missed it" and send a catch-up — see
            # telegram_bot._catch_up_missed_schedule().
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_message_log (
                    hour TINYINT PRIMARY KEY,
                    last_sent_date DATE NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
                """
            )
            # Every issued forecast, one row per (source_layer, lead_minutes)
            # — see plan.md Phase 5. Built ahead of the Phase 2 radar-advection
            # work specifically so a new source_layer value (e.g. "radar")
            # just works with forecast_scoring.py without further changes.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_log (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    issued_at DATETIME(6) NOT NULL,
                    valid_at DATETIME(6) NOT NULL,
                    lead_minutes INT NOT NULL,
                    rain_prob DOUBLE NOT NULL,
                    source_layer VARCHAR(32) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version VARCHAR(128),
                    unadjusted_rain_prob DOUBLE,
                    probability_postprocessing VARCHAR(32),
                    data_confidence DOUBLE,
                    data_age_seconds DOUBLE,
                    feature_coverage DOUBLE,
                    KEY idx_valid_at (valid_at),
                    KEY idx_issued_at (issued_at)
                ) ENGINE=InnoDB
                """
            )
            for statement in (
                "ALTER TABLE forecast_log ADD COLUMN IF NOT EXISTS model_version VARCHAR(128)",
                "ALTER TABLE forecast_log ADD COLUMN IF NOT EXISTS unadjusted_rain_prob DOUBLE",
                "ALTER TABLE forecast_log ADD COLUMN IF NOT EXISTS probability_postprocessing VARCHAR(32)",
                "ALTER TABLE forecast_log ADD COLUMN IF NOT EXISTS data_confidence DOUBLE",
                "ALTER TABLE forecast_log ADD COLUMN IF NOT EXISTS data_age_seconds DOUBLE",
                "ALTER TABLE forecast_log ADD COLUMN IF NOT EXISTS feature_coverage DOUBLE",
            ):
                cursor.execute(statement)
            # Nightly aggregate scores from forecast_scoring.py. One row per
            # (as_of_date, source_layer, lead_minutes) — re-running scoring
            # for a date that's already scored upserts rather than duplicating.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_score_log (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    as_of_date DATE NOT NULL,
                    source_layer VARCHAR(32) NOT NULL,
                    lead_minutes INT NOT NULL,
                    n INT NOT NULL,
                    brier_score DOUBLE NOT NULL,
                    climatology_brier DOUBLE NOT NULL,
                    skill_score DOUBLE,
                    mean_predicted DOUBLE NOT NULL,
                    mean_observed DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_score (as_of_date, source_layer, lead_minutes)
                ) ENGINE=InnoDB
                """
            )
            # Delivery record for every outbound Telegram message — see
            # plan.md "Phase 6b". Previously there was no way to tell
            # whether a message was sent, retried, or silently lost.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS message_log (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    sent_at DATETIME(6) NOT NULL,
                    success BOOLEAN NOT NULL,
                    attempts INT NOT NULL,
                    char_length INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    KEY idx_sent_at (sent_at)
                ) ENGINE=InnoDB
                """
            )
            # Every Open-Meteo hourly temperature forecast, one row per
            # (issued_hour, lead_hours) — see plan.md "Phase D" (MOS bias
            # correction). issued_hour/valid_at are naive UTC, same convention
            # as weather_readings.reading_time, so a later bias job can join
            # this straight against observed temp with no timezone handling.
            # Unique key makes re-logging within the same hour a safe no-op —
            # nwp_forecast.py's own fetch is only 15-min cached, but this table
            # only needs one row per hour per lead.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nwp_forecast_log (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    issued_hour DATETIME NOT NULL,
                    valid_at DATETIME NOT NULL,
                    lead_hours INT NOT NULL,
                    temp_forecast DOUBLE NOT NULL,
                    temp_forecast_raw DOUBLE,
                    temp_bias_correction DOUBLE,
                    temp_postprocessing VARCHAR(32),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_nwp (issued_hour, lead_hours),
                    KEY idx_valid_at (valid_at)
                ) ENGINE=InnoDB
                """
            )
            for statement in (
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS temp_forecast_raw DOUBLE",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS temp_bias_correction DOUBLE",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS temp_postprocessing VARCHAR(32)",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS precipitation_probability DOUBLE",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS precipitation DOUBLE",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS weathercode INT",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS wind_speed DOUBLE",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS wind_direction DOUBLE",
                "ALTER TABLE nwp_forecast_log ADD COLUMN IF NOT EXISTS cloud_cover DOUBLE",
            ):
                cursor.execute(statement)


def insert_reading(
    reading_time_iso,
    temp,
    humidity,
    pressure,
    rain_flag,
    rain,
    rain_sensor=None,
    light=None,
    radar=None,
    cloud=None,
    wind_available=None,
    wind_speed=None,
    wind_gust=None,
    wind_direction=None,
):
    """radar/cloud: the dicts returned by radar_nowcast.get_radar_signal() /
    cloud_nowcast.get_cloud_signal(), or None if not fetched. Stored as-is
    (only the fields we care about) so there's a real history to backtest
    against later — see weather_model-findings memory for why this matters."""
    radar = radar or {}
    cloud = cloud or {}
    if wind_available is None:
        wind_available = any(value is not None for value in (wind_speed, wind_gust, wind_direction))

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT IGNORE INTO weather_readings
                    (reading_time, temp, humidity, pressure, rain_flag, rain, rain_sensor, light,
                     wind_available, wind_speed, wind_gust, wind_direction,
                     radar_available, radar_point_intensity, radar_nearby_max_intensity, radar_trend_rising,
                     cloud_available, cloud_cover_now, cloud_cover_low_now, cloud_trend_rising)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    _normalize_reading_time(reading_time_iso),
                    temp,
                    humidity,
                    pressure,
                    rain_flag,
                    str(rain),
                    rain_sensor,
                    None if light is None else str(light),
                    bool(wind_available),
                    wind_speed,
                    wind_gust,
                    wind_direction,
                    radar.get("available", False),
                    radar.get("point_intensity"),
                    radar.get("nearby_max_intensity"),
                    radar.get("trend_rising"),
                    cloud.get("available", False),
                    cloud.get("cloud_cover_now"),
                    cloud.get("cloud_cover_low_now"),
                    cloud.get("trend_rising"),
                ),
            )


def insert_readings_bulk(rows):
    """rows: iterable of (reading_time, temp, humidity, pressure, rain_flag, rain) tuples."""
    rows = [
        (_normalize_reading_time(reading_time), temp, humidity, pressure, rain_flag, rain)
        for reading_time, temp, humidity, pressure, rain_flag, rain in rows
    ]
    if not rows:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT IGNORE INTO weather_readings
                    (reading_time, temp, humidity, pressure, rain_flag, rain)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            return cursor.rowcount


def get_readings(start_time, end_time):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT reading_time, temp, humidity, pressure, rain_flag,
                       wind_available, wind_speed, wind_gust, wind_direction,
                       radar_available, radar_point_intensity, radar_nearby_max_intensity,
                       radar_trend_rising, cloud_available, cloud_cover_now,
                       cloud_cover_low_now, cloud_trend_rising
                FROM weather_readings
                WHERE reading_time >= %s AND reading_time < %s
                ORDER BY reading_time
                """,
                (_normalize_reading_time(start_time), _normalize_reading_time(end_time)),
            )
            return cursor.fetchall()


def insert_message_log(success, attempts, char_length):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO message_log (sent_at, success, attempts, char_length)
                VALUES (%s, %s, %s, %s)
                """,
                (_normalize_reading_time(datetime.utcnow()), success, attempts, char_length),
            )


def insert_forecast_log(rows, metadata=None):
    """Insert forecast rows with optional provenance and data-quality metadata.

    The five-tuple input remains backward compatible. ``metadata`` may be a
    mapping keyed by ``(source_layer, lead_minutes)`` or by source layer, and
    accepts model_version, unadjusted_rain_prob, probability_postprocessing,
    data_confidence, data_age_seconds and feature_coverage.
    """
    metadata = metadata or {}
    normalized = []
    for issued_at, valid_at, lead_minutes, rain_prob, source_layer in rows:
        item = metadata.get((source_layer, int(lead_minutes)), metadata.get(source_layer, {}))
        normalized.append(
            (
                _normalize_reading_time(issued_at),
                _normalize_reading_time(valid_at),
                lead_minutes,
                rain_prob,
                source_layer,
                item.get("model_version"),
                item.get("unadjusted_rain_prob"),
                item.get("probability_postprocessing"),
                item.get("data_confidence"),
                item.get("data_age_seconds"),
                item.get("feature_coverage"),
            )
        )
    if not normalized:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO forecast_log (
                    issued_at, valid_at, lead_minutes, rain_prob, source_layer,
                    model_version, unadjusted_rain_prob, probability_postprocessing,
                    data_confidence, data_age_seconds, feature_coverage
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                normalized,
            )
            return cursor.rowcount


def insert_nwp_forecast_log(rows, metadata=None):
    """rows: iterable of (issued_hour, valid_at, lead_hours, temp_forecast).
    INSERT IGNORE so a repeat log within the same hour (e.g. after a restart)
    is a safe no-op rather than a duplicate-key error — see plan.md Phase D."""
    metadata = metadata or {}
    normalized = []
    for issued_hour, valid_at, lead_hours, temp_forecast in rows:
        item = metadata.get(int(lead_hours), {})
        normalized.append((
            _normalize_reading_time(issued_hour), _normalize_reading_time(valid_at),
            lead_hours, temp_forecast, item.get("temp_forecast_raw"),
            item.get("temp_bias_correction"), item.get("temp_postprocessing"),
            item.get("precipitation_probability"), item.get("precipitation"),
            item.get("weathercode"), item.get("wind_speed"),
            item.get("wind_direction"), item.get("cloud_cover"),
        ))
    rows = normalized
    if not rows:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT IGNORE INTO nwp_forecast_log (
                    issued_hour, valid_at, lead_hours, temp_forecast,
                    temp_forecast_raw, temp_bias_correction, temp_postprocessing,
                    precipitation_probability, precipitation, weathercode,
                    wind_speed, wind_direction, cloud_cover
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            return cursor.rowcount


def get_forecast_log(start_time, end_time):
    """Rows whose issued_at falls in [start_time, end_time)."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT issued_at, valid_at, lead_minutes, rain_prob, source_layer,
                       model_version, unadjusted_rain_prob, probability_postprocessing,
                       data_confidence, data_age_seconds, feature_coverage
                FROM forecast_log
                WHERE issued_at >= %s AND issued_at < %s
                ORDER BY valid_at
                """,
                (_normalize_reading_time(start_time), _normalize_reading_time(end_time)),
            )
            return cursor.fetchall()


def get_nwp_forecast_rows(start_time=None, end_time=None):
    """Return logged NWP rows for offline calibration/replay.

    This is read-only and intentionally separate from the runtime insert path;
    interval/bias fitting can therefore use a snapshot without changing the
    production schema or forecast state.
    """
    clauses, params = [], []
    if start_time is not None:
        clauses.append("valid_at >= %s")
        params.append(_normalize_reading_time(start_time))
    if end_time is not None:
        clauses.append("valid_at < %s")
        params.append(_normalize_reading_time(end_time))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Older production snapshots predate the raw/correction columns;
            # calibration only needs the canonical forecast value, so keep the
            # read path compatible without requiring a migration first.
            cursor.execute(
                "SELECT issued_hour, valid_at, lead_hours, temp_forecast, "
                "precipitation_probability, precipitation, weathercode, "
                "wind_speed, wind_direction, cloud_cover "
                f"FROM nwp_forecast_log{where} ORDER BY valid_at, lead_hours",
                tuple(params),
            )
            return cursor.fetchall()


def get_temperature_observations(start_time=None, end_time=None):
    """Return timestamped temperature observations for calibration jobs."""
    clauses, params = [], []
    if start_time is not None:
        clauses.append("reading_time >= %s")
        params.append(_normalize_reading_time(start_time))
    if end_time is not None:
        clauses.append("reading_time < %s")
        params.append(_normalize_reading_time(end_time))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT reading_time, temp FROM weather_readings{where} ORDER BY reading_time",
                tuple(params),
            )
            return cursor.fetchall()


def insert_forecast_score(as_of_date, score):
    """score: dict with source_layer, lead_minutes, n, brier_score,
    climatology_brier, skill_score, mean_predicted, mean_observed. Upserts
    on (as_of_date, source_layer, lead_minutes) so re-running scoring for an
    already-scored date replaces the row rather than duplicating it."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO forecast_score_log (
                    as_of_date, source_layer, lead_minutes, n, brier_score,
                    climatology_brier, skill_score, mean_predicted, mean_observed
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    n = VALUES(n),
                    brier_score = VALUES(brier_score),
                    climatology_brier = VALUES(climatology_brier),
                    skill_score = VALUES(skill_score),
                    mean_predicted = VALUES(mean_predicted),
                    mean_observed = VALUES(mean_observed)
                """,
                (
                    as_of_date,
                    score["source_layer"],
                    score["lead_minutes"],
                    score["n"],
                    score["brier_score"],
                    score["climatology_brier"],
                    score["skill_score"],
                    score["mean_predicted"],
                    score["mean_observed"],
                ),
            )


def get_last_sent_date(hour):
    """Return the date() a scheduled push for this hour last actually went
    out, or None if it's never been sent."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT last_sent_date FROM scheduled_message_log WHERE hour = %s", (hour,)
            )
            row = cursor.fetchone()
            return row["last_sent_date"] if row else None


def mark_scheduled_sent(hour, sent_date):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO scheduled_message_log (hour, last_sent_date)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE last_sent_date = VALUES(last_sent_date)
                """,
                (hour, sent_date),
            )


def get_alert_state():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM alert_state WHERE id = 1")
            return cursor.fetchone()


def set_alert_state(
    is_raining_now,
    any_rain_alert,
    next_rain_alert_horizon,
    consecutive_alert_count=0,
    radar_alert_count=0,
    last_notification_at=None,
    last_notification_kind=None,
):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO alert_state (
                    id, is_raining_now, any_rain_alert, next_rain_alert_horizon,
                    consecutive_alert_count, radar_alert_count,
                    last_notification_at, last_notification_kind
                )
                VALUES (1, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    is_raining_now = VALUES(is_raining_now),
                    any_rain_alert = VALUES(any_rain_alert),
                    next_rain_alert_horizon = VALUES(next_rain_alert_horizon),
                    consecutive_alert_count = VALUES(consecutive_alert_count),
                    radar_alert_count = VALUES(radar_alert_count),
                    last_notification_at = VALUES(last_notification_at),
                    last_notification_kind = VALUES(last_notification_kind)
                """,
                (
                    is_raining_now,
                    any_rain_alert,
                    next_rain_alert_horizon,
                    consecutive_alert_count,
                    radar_alert_count,
                    last_notification_at,
                    last_notification_kind,
                ),
            )
