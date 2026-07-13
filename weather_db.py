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
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_state (
                    id TINYINT PRIMARY KEY,
                    is_raining_now BOOLEAN NOT NULL,
                    any_rain_alert BOOLEAN NOT NULL,
                    next_rain_alert_horizon VARCHAR(8),
                    consecutive_alert_count INT NOT NULL DEFAULT 0,
                    radar_alert_count INT NOT NULL DEFAULT 0,
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


def insert_reading(reading_time_iso, temp, humidity, pressure, rain_flag, rain, rain_sensor=None, light=None):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT IGNORE INTO weather_readings
                    (reading_time, temp, humidity, pressure, rain_flag, rain, rain_sensor, light)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                SELECT reading_time, temp, humidity, pressure, rain_flag
                FROM weather_readings
                WHERE reading_time >= %s AND reading_time < %s
                ORDER BY reading_time
                """,
                (_normalize_reading_time(start_time), _normalize_reading_time(end_time)),
            )
            return cursor.fetchall()


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
):
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO alert_state (
                    id, is_raining_now, any_rain_alert, next_rain_alert_horizon,
                    consecutive_alert_count, radar_alert_count
                )
                VALUES (1, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    is_raining_now = VALUES(is_raining_now),
                    any_rain_alert = VALUES(any_rain_alert),
                    next_rain_alert_horizon = VALUES(next_rain_alert_horizon),
                    consecutive_alert_count = VALUES(consecutive_alert_count),
                    radar_alert_count = VALUES(radar_alert_count)
                """,
                (
                    is_raining_now,
                    any_rain_alert,
                    next_rain_alert_horizon,
                    consecutive_alert_count,
                    radar_alert_count,
                ),
            )
