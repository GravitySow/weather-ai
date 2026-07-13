import csv
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import ha_publisher
import telegram_bot
import telegram_notifier
import weather_db
from predict_weather_ai import predict


DATA_DIR = Path(os.getenv("WEATHER_DATA_DIR", "dataset"))
MODEL_KIND = os.getenv("WEATHER_MODEL_KIND", "rf")

# Require the model to predict "rain incoming" for this many consecutive
# /reading calls (~1/min) before actually notifying Telegram. A single noisy
# minute crossing the tuned threshold used to fire an alert immediately, then
# flip back the next minute with no rain ever arriving. Requiring a streak
# filters out that kind of one-off blip at the cost of a ~N-minute delay on
# genuine alerts.
RAIN_ALERT_CONFIRM_STREAK = int(os.getenv("RAIN_ALERT_CONFIRM_STREAK", "3"))

# Radar-derived alert (radar_nowcast.py): unlike the ML alert above, this
# signal has never been backtested — RainViewer's free tier has no nowcast
# archive, so there's no historical data to validate precision/recall against.
# Deliberately more conservative than the ML streak: a higher intensity floor
# (require at least "moderate" rain somewhere nearby, not just a trace) and a
# longer confirm streak, to reduce false-alarm risk while it's unproven.
RADAR_ALERT_CONFIRM_STREAK = int(os.getenv("RADAR_ALERT_CONFIRM_STREAK", "5"))
RADAR_ALERT_MIN_INTENSITY = int(os.getenv("RADAR_ALERT_MIN_INTENSITY", "2"))

logger = logging.getLogger(__name__)

app = FastAPI(title="Weather AI API")


@app.on_event("startup")
def on_startup():
    try:
        weather_db.init_db()
    except Exception:
        logger.exception("DB init failed; continuing without DB persistence")

    threading.Thread(target=telegram_bot.poll_loop, daemon=True).start()
    threading.Thread(target=telegram_bot.schedule_loop, daemon=True).start()


class WeatherReading(BaseModel):
    """Extra device-computed fields (dh15, dp1h, nwp, ...) are accepted and
    ignored — the server derives its own features from the raw columns below
    so train/inference stays consistent with weather_features_lib."""

    model_config = ConfigDict(populate_by_name=True)

    timestamp: str | None = Field(
        default=None,
        description="ISO timestamp. If omitted, server UTC time is used.",
    )
    temp: float
    humidity: float
    pressure: float
    rain_flag: float = 0.0
    rain: bool | int | float | str | None = None
    rain_sensor: float | int | None = Field(default=None, alias="rainSensor")
    light: float | int | str | dict | None = None


def normalize_timestamp(timestamp):
    if timestamp:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    else:
        parsed = datetime.now(timezone.utc)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def csv_path_for(timestamp):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"history_{timestamp.date().isoformat()}.csv"


def append_reading(reading):
    timestamp = normalize_timestamp(reading.timestamp)
    csv_path = csv_path_for(timestamp)
    file_exists = csv_path.exists()

    row = {
        "timestamp": timestamp.isoformat(),
        "temp": reading.temp,
        "humidity": reading.humidity,
        "pressure": reading.pressure,
        "rain": reading.rain if reading.rain is not None else bool(reading.rain_flag),
        "rain_flag": reading.rain_flag,
    }

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["timestamp", "temp", "humidity", "pressure", "rain", "rain_flag"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return csv_path, row


def save_reading_to_db(reading, row):
    try:
        weather_db.insert_reading(
            reading_time_iso=row["timestamp"],
            temp=float(row["temp"]),
            humidity=float(row["humidity"]),
            pressure=float(row["pressure"]),
            rain_flag=float(row["rain_flag"]),
            rain=row["rain"],
            rain_sensor=reading.rain_sensor,
            light=reading.light,
        )
    except Exception:
        logger.exception("Failed to write reading to DB; continuing with CSV only")


def notify_rain_state_change(prediction):
    is_raining_now = prediction["is_raining_now"]
    any_rain_alert = prediction["any_rain_alert"]
    next_horizon = prediction["next_rain_alert_horizon"]

    radar = prediction.get("radar") or {}
    radar_signal = bool(
        radar.get("available")
        and not is_raining_now
        and radar.get("nearby_max_intensity", 0) >= RADAR_ALERT_MIN_INTENSITY
        and radar.get("trend_rising")
    )

    try:
        previous = weather_db.get_alert_state()
    except Exception:
        logger.exception("Could not read alert state from DB; skipping Telegram notification")
        return

    previous_raining = bool(previous["is_raining_now"]) if previous else False
    previous_count = int(previous.get("consecutive_alert_count", 0)) if previous else 0
    previous_confirmed = previous_count >= RAIN_ALERT_CONFIRM_STREAK

    previous_radar_count = int(previous.get("radar_alert_count", 0)) if previous else 0
    previous_radar_confirmed = previous_radar_count >= RADAR_ALERT_CONFIRM_STREAK

    consecutive_count = previous_count + 1 if any_rain_alert else 0
    confirmed_alert = consecutive_count >= RAIN_ALERT_CONFIRM_STREAK

    radar_count = previous_radar_count + 1 if radar_signal else 0
    confirmed_radar_alert = radar_count >= RADAR_ALERT_CONFIRM_STREAK

    if is_raining_now and not previous_raining:
        telegram_notifier.send_message(
            f"\U0001F327️ ฝนเริ่มตกแล้ว ({prediction['timestamp']})\n"
            f"อุณหภูมิ {prediction['temp']:.1f}°C "
            f"ความชื้น {prediction['humidity']:.0f}%"
        )
    elif not is_raining_now and previous_raining:
        telegram_notifier.send_message(
            f"☀️ ฝนหยุดตกแล้ว ({prediction['timestamp']})"
        )

    if confirmed_alert and not previous_confirmed:
        telegram_notifier.send_message(
            f"⚠️ คาดว่าฝนจะตกใน {next_horizon} จากนี้"
        )
    elif not confirmed_alert and previous_confirmed and not is_raining_now:
        telegram_notifier.send_message(
            "✅ ยกเลิกการแจ้งเตือนฝน "
            "ยังไม่มีฝนตกในช่วงนี้"
        )

    if confirmed_radar_alert and not previous_radar_confirmed:
        telegram_notifier.send_message(
            "\U0001F4E1 เรดาร์ตรวจพบฝนก่อตัวใกล้เคียง คาดว่าจะมีฝนเร็วๆ นี้ "
            "(สัญญาณทดลอง ยังไม่ผ่านการพิสูจน์ความแม่นยำ)"
        )
    elif not confirmed_radar_alert and previous_radar_confirmed and not is_raining_now:
        telegram_notifier.send_message(
            "✅ ยกเลิกการแจ้งเตือนฝนจากเรดาร์ ยังไม่มีฝนตกในช่วงนี้"
        )

    try:
        weather_db.set_alert_state(
            is_raining_now, confirmed_alert, next_horizon, consecutive_count, radar_count
        )
    except Exception:
        logger.exception("Could not persist alert state to DB")


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_kind": MODEL_KIND,
        "data_dir": str(DATA_DIR),
    }


@app.post("/reading")
def add_reading(reading: WeatherReading):
    csv_path, row = append_reading(reading)
    save_reading_to_db(reading, row)

    try:
        prediction = predict(model_kind=MODEL_KIND)
    except ValueError as exc:
        return {
            "saved": True,
            "csv": str(csv_path),
            "row": row,
            "prediction_ready": False,
            "message": str(exc),
        }
    except Exception as exc:
        logger.exception("predict() failed in /reading")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        notify_rain_state_change(prediction)
    except Exception:
        logger.exception("notify_rain_state_change failed; continuing")

    try:
        ha_publisher.publish_prediction(prediction)
    except Exception:
        logger.exception("ha_publisher.publish_prediction failed; continuing")

    return {
        "saved": True,
        "csv": str(csv_path),
        "row": row,
        "prediction_ready": True,
        "prediction": prediction,
    }


@app.get("/predict")
def predict_latest(model: str | None = None):
    model_kind = model or MODEL_KIND
    if model_kind not in ["xgb", "rf"]:
        raise HTTPException(status_code=400, detail='model must be "xgb" or "rf"')

    try:
        return predict(model_kind=model_kind)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
