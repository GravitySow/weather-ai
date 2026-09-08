import csv
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

import cloud_nowcast
import alert_policy
import forecast_revision
import forecast_scoring
import forecast_service
import ha_publisher
import ha_source
import nwp_forecast
import radar_nowcast
import official_warnings
import rain_stop
import telegram_bot
import telegram_notifier
import weather_db
from predict_weather_ai import predict


DATA_DIR = Path(os.getenv("WEATHER_DATA_DIR", "dataset"))
MODEL_KIND = os.getenv("WEATHER_MODEL_KIND", "rf")

# Thailand has no DST, so a fixed offset is exact — same convention already
# duplicated in forecast_scoring.py/ha_publisher.py/telegram_bot.py.
BANGKOK_OFFSET = timedelta(hours=7)

# In-process guard so a ~1/min /reading call doesn't retry the same hour's
# NWP log 60 times; the DB's UNIQUE KEY (issued_hour, lead_hours) is the real
# backstop across restarts, this is just to avoid pointless executemany calls.
_last_nwp_log_hour = None

# Require the model to predict "rain incoming" for this many consecutive
# /reading calls (~1/min) before actually notifying Telegram. A single noisy
# minute crossing the tuned threshold used to fire an alert immediately, then
# flip back the next minute with no rain ever arriving. Requiring a streak
# filters out that kind of one-off blip at the cost of a ~N-minute delay on
# genuine alerts.
RAIN_ALERT_CONFIRM_STREAK = int(os.getenv("RAIN_ALERT_CONFIRM_STREAK", "3"))

# A rain sensor can report a dry minute between drops.  Keep the last
# confirmed rain state until this many consecutive source polls are dry, so a
# brief flicker neither creates a false "rain stopped" notification nor hides
# the later return of genuine rain behind the rain-start cooldown.
RAIN_STOP_CONFIRM_STREAK = max(1, int(os.getenv("RAIN_STOP_CONFIRM_STREAK", "3")))

# Radar-derived alert (radar_nowcast.py): unlike the ML alert above, this
# signal has never been backtested — RainViewer's free tier has no nowcast
# archive, so there's no historical data to validate precision/recall against.
# Deliberately more conservative than the ML streak: a higher intensity floor
# (require at least "moderate" rain somewhere nearby, not just a trace) and a
# longer confirm streak, to reduce false-alarm risk while it's unproven.
# Gates on close_max_intensity (max of the 0-10km/10-25km rings), not the
# deprecated whole-tile nearby_max_intensity — see radar_nowcast.py's
# 2026-07-21 geometry-fix note for why that distinction matters.
RADAR_ALERT_CONFIRM_STREAK = int(os.getenv("RADAR_ALERT_CONFIRM_STREAK", "5"))
RADAR_ALERT_MIN_INTENSITY = int(os.getenv("RADAR_ALERT_MIN_INTENSITY", "2"))
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "1800"))
REVISION_ALERT_ENABLED = (os.getenv("REVISION_ALERT_ENABLED", "0").lower() in ("1", "true", "yes", "on"))
REVISION_ALERT_COOLDOWN_SECONDS = int(os.getenv("REVISION_ALERT_COOLDOWN_SECONDS", "3600"))
REVISION_POLL_SECONDS = int(os.getenv("REVISION_POLL_SECONDS", "900"))

logger = logging.getLogger(__name__)

app = FastAPI(title="Weather AI API")

# HA polling must not wait for the comparatively expensive prediction path
# (DB/radar/cloud lookups and model inference). Keeping observations in a
# bounded FIFO preserves their source timestamps while the worker serializes
# writes/predictions and prevents poll gaps caused by callback latency.
_HA_INGEST_QUEUE = queue.Queue(maxsize=300)


@app.on_event("startup")
def on_startup():
    try:
        weather_db.init_db()
    except Exception:
        logger.exception("DB init failed; continuing without DB persistence")

    threading.Thread(target=telegram_bot.poll_loop, daemon=True).start()
    threading.Thread(target=telegram_bot.schedule_loop, daemon=True).start()
    threading.Thread(target=forecast_scoring.schedule_loop, daemon=True).start()
    if ha_source.is_enabled():
        threading.Thread(
            target=_ha_ingest_worker,
            daemon=True,
            name="ha-ingest-worker",
        ).start()
        threading.Thread(
            target=ha_source.poll_loop,
            args=(_enqueue_homeassistant_reading,),
            daemon=True,
            name="ha-source-poll",
        ).start()
    if REVISION_ALERT_ENABLED:
        threading.Thread(target=revision_alert_loop, daemon=True).start()


def _enqueue_homeassistant_reading(payload: dict) -> None:
    """Queue a fetched observation without blocking the HA poller."""
    try:
        _HA_INGEST_QUEUE.put_nowait(payload)
    except queue.Full:
        # Never block the source poller behind a slow model. A bounded queue
        # protects memory; the status/log makes an overload actionable.
        logger.error("Home Assistant ingest queue is full; dropping observation")


def _ha_ingest_worker() -> None:
    """Serialize CSV/DB writes and inference in observation order."""
    while True:
        payload = _HA_INGEST_QUEUE.get()
        try:
            ingest_homeassistant_reading(payload)
        finally:
            _HA_INGEST_QUEUE.task_done()


def revision_alert_loop():
    """Optionally notify Telegram when a provider revision crosses thresholds.

    Disabled by default because proactive notifications require an explicit
    operator opt-in. A failed delivery is not marked as sent and is retried on
    a later poll, while a persisted revision id prevents duplicates.
    """
    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        logger.warning("Revision alerts enabled but Telegram credentials are missing; loop disabled")
        return
    while True:
        try:
            result = forecast_service.get_forecast(model_kind=MODEL_KIND)
            if result:
                revision = forecast_revision.observe(result)
                if forecast_revision.notification_allowed(revision, REVISION_ALERT_COOLDOWN_SECONDS):
                    message = telegram_bot._format_revision(revision)
                    if message and telegram_notifier.send_message(message, silent=True):
                        forecast_revision.mark_notified(revision)
        except Exception:
            logger.exception("Forecast revision alert poll failed; retrying")
        time.sleep(max(60, REVISION_POLL_SECONDS))


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
    # Optional wind covariates.  HA ingestion normalizes speed/gust to m/s;
    # direct POST callers may provide the same canonical units.
    wind_available: float | int | bool | None = Field(default=None, alias="windAvailable")
    wind_speed: float | int | None = Field(default=None, alias="windSpeed")
    wind_gust: float | int | None = Field(default=None, alias="windGust")
    wind_direction: float | int | None = Field(default=None, alias="windDirection")


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
    wind_available = reading.wind_available
    if wind_available is None:
        wind_available = 1.0 if any(
            value is not None for value in (reading.wind_speed, reading.wind_gust, reading.wind_direction)
        ) else 0.0

    row = {
        "timestamp": timestamp.isoformat(),
        "temp": reading.temp,
        "humidity": reading.humidity,
        "pressure": reading.pressure,
        "rain": reading.rain if reading.rain is not None else bool(reading.rain_flag),
        "rain_flag": reading.rain_flag,
        "wind_available": wind_available,
        "wind_speed": reading.wind_speed,
        "wind_gust": reading.wind_gust,
        "wind_direction": reading.wind_direction,
    }

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "timestamp", "temp", "humidity", "pressure", "rain", "rain_flag",
            "wind_available", "wind_speed", "wind_gust", "wind_direction",
        ]
        # Existing daily files may have the pre-wind six-column header. Keep
        # their schema while DB persistence still captures the new fields;
        # new files use the expanded schema and are fully trainable offline.
        if file_exists:
            csv_file.seek(0)
            header = csv_file.readline().strip()
            if header:
                existing_fields = next(csv.reader([header]), [])
                if existing_fields:
                    fieldnames = existing_fields
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})

    return csv_path, row


def save_reading_to_db(reading, row):
    # Fetched independently of predict()'s own result["radar"]/result["cloud"]
    # (both cached with their own TTLs, so this is cheap) so a reading is
    # still stored with radar/cloud context even on the early-return path
    # below where predict() hasn't run yet (not enough history).
    try:
        radar = radar_nowcast.get_radar_signal()
    except Exception:
        logger.exception("radar_nowcast failed while saving reading; continuing without it")
        radar = None

    try:
        cloud = cloud_nowcast.get_cloud_signal()
    except Exception:
        logger.exception("cloud_nowcast failed while saving reading; continuing without it")
        cloud = None

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
            wind_available=reading.wind_available,
            wind_speed=reading.wind_speed,
            wind_gust=reading.wind_gust,
            wind_direction=reading.wind_direction,
            radar=radar,
            cloud=cloud,
        )
    except Exception:
        logger.exception("Failed to write reading to DB; continuing with CSV only")


def notify_rain_state_change(prediction):
    observed_raining = bool(prediction["is_raining_now"])
    raw_any_rain_alert = bool(prediction["any_rain_alert"])
    next_horizon = prediction["next_rain_alert_horizon"]

    try:
        previous = weather_db.get_alert_state()
    except Exception:
        logger.exception("Could not read alert state from DB; skipping Telegram notification")
        return

    previous_raining = bool(previous["is_raining_now"]) if previous else False
    previous_dry_streak = int(previous.get("dry_streak", 0)) if previous else 0

    if observed_raining:
        dry_streak = 0
    elif previous_raining:
        dry_streak = previous_dry_streak + 1
    else:
        # Keep the stored value bounded.  The streak only matters while a
        # confirmed rain session is active, but recording a recent dry state
        # makes restarts deterministic too.
        dry_streak = min(previous_dry_streak + 1, RAIN_STOP_CONFIRM_STREAK)

    # `is_raining_now` is intentionally the debounced state, not the raw
    # sensor sample.  Until the dry streak is confirmed, suppress "incoming
    # rain" signals because the current rain event is still in progress.
    is_raining_now = bool(
        observed_raining
        or (previous_raining and dry_streak < RAIN_STOP_CONFIRM_STREAK)
    )
    any_rain_alert = raw_any_rain_alert and not is_raining_now

    radar = prediction.get("radar") or {}
    radar_signal = bool(
        radar.get("available")
        and not is_raining_now
        and radar.get("close_max_intensity", 0) >= RADAR_ALERT_MIN_INTENSITY
        and radar.get("trend_rising")
    )

    previous_count = int(previous.get("consecutive_alert_count", 0)) if previous else 0
    previous_confirmed = previous_count >= RAIN_ALERT_CONFIRM_STREAK

    previous_radar_count = int(previous.get("radar_alert_count", 0)) if previous else 0
    previous_radar_confirmed = previous_radar_count >= RADAR_ALERT_CONFIRM_STREAK

    consecutive_count = previous_count + 1 if any_rain_alert else 0
    confirmed_alert = consecutive_count >= RAIN_ALERT_CONFIRM_STREAK

    radar_count = previous_radar_count + 1 if radar_signal else 0
    confirmed_radar_alert = radar_count >= RADAR_ALERT_CONFIRM_STREAK

    # Persisting the last notification kind/time prevents an alert storm when
    # a noisy signal repeatedly crosses a threshold after a brief reset.  A
    # different state transition remains immediately actionable.
    policy_state = dict(previous or {})
    last_notification_at = policy_state.get("last_notification_at")
    last_notification_kind = policy_state.get("last_notification_kind")

    def emit(kind, text):
        nonlocal last_notification_at, last_notification_kind
        now = datetime.now(timezone.utc)
        policy_state["last_notification_at"] = last_notification_at
        policy_state["last_notification_kind"] = last_notification_kind
        if not alert_policy.can_notify(policy_state, kind, now, ALERT_COOLDOWN_SECONDS):
            logger.info("Suppressing duplicate %s notification during cooldown", kind)
            return False
        delivered = telegram_notifier.send_message(text)
        if delivered:
            last_notification_at = now
            last_notification_kind = kind
        return delivered

    alert_context = telegram_bot.format_prediction_context(prediction)

    if is_raining_now and not previous_raining:
        kind = "rain_resume" if previous else "rain_start"
        headline = "🌧️ ฝนกลับมาตกอีกครั้ง" if previous else "🌧️ ฝนเริ่มตกแล้ว"
        emit(
            kind,
            f"{headline} ({prediction['timestamp']})\n"
            f"อุณหภูมิ {prediction['temp']:.1f}°C "
            f"ความชื้น {prediction['humidity']:.0f}%\n\n{alert_context}"
        )
    elif not is_raining_now and previous_raining:
        emit(
            "rain_stop",
            f"☀️ ฝนหยุดตกแล้ว ({prediction['timestamp']})\n"
            f"<i>ยืนยันว่าเซนเซอร์แห้งต่อเนื่อง {RAIN_STOP_CONFIRM_STREAK} รอบ</i>"
        )

    if confirmed_alert and not previous_confirmed:
        emit(
            "ml_alert",
            f"⚠️ คาดว่าฝนจะตกใน {next_horizon} จากนี้\n\n{alert_context}"
        )
    elif not confirmed_alert and previous_confirmed and not is_raining_now:
        emit(
            "ml_cancel",
            "✅ ยกเลิกการแจ้งเตือนฝน "
            "ยังไม่มีฝนตกในช่วงนี้"
        )

    if confirmed_radar_alert and not previous_radar_confirmed:
        emit(
            "radar_alert",
            "\U0001F4E1 เรดาร์ตรวจพบฝนก่อตัวใกล้เคียง คาดว่าจะมีฝนเร็วๆ นี้ "
            "(สัญญาณทดลอง ยังไม่ผ่านการพิสูจน์ความแม่นยำ)"
        )
    elif not confirmed_radar_alert and previous_radar_confirmed and not is_raining_now:
        emit(
            "radar_cancel",
            "✅ ยกเลิกการแจ้งเตือนฝนจากเรดาร์ ยังไม่มีฝนตกในช่วงนี้"
        )

    try:
        weather_db.set_alert_state(
            is_raining_now, confirmed_alert, next_horizon, consecutive_count, radar_count,
            last_notification_at, last_notification_kind, dry_streak=dry_streak,
        )
    except Exception:
        logger.exception("Could not persist alert state to DB")


def log_forecast_to_db(prediction):
    """Logs every issued forecast to forecast_log so forecast_scoring.py can
    score it against what actually happens — see plan.md "Phase 5".
    Deliberately independent of predict()'s own success/failure path: it's a
    fresh derivation from `prediction`, not a second prediction call.

    Two source_layer values today:
    - "local_rf": the RF nowcast's per-horizon rain probability, always
      logged (there's always a probability to check, rain or not).
    - "radar_advection": only logged when rain_arriving is True — that's
      the one falsifiable claim ("rain reaches the sensor at this specific
      lead time") this shadow-mode signal makes; "not arriving within the
      lookahead window" isn't a specific enough claim to score the same way.
      confidence is used as a proxy for rain_prob since this signal is a
      deterministic ETA projection, not a calibrated probability — see
      radar_advection.py's module docstring on why it must stay out of any
      Telegram alert until this backtest shows it beats climatology.
    """
    # Local model horizons start at the latest sensor observation. Using request
    # time silently shifts labels whenever ingestion/inference is delayed.
    issued_at = datetime.fromisoformat(prediction["timestamp"].replace("Z", "+00:00"))
    if issued_at.tzinfo is None:
        issued_at = issued_at.replace(tzinfo=timezone.utc)
    issued_at = issued_at.astimezone(timezone.utc)
    rows = []
    metadata = {}
    for horizon, pred in prediction["predictions"].items():
        lead_minutes = int(horizon.rstrip("m"))
        source_layer = f"local_{prediction.get('model', 'rf')}"
        rows.append((
            issued_at,
            issued_at + timedelta(minutes=lead_minutes),
            lead_minutes,
            float(pred["probability"]),
            source_layer,
        ))
        metadata[(source_layer, lead_minutes)] = {
            "model_version": prediction.get("model_version"),
            "unadjusted_rain_prob": pred.get("unadjusted_probability", pred["probability"]),
            "probability_postprocessing": prediction.get("probability_postprocessing", "none"),
            "data_confidence": prediction.get("data_confidence"),
            "data_age_seconds": prediction.get("data_age_seconds"),
            "feature_coverage": prediction.get("feature_coverage"),
        }

    advection = prediction.get("radar_advection") or {}
    if advection.get("available") and advection.get("rain_arriving"):
        radar_issued_at = datetime.now(timezone.utc)
        lead_minutes = int(advection["eta_minutes"])
        rows.append((
            radar_issued_at,
            radar_issued_at + timedelta(minutes=lead_minutes),
            lead_minutes,
            float(advection["confidence"]),
            "radar_advection",
        ))
        metadata[("radar_advection", lead_minutes)] = {
            "model_version": "radar_advection",
            "unadjusted_rain_prob": float(advection["confidence"]),
            "probability_postprocessing": "none",
            "data_confidence": float(advection["confidence"]),
        }

    weather_db.insert_forecast_log(rows, metadata=metadata)


def log_nwp_forecast_to_db():
    """Logs the current Open-Meteo hourly temperature forecast to
    nwp_forecast_log, once per hour, so a later job can measure NWP bias
    against this sensor's actual readings — see plan.md "Phase D" (MOS).
    Deliberately started now, well before the bias-correction logic itself,
    since it needs ~2 weeks of accumulated (forecast, observed) pairs.

    Only logs once per hour (guarded in-process by _last_nwp_log_hour, and
    at the DB level by nwp_forecast_log's UNIQUE KEY) — nwp_forecast.py's
    own fetch is cached 15min, but /reading fires ~1/min and there's no
    value in more than one row per (issued_hour, lead_hours).
    """
    global _last_nwp_log_hour

    now_bkk = datetime.utcnow() + BANGKOK_OFFSET
    issued_hour_bkk = now_bkk.replace(minute=0, second=0, microsecond=0)
    if issued_hour_bkk == _last_nwp_log_hour:
        return
    issued_hour_utc = issued_hour_bkk - BANGKOK_OFFSET

    timeline = nwp_forecast.get_hourly_timeline(hours=24)
    rows = []
    metadata = {}
    for entry in timeline:
        temp = entry.get("temp")
        if temp is None:
            continue
        valid_at_bkk = datetime.fromisoformat(entry["time"])
        valid_at_utc = valid_at_bkk - BANGKOK_OFFSET
        lead_hours = round((valid_at_utc - issued_hour_utc).total_seconds() / 3600)
        if lead_hours < 0:
            continue
        rows.append((issued_hour_utc, valid_at_utc, lead_hours, float(temp)))
        metadata[lead_hours] = {
            "temp_forecast_raw": entry.get("temp_raw", temp),
            "temp_bias_correction": entry.get("temp_bias_correction", 0.0),
            "temp_postprocessing": "nwp_bias_correction" if entry.get(
                "temp_bias_correction_enabled"
            ) else "none",
            "precipitation_probability": entry.get("rain_prob"),
            "precipitation": entry.get("precip_mm"),
            "weathercode": entry.get("weathercode"),
            # Open-Meteo reports wind in km/h; store canonical m/s to match
            # Home Assistant wind ingestion and the shared feature builder.
            "wind_speed": (
                float(entry["wind_kmh"]) / 3.6
                if entry.get("wind_kmh") is not None else None
            ),
            "wind_direction": entry.get("wind_dir_deg"),
            "cloud_cover": entry.get("cloud_cover"),
        }

    if not rows:
        # NWP fetch unavailable this call — don't mark the hour as logged so
        # a later /reading call this same hour can retry.
        return
    _last_nwp_log_hour = issued_hour_bkk
    weather_db.insert_nwp_forecast_log(rows, metadata=metadata)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_kind": MODEL_KIND,
        "data_dir": str(DATA_DIR),
        "ha_source": ha_source.get_source_status(),
    }


def process_reading(reading: WeatherReading):
    """Persist one reading and run the common prediction/notification path."""
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

    try:
        log_forecast_to_db(prediction)
    except Exception:
        logger.exception("log_forecast_to_db failed; continuing")

    try:
        log_nwp_forecast_to_db()
    except Exception:
        logger.exception("log_nwp_forecast_to_db failed; continuing")

    return {
        "saved": True,
        "csv": str(csv_path),
        "row": row,
        "prediction_ready": True,
        "prediction": prediction,
    }


def ingest_homeassistant_reading(payload: dict):
    """Callback used by ha_source.poll_loop; failures never stop polling."""
    try:
        reading = WeatherReading(**payload)
        result = process_reading(reading)
        ha_source.record_ingest_result(result, reading.timestamp)
        logger.info("Ingested Home Assistant observation at %s (prediction_ready=%s)", reading.timestamp, result.get("prediction_ready"))
    except Exception:
        ha_source.record_ingest_result({"prediction_ready": False, "message": "ingest_error"})
        logger.exception("Home Assistant reading ingestion failed")


@app.post("/reading")
def add_reading(reading: WeatherReading):
    return process_reading(reading)


@app.get("/ha-source/status")
def ha_source_status():
    """Return sanitized Home Assistant source health/configuration."""
    return ha_source.get_source_status()


@app.get("/predict")
def predict_latest(model: str | None = None):
    model_kind = model or MODEL_KIND
    if model_kind not in ["xgb", "rf", "extra_trees"]:
        raise HTTPException(status_code=400, detail='model must be "xgb", "rf" or "extra_trees"')

    try:
        return predict(model_kind=model_kind)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/forecast")
def unified_forecast(model: str | None = None):
    """Return the user-facing nowcast → NWP forecast assembled in one place."""
    model_kind = model or MODEL_KIND
    if model_kind not in ["xgb", "rf", "extra_trees"]:
        raise HTTPException(status_code=400, detail='model must be "xgb", "rf" or "extra_trees"')
    result = forecast_service.get_forecast(model_kind=model_kind)
    if result is None:
        raise HTTPException(status_code=503, detail="forecast is not ready; need continuous sensor history")
    # Enrichment is deliberately best-effort: revision state, rain-stop
    # history and official feeds must never take down the core forecast.
    try:
        result["revision"] = forecast_revision.observe(result)
    except Exception as exc:
        logger.exception("forecast revision comparison failed")
        result["revision"] = {"status": "unavailable", "reason": str(exc)}
    try:
        result["official_warnings"] = official_warnings.get_warnings()
        result.setdefault("source_status", {})["official"] = {
            "status": result["official_warnings"].get("status", "unavailable"),
            "fetched_at_utc": result["official_warnings"].get("fetched_at_utc"),
            "cache_age_seconds": result["official_warnings"].get("cache_age_seconds"),
            "reason": result["official_warnings"].get("reason"),
        }
    except Exception as exc:
        logger.exception("official warning enrichment failed")
        result["official_warnings"] = {
            "status": "unavailable", "warnings": [], "reason": str(exc),
        }
        result.setdefault("source_status", {})["official"] = {
            "status": "unavailable", "reason": str(exc),
        }
    result.setdefault("source_status", {})["rain_stop"] = {
        "status": (result.get("rain_stop") or {}).get("status", "unavailable"),
        "reason": (result.get("rain_stop") or {}).get("reason"),
    }
    try:
        result["source_status"]["ha"] = ha_source.get_source_status()
    except Exception as exc:
        logger.exception("HA source status enrichment failed")
        result["source_status"]["ha"] = {"status": "unavailable", "reason": str(exc)}
    try:
        ha_publisher.publish_forecast_extras(result)
    except Exception:
        logger.exception("HA forecast extras publish failed; continuing")
    return result


@app.get("/official-warnings")
def official_warning_feed(force: bool = False):
    """Return TMD CAP warnings matched to the configured station area."""
    return official_warnings.get_warnings(force=force)


@app.get("/rain-stop")
def rain_stop_forecast():
    """Return the experimental, history-based rain-stop estimate."""
    return rain_stop.get_rain_stop_forecast()
