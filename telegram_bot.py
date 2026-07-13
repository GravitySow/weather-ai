"""Long-polls the Telegram Bot API for incoming chat commands.

Runs in a background thread started from weather_api.py's startup event.
Only responds to messages from TELEGRAM_CHAT_ID — anyone else's commands
are logged and ignored so a leaked bot username can't be used to pull data.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import requests

import nwp_forecast
import telegram_notifier
import weather_db
from predict_weather_ai import predict
from weather_features_lib import comfort_level, heat_index_celsius

logger = logging.getLogger(__name__)

MODEL_KIND = os.getenv("WEATHER_MODEL_KIND") or "rf"
BANGKOK_OFFSET = timedelta(hours=7)
SCHEDULED_HOURS = (6, 21)  # 06:00 morning forecast, 21:00 evening recap

HELP_TEXT = (
    "\U0001F916 <b>คำสั่งที่ใช้ได้</b>\n"
    "/daily - สรุปสภาพอากาศวันนี้\n"
    "/weekly - สรุปสภาพอากาศ 7 วันล่าสุด\n"
    "/forecast - พยากรณ์วันนี้และคืนนี้\n"
    "/help - แสดงข้อความนี้\n\n"
    "ระบบจะส่งพยากรณ์อัตโนมัติทุกเช้า 06:00 "
    "และสรุปประจำวันทุกค่ำ 21:00 (เวลาไทย)"
)


def _bangkok_now():
    return datetime.utcnow() + BANGKOK_OFFSET


def _compute_summary(rows):
    if not rows:
        return None

    temps = np.array([r["temp"] for r in rows], dtype=float)
    hums = np.array([r["humidity"] for r in rows], dtype=float)
    pressures = np.array([r["pressure"] for r in rows], dtype=float)
    rain_flags = np.array([r["rain_flag"] for r in rows], dtype=float)
    heat_indices = heat_index_celsius(temps, hums)
    peak_idx = int(np.argmax(heat_indices))

    return {
        "count": len(rows),
        "temp_min": float(temps.min()),
        "temp_max": float(temps.max()),
        "temp_avg": float(temps.mean()),
        "humidity_min": float(hums.min()),
        "humidity_max": float(hums.max()),
        "humidity_avg": float(hums.mean()),
        "pressure_min": float(pressures.min()),
        "pressure_max": float(pressures.max()),
        "rain_minutes": float((rain_flags > 0).sum()),
        "heat_index_max": float(heat_indices[peak_idx]),
        "heat_index_avg": float(heat_indices.mean()),
        "comfort_level_peak": comfort_level(heat_indices[peak_idx]),
    }


def _format_summary(title, summary):
    if not summary:
        return f"{title}\nยังไม่มีข้อมูลในช่วงนี้"

    rain_pct = (summary["rain_minutes"] / summary["count"] * 100) if summary["count"] else 0
    return (
        f"{title}\n"
        f"\U0001F321️ อุณหภูมิ: {summary['temp_min']:.1f} - {summary['temp_max']:.1f}°C "
        f"(เฉลี่ย {summary['temp_avg']:.1f}°C)\n"
        f"\U0001F525 ดัชนีความร้อนสูงสุด: {summary['heat_index_max']:.1f}°C "
        f"({summary['comfort_level_peak']}) เฉลี่ย {summary['heat_index_avg']:.1f}°C\n"
        f"\U0001F4A7 ความชื้น: {summary['humidity_min']:.0f} - {summary['humidity_max']:.0f}% "
        f"(เฉลี่ย {summary['humidity_avg']:.0f}%)\n"
        f"\U0001F4C9 ความกดอากาศ: {summary['pressure_min']:.1f} - {summary['pressure_max']:.1f} hPa\n"
        f"\U0001F327️ ฝนตก: {summary['rain_minutes']:.0f} นาที ({rain_pct:.0f}% ของช่วงเวลา)\n"
        f"\U0001F4CA จำนวนข้อมูล: {summary['count']} readings"
    )


def _format_forecast(forecast):
    if not forecast or not (forecast.get("today") or forecast.get("tonight")):
        return None

    lines = ["\U0001F324️ <b>พยากรณ์วันนี้และคืนนี้</b> (Open-Meteo)"]

    today = forecast.get("today")
    if today:
        lines.append(
            f"☀️ กลางวัน: {today['temp_min']:.0f}-{today['temp_max']:.0f}°C "
            f"โอกาสฝน {today['rain_prob_max']:.0f}% ({today['condition']})"
        )

    tonight = forecast.get("tonight")
    if tonight:
        lines.append(
            f"\U0001F319 กลางคืน: {tonight['temp_min']:.0f}-{tonight['temp_max']:.0f}°C "
            f"โอกาสฝน {tonight['rain_prob_max']:.0f}% ({tonight['condition']})"
        )

    return "\n".join(lines)


def _format_sensor_nowcast():
    """Current-moment nowcast from our own sensor model (5/10/30-minute rain
    probability) — reliable only at this short range, unlike the NWP outlook."""
    try:
        result = predict(model_kind=MODEL_KIND)
    except Exception:
        logger.exception("Failed to build sensor nowcast")
        return None

    horizon_bits = ", ".join(
        f"{horizon}: {pred['probability'] * 100:.0f}%"
        for horizon, pred in result["predictions"].items()
    )

    lines = [
        "\U0001F4E1 <b>จากเซ็นเซอร์เรา (ตอนนี้)</b>",
        f"\U0001F321️ {result['temp']:.1f}°C | \U0001F4A7 {result['humidity']:.0f}% | "
        f"\U0001F525 HI {result['heat_index']:.1f}°C ({result['comfort_level']})",
        "\U0001F327️ กำลังฝนตก" if result["is_raining_now"] else "☀️ ไม่มีฝนตอนนี้",
        f"โอกาสฝน (nowcast): {horizon_bits}",
    ]
    if result["any_rain_alert"]:
        lines.append(f"⚠️ {result['alert_message']}")

    return "\n".join(lines)


def _today_readings_summary():
    now_bkk = _bangkok_now()
    start_bkk_midnight = now_bkk.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_bkk_midnight - BANGKOK_OFFSET
    end_utc = datetime.utcnow()
    return _compute_summary(weather_db.get_readings(start_utc, end_utc))


def handle_command(text):
    command = text.strip().split("@")[0].split()[0].lower()

    if command in ("/help", "/start"):
        return HELP_TEXT

    if command == "/daily":
        try:
            summary = _today_readings_summary()
        except Exception:
            logger.exception("Failed to build /daily summary")
            return "⚠️ ดึงข้อมูลสรุปวันนี้ไม่สำเร็จ"
        return _format_summary("\U0001F4C5 <b>สรุปวันนี้</b>", summary)

    if command == "/forecast":
        parts = [
            text
            for text in (_format_sensor_nowcast(), _format_forecast(nwp_forecast.get_today_tonight_forecast()))
            if text
        ]
        return "\n\n".join(parts) if parts else "⚠️ ดึงพยากรณ์ไม่สำเร็จ"

    if command == "/weekly":
        end_utc = datetime.utcnow()
        start_utc = end_utc - timedelta(days=7)
        try:
            summary = _compute_summary(weather_db.get_readings(start_utc, end_utc))
        except Exception:
            logger.exception("Failed to build /weekly summary")
            return "⚠️ ดึงข้อมูลสรุป 7 วันไม่สำเร็จ"
        return _format_summary("\U0001F4C5 <b>สรุป 7 วันล่าสุด</b>", summary)

    return None


def _delete_webhook(token):
    """getUpdates 409s if a webhook is registered for this bot — clear it
    unconditionally before polling so stale webhook config can't block us."""
    try:
        requests.post(
            f"{telegram_notifier.TELEGRAM_API_BASE}/bot{token}/deleteWebhook",
            timeout=10,
        )
    except requests.RequestException:
        logger.exception("Failed to clear Telegram webhook before polling")


def poll_loop():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    allowed_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set; Telegram command polling disabled")
        return
    if not allowed_chat_id:
        logger.warning("TELEGRAM_CHAT_ID not set; Telegram command polling disabled")
        return

    _delete_webhook(token)

    url = f"{telegram_notifier.TELEGRAM_API_BASE}/bot{token}/getUpdates"
    offset = None

    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset

            response = requests.get(url, params=params, timeout=35)
            response.raise_for_status()
            updates = response.json().get("result", [])

            for update in updates:
                offset = update["update_id"] + 1
                message = update.get("message") or {}
                text = message.get("text")
                chat_id = str(message.get("chat", {}).get("id", ""))

                if not text or not text.startswith("/"):
                    continue
                if chat_id != str(allowed_chat_id):
                    logger.warning("Ignoring Telegram command from unauthorized chat_id=%s", chat_id)
                    continue

                reply = handle_command(text)
                if reply:
                    telegram_notifier.send_message(reply)

        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 409:
                logger.warning(
                    "Telegram getUpdates 409: another poller (old process instance, "
                    "or a webhook) is still active; backing off 15s to let it clear"
                )
                time.sleep(15)
            else:
                logger.exception("Telegram getUpdates failed; retrying in 5s")
                time.sleep(5)
        except requests.RequestException:
            logger.exception("Telegram getUpdates failed; retrying in 5s")
            time.sleep(5)
        except Exception:
            logger.exception("Unexpected error in Telegram poll loop; retrying in 5s")
            time.sleep(5)


def _next_scheduled_run():
    """Return (hour, next_run_bangkok_datetime) for the soonest of SCHEDULED_HOURS."""
    now_bkk = _bangkok_now()
    today = now_bkk.date()

    candidates = []
    for hour in SCHEDULED_HOURS:
        run_at = datetime.combine(today, datetime.min.time()).replace(hour=hour)
        if run_at <= now_bkk:
            run_at += timedelta(days=1)
        candidates.append((hour, run_at))

    return min(candidates, key=lambda pair: pair[1])


def _send_morning_forecast():
    parts = [
        text
        for text in (_format_sensor_nowcast(), _format_forecast(nwp_forecast.get_today_tonight_forecast()))
        if text
    ]
    if parts:
        telegram_notifier.send_message("\n\n".join(parts))
    else:
        logger.warning("No forecast/nowcast available for the scheduled 06:00 push")


def _send_evening_recap():
    try:
        summary = _today_readings_summary()
    except Exception:
        logger.exception("Failed to build the scheduled 21:00 recap")
        return
    telegram_notifier.send_message(_format_summary("\U0001F4C5 <b>สรุปประจำวัน</b>", summary))


def schedule_loop():
    """Sleeps until each of SCHEDULED_HOURS (Bangkok time) and pushes a message."""
    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        logger.warning("Telegram not configured; scheduled summaries disabled")
        return

    while True:
        hour, next_run_bkk = _next_scheduled_run()
        sleep_seconds = (next_run_bkk - _bangkok_now()).total_seconds()
        time.sleep(max(sleep_seconds, 1))

        try:
            if hour == 6:
                _send_morning_forecast()
            else:
                _send_evening_recap()
        except Exception:
            logger.exception("Scheduled Telegram push failed")
