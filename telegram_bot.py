"""Long-polls the Telegram Bot API for incoming chat commands.

Runs in a background thread started from weather_api.py's startup event.
Only responds to messages from TELEGRAM_CHAT_ID — anyone else's commands
are logged and ignored so a leaked bot username can't be used to pull data.
"""

import html
import logging
import os
import time
from datetime import datetime, timedelta

import numpy as np
import requests

import forecast_service
import forecast_revision
import nwp_forecast
import official_warnings
import telegram_notifier
import weather_db
from predict_weather_ai import predict
from weather_features_lib import comfort_level, heat_index_celsius

logger = logging.getLogger(__name__)


def _esc(value):
    """Escapes a value for interpolation into an HTML parse_mode message.
    Every dynamic string field currently traces back to a fixed vocabulary
    (comfort_level's buckets, our own weather-code dict) so nothing can
    actually contain '<'/'&' today — but a stray one would 400 and drop the
    WHOLE message (Telegram doesn't partially render), so anything that
    isn't a number we control the formatting of gets escaped defensively
    rather than relying on every future field staying safe by luck."""
    return html.escape(str(value), quote=False)

MODEL_KIND = os.getenv("WEATHER_MODEL_KIND") or "rf"
BANGKOK_OFFSET = timedelta(hours=7)
SCHEDULED_HOURS = (6, 21)  # 06:00 morning forecast, 21:00 evening recap

HELP_TEXT = (
    "\U0001F916 <b>คำสั่งที่ใช้ได้</b>\n"
    "/daily - สรุปสภาพอากาศวันนี้\n"
    "/weekly - สรุปสภาพอากาศ 7 วันล่าสุด (จากเซ็นเซอร์)\n"
    "/forecast - พยากรณ์ตอนนี้ถึง 6 ชั่วโมงข้างหน้า + วันนี้/คืนนี้\n"
    "/warnings - ประกาศเตือนภัยทางการตามพื้นที่\n"
    "/week - พยากรณ์ล่วงหน้า 7 วัน (Open-Meteo)\n"
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
        f"({_esc(summary['comfort_level_peak'])}) เฉลี่ย {summary['heat_index_avg']:.1f}°C\n"
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
            f"โอกาสฝน {today['rain_prob_max']:.0f}% ({_esc(today['condition'])})"
        )

    tonight = forecast.get("tonight")
    if tonight:
        lines.append(
            f"\U0001F319 กลางคืน: {tonight['temp_min']:.0f}-{tonight['temp_max']:.0f}°C "
            f"โอกาสฝน {tonight['rain_prob_max']:.0f}% ({_esc(tonight['condition'])})"
        )

    return "\n".join(lines)


# Unicode block sparkline levels, empty (no data) through full (100%).
_SPARKLINE_LEVELS = " ▁▂▃▄▅▆▇█"

# rain_prob% at/above which an hour counts as "likely raining" for the
# rain-window summary in the 06:00 push / /forecast — see plan.md Phase 4:
# "timed windows are the biggest UX jump over today's single max-probability
# number."
_RAIN_WINDOW_THRESHOLD = 50


def _sparkline(values, max_value=100):
    chars = []
    for value in values:
        if value is None:
            chars.append(" ")
            continue
        level = int(round((value / max_value) * (len(_SPARKLINE_LEVELS) - 1)))
        level = max(0, min(level, len(_SPARKLINE_LEVELS) - 1))
        chars.append(_SPARKLINE_LEVELS[level])
    return "".join(chars)


def _format_rain_sparkline(hourly):
    """3-hourly rain-probability sparkline across the next ~24h."""
    if not hourly:
        return None
    sampled = hourly[::3][:8]
    if len(sampled) < 2:
        return None
    bars = _sparkline([h.get("rain_prob") for h in sampled])
    start_label = sampled[0]["time"][11:16]
    end_label = sampled[-1]["time"][11:16]
    return f"\U0001F4C8 โอกาสฝนทุก 3ชม. ({start_label}-{end_label}): {bars}"


def _find_rain_windows(hourly, threshold=_RAIN_WINDOW_THRESHOLD):
    """[(start_time_str, end_time_str), ...] — contiguous runs of hourly
    entries at/above `threshold`, merging adjacent wet hours into one
    window instead of reporting each hour separately."""
    windows = []
    window_start = None
    prev_time = None
    for entry in hourly:
        prob = entry.get("rain_prob")
        is_wet = prob is not None and prob >= threshold
        if is_wet and window_start is None:
            window_start = entry["time"]
        if not is_wet and window_start is not None:
            windows.append((window_start, prev_time))
            window_start = None
        prev_time = entry["time"]
    if window_start is not None:
        windows.append((window_start, prev_time))
    return windows


def _format_rain_windows(hourly):
    """"ฝนน่าจะตกช่วง 15:00-18:00" style summary — each hourly entry covers
    the hour starting at its timestamp, so a window's displayed end is the
    last wet hour's start + 1h."""
    windows = _find_rain_windows(hourly)
    if not windows:
        return None
    labels = []
    for start, end in windows:
        start_label = start[11:16]
        end_label = (datetime.fromisoformat(end) + timedelta(hours=1)).strftime("%H:%M")
        labels.append(f"{start_label}-{end_label}")
    return f"\U0001F327️ ช่วงที่ฝนน่าจะตก (≥{_RAIN_WINDOW_THRESHOLD}%): " + ", ".join(labels)


def _format_daily_highlights(daily):
    """Wind/UV/sunrise-sunset for today — see plan.md Phase 4."""
    if not daily:
        return None
    today = daily[0]
    sunrise = today["sunrise"][11:16] if today.get("sunrise") else "-"
    sunset = today["sunset"][11:16] if today.get("sunset") else "-"
    wind = f"{today['wind_kmh_max']:.0f}" if today.get("wind_kmh_max") is not None else "-"
    uv = f"{today['uv_index_max']:.1f}" if today.get("uv_index_max") is not None else "-"
    return (
        f"\U0001F4A8 ลมสูงสุด {wind} กม./ชม. | ☀️ UV สูงสุด {uv}\n"
        f"\U0001F305 พระอาทิตย์ขึ้น {sunrise} | \U0001F307 ตก {sunset}"
    )


def _format_nowcast_card(forecast):
    """"Now" + next-2h nowcast + radar arrival ETA, built from
    forecast_service.get_forecast()'s blended object — see plan.md Phase 3/4."""
    now = forecast.get("now")
    if now:
        lines = [
            "\U0001F4E1 <b>ตอนนี้</b>",
            f"\U0001F321️ {now['temp']:.1f}°C | \U0001F4A7 {now['humidity']:.0f}% | "
            f"\U0001F525 HI {now['heat_index']:.1f}°C ({_esc(now['comfort_level'])})",
            "\U0001F327️ กำลังฝนตก" if now["is_raining_now"] else "☀️ ไม่มีฝนตอนนี้",
            f"\U0001F4CA ความกดอากาศ {now['pressure']:.1f} hPa แนวโน้ม{now['trend_arrow']} "
            f"({now['pressure_trend']:+.1f} hPa/ชม.)",
        ]
    else:
        local = (forecast.get("source_status") or {}).get("local") or {}
        lines = [
            "\U0001F4E1 <b>พยากรณ์บางส่วน</b>",
            "⚠️ ข้อมูลเซนเซอร์/โมเดลเฉพาะจุดยังไม่พร้อม "
            f"({_esc(local.get('reason') or 'ยังไม่มีข้อมูลต่อเนื่อง')})",
        ]

    nowcast_bits = ", ".join(
        f"{p['lead_min']}min: {p['rain_prob'] * 100:.0f}%" for p in forecast["nowcast"]
    )
    if nowcast_bits:
        lines.append(f"โอกาสฝน: {nowcast_bits}")

    arrival = forecast.get("arrival")
    if arrival:
        lines.append(
            f"\U0001F4E1 เรดาร์: ฝนกำลังเข้า คาดถึงใน ~{arrival['eta_minutes']}นาที "
            f"(ระดับ {arrival['expected_intensity']}/4, ความเชื่อมั่น {arrival['confidence'] * 100:.0f}%) "
            f"(สัญญาณทดลอง ยังไม่ผ่านการพิสูจน์ความแม่นยำ, RainViewer)"
        )

    return "\n".join(lines)


def _format_6h_timeline(hourly):
    points = hourly[:6]
    if not points:
        return None
    lines = ["\U0001F550 <b>6 ชั่วโมงข้างหน้า</b>"]
    for point in points:
        time_label = point["time"][11:16]
        temp = f"{point['temp']:.0f}°C" if point.get("temp") is not None else "-"
        interval = point.get("temp_interval") or {}
        if interval.get("lower") is not None and interval.get("upper") is not None:
            temp += f" ({interval['lower']:.0f}-{interval['upper']:.0f})"
        rain = f"{point['rain_prob']:.0f}%" if point.get("rain_prob") is not None else "-"
        condition = _esc(point.get("condition") or "")
        lines.append(f"{time_label}  {temp}  ฝน {rain}  {condition}")
    return "\n".join(lines)


def _format_revision(revision):
    if not revision or revision.get("status") in ("unchanged", "baseline", "not_comparable"):
        return None
    if revision.get("status") != "changed":
        return None
    labels = {
        "rain_probability": "โอกาสฝนเปลี่ยน",
        "rain_window_shift": "ช่วงฝนเลื่อน",
        "temperature": "อุณหภูมิเปลี่ยน",
    }
    reasons = ", ".join(labels.get(reason, reason) for reason in revision.get("reasons", []))
    details = []
    if revision.get("rain_window_shift_minutes") is not None:
        details.append(f"ช่วงฝน {revision['rain_window_shift_minutes']:+.0f} นาที")
    if revision.get("max_rain_probability_delta_points"):
        details.append(f"โอกาสฝนต่าง {revision['max_rain_probability_delta_points']:.0f} จุด")
    if revision.get("max_temperature_delta_c"):
        details.append(f"อุณหภูมิต่าง {revision['max_temperature_delta_c']:.1f}°C")
    return "🔄 <b>พยากรณ์มีการปรับปรุง</b> — " + (reasons or "มีการเปลี่ยนแปลง") + \
        (f" ({'; '.join(details)})" if details else "")


def _format_rain_stop(rain_stop):
    if not rain_stop or rain_stop.get("status") != "experimental":
        return None
    remaining = rain_stop.get("remaining_minutes") or {}
    estimate = rain_stop.get("estimated_stop_at", "")
    try:
        estimate_label = (datetime.fromisoformat(estimate.replace("Z", "+00:00")) + BANGKOK_OFFSET).strftime("%H:%M")
    except (TypeError, ValueError):
        estimate_label = "-"
    return (
        f"🌦️ ฝนอาจหยุดราว {estimate_label} "
        f"(เหลือประมาณ {remaining.get('lower', 0):.0f}-{remaining.get('upper', 0):.0f} นาที)\n"
        "<i>สถิติจากเหตุการณ์ฝนในพื้นที่ ยังเป็นฟีเจอร์ทดลอง</i>"
    )


def _format_official_warnings(feed):
    if not feed:
        return None
    warnings = feed.get("warnings") or []
    status = feed.get("status")
    if not warnings:
        if status in ("unavailable", "stale"):
            return "⚠️ ประกาศเตือนภัยทางการ: ตรวจสอบแหล่งข้อมูลไม่ได้ในขณะนี้"
        return None
    lines = ["🚨 <b>ประกาศเตือนภัยทางการ (TMD)</b>"]
    if status == "stale":
        lines.append("⚠️ ข้อมูลจาก cache อาจไม่ใช่ฉบับล่าสุด")
    for warning in warnings[:3]:
        headline = _esc(warning.get("headline") or warning.get("event") or "ประกาศเตือนภัย")
        expires = warning.get("expires") or "ไม่ระบุเวลาหมดอายุ"
        try:
            expires = datetime.fromisoformat(expires.replace("Z", "+00:00")).astimezone().strftime("%d/%m %H:%M")
        except (TypeError, ValueError):
            pass
        lines.append(f"• {headline} ({_esc(warning.get('severity') or 'ไม่ระบุ')}) ถึง {expires}")
        if warning.get("source_url"):
            lines.append(f"  {_esc(warning['source_url'])}")
    return "\n".join(lines)


def _format_weekly_forecast(daily):
    if not daily:
        return None
    lines = ["\U0001F4C5 <b>พยากรณ์ 7 วัน</b> (Open-Meteo)"]
    for day in daily:
        tmin = f"{day['temp_min']:.0f}" if day.get("temp_min") is not None else "-"
        tmax = f"{day['temp_max']:.0f}" if day.get("temp_max") is not None else "-"
        rain = f"{day['rain_prob_max']:.0f}%" if day.get("rain_prob_max") is not None else "-"
        condition = _esc(day.get("condition") or "")
        lines.append(f"{day['date']}  {tmin}-{tmax}°C  ฝน {rain}  {condition}")
    return "\n".join(lines)


def _pressure_weekly_trend(current_pressure):
    """Compares the current (~06:00) pressure against the mean of the same
    06:00-07:00 local window over the previous mornings. Only meaningful right
    at the 06:00 push — pressure has a strong diurnal cycle, so comparing an
    afternoon reading against a 06:00 baseline would just measure time-of-day,
    not a real multi-day drift."""
    now_utc = datetime.utcnow()
    try:
        rows = weather_db.get_readings(now_utc - timedelta(days=8), now_utc)
    except Exception:
        logger.exception("Failed to fetch history for weekly pressure trend")
        return None

    today_bkk = _bangkok_now().date()
    daily_morning = {}
    for row in rows:
        local = row["reading_time"] + BANGKOK_OFFSET
        if local.hour != 6 or local.date() == today_bkk:
            continue
        daily_morning.setdefault(local.date(), []).append(float(row["pressure"]))

    if len(daily_morning) < 3:
        return None

    baseline_days = sorted(daily_morning)[-7:]
    daily_means = [float(np.mean(daily_morning[d])) for d in baseline_days]
    baseline = float(np.mean(daily_means))
    anomaly = current_pressure - baseline

    if anomaly < -0.5:
        direction = f"ต่ำกว่าค่าเฉลี่ย {abs(anomaly):.1f} hPa (มีแนวโน้มลดลงช่วงนี้)"
    elif anomaly > 0.5:
        direction = f"สูงกว่าค่าเฉลี่ย {anomaly:.1f} hPa"
    else:
        direction = "ใกล้เคียงค่าเฉลี่ยช่วงนี้"

    return (
        f"\U0001F4C9 เทียบเช้า {len(baseline_days)} วันที่ผ่านมา "
        f"(baseline {baseline:.1f} hPa): {direction}"
    )


def _format_sensor_nowcast(include_weekly_pressure_trend=False):
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
        f"\U0001F525 HI {result['heat_index']:.1f}°C ({_esc(result['comfort_level'])})",
        "\U0001F327️ กำลังฝนตก" if result["is_raining_now"] else "☀️ ไม่มีฝนตอนนี้",
        f"โอกาสฝน (nowcast): {horizon_bits}",
        f"\U0001F4CA ความกดอากาศ {result['pressure']:.1f} hPa "
        f"แนวโน้ม{result['trend_arrow']} ({result['pressure_trend']:+.1f} hPa/ชม.)",
    ]
    if include_weekly_pressure_trend:
        weekly_line = _pressure_weekly_trend(result["pressure"])
        if weekly_line:
            lines.append(weekly_line)
    if result["any_rain_alert"]:
        lines.append(f"⚠️ {result['alert_message']}")

    radar = result.get("radar") or {}
    if radar.get("available"):
        trend = "เพิ่มขึ้น ↑" if radar.get("trend_rising") else "คงที่/ลดลง"
        lines.append(
            f"\U0001F4E1 เรดาร์: ฝนใกล้เคียง (25กม.) ระดับ {radar['close_max_intensity']}/4 "
            f"แนวโน้ม{trend} (สัญญาณทดลอง, RainViewer)"
        )

    cloud = result.get("cloud") or {}
    if cloud.get("available"):
        trend = "เพิ่มขึ้น ↑" if cloud.get("trend_rising") else "คงที่/ลดลง"
        lines.append(
            f"☁️ เมฆ: {cloud['cloud_cover_now']}% (เมฆต่ำ {cloud['cloud_cover_low_now']}%) "
            f"แนวโน้ม 1ชม.{trend} (สัญญาณทดลอง)"
        )

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
        forecast = forecast_service.get_forecast()
        if forecast:
            try:
                forecast["revision"] = forecast_revision.observe(forecast)
            except Exception:
                logger.exception("Failed to record forecast revision from Telegram")
            try:
                forecast["official_warnings"] = official_warnings.get_warnings()
            except Exception:
                logger.exception("Failed to fetch official warnings from Telegram")
        hourly = forecast["hourly"] if forecast else []
        parts = [
            text
            for text in (
                _format_nowcast_card(forecast) if forecast else None,
                _format_revision(forecast.get("revision")) if forecast else None,
                _format_6h_timeline(hourly),
                _format_rain_windows(hourly),
                _format_rain_sparkline(hourly),
                _format_rain_stop(forecast.get("rain_stop")) if forecast else None,
                _format_official_warnings(forecast.get("official_warnings")) if forecast else None,
                _format_forecast(nwp_forecast.get_today_tonight_forecast()),
            )
            if text
        ]
        return "\n\n".join(parts) if parts else "⚠️ ดึงพยากรณ์ไม่สำเร็จ"

    if command == "/warnings":
        try:
            return _format_official_warnings(official_warnings.get_warnings(force=True)) or \
                "ยังไม่มีประกาศเตือนภัยทางการที่ตรงกับพื้นที่"
        except Exception:
            logger.exception("Failed to build official warning message")
            return "⚠️ ตรวจสอบประกาศเตือนภัยทางการไม่สำเร็จ"

    if command == "/week":
        daily = nwp_forecast.get_daily_forecast(days=7)
        return _format_weekly_forecast(daily) or "⚠️ ดึงพยากรณ์ 7 วันไม่สำเร็จ"

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
    """Returns True if the message was actually sent."""
    hourly = nwp_forecast.get_hourly_timeline(hours=24)
    daily = nwp_forecast.get_daily_forecast(days=1)
    parts = [
        text
        for text in (
            _format_sensor_nowcast(include_weekly_pressure_trend=True),
            _format_forecast(nwp_forecast.get_today_tonight_forecast()),
            # Rain windows with actual times are the biggest UX jump over a
            # single max-probability number — see plan.md "Phase 4".
            _format_rain_windows(hourly),
            _format_rain_sparkline(hourly),
            _format_daily_highlights(daily),
        )
        if text
    ]
    if not parts:
        logger.warning("No forecast/nowcast available for the scheduled 06:00 push")
        return False
    # silent=True: a routine daily digest, not something actionable enough
    # to warrant a notification sound — see plan.md "Phase 6b" severity tiers.
    return telegram_notifier.send_message("\n\n".join(parts), silent=True)


def _send_evening_recap():
    """Returns True if the message was actually sent."""
    try:
        summary = _today_readings_summary()
    except Exception:
        logger.exception("Failed to build the scheduled 21:00 recap")
        return False
    return telegram_notifier.send_message(
        _format_summary("\U0001F4C5 <b>สรุปประจำวัน</b>", summary), silent=True
    )


def _send_scheduled(hour):
    """Sends the push for this scheduled hour and records it in
    scheduled_message_log on success, so a later restart can tell "already
    sent today" apart from "missed it" (see _catch_up_missed_schedule)."""
    sent = _send_morning_forecast() if hour == 6 else _send_evening_recap()
    if sent:
        try:
            weather_db.mark_scheduled_sent(hour, _bangkok_now().date())
        except Exception:
            logger.exception("Failed to record scheduled-message send for hour=%s", hour)
    return sent


def _catch_up_missed_schedule():
    """If the process (re)started after a scheduled hour already passed
    today without that day's message having gone out — e.g. an add-on
    rebuild at 06:01 — send it now instead of silently waiting for
    tomorrow. Idempotent: a restart that finds today's slot already
    recorded as sent does nothing. Runs once before schedule_loop's
    forward-looking sleep loop starts."""
    now_bkk = _bangkok_now()
    today = now_bkk.date()

    for hour in SCHEDULED_HOURS:
        if now_bkk.hour < hour:
            continue  # that slot hasn't happened yet today

        try:
            last_sent = weather_db.get_last_sent_date(hour)
        except Exception:
            logger.exception(
                "Could not check scheduled-message history; skipping catch-up for hour=%02d:00",
                hour,
            )
            continue

        if last_sent == today:
            continue

        logger.warning(
            "Scheduled %02d:00 push for %s appears to have been missed "
            "(likely a restart); sending catch-up now",
            hour, today,
        )
        try:
            _send_scheduled(hour)
        except Exception:
            logger.exception("Catch-up send failed for hour=%02d:00", hour)


def schedule_loop():
    """Sleeps until each of SCHEDULED_HOURS (Bangkok time) and pushes a message."""
    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        logger.warning("Telegram not configured; scheduled summaries disabled")
        return

    _catch_up_missed_schedule()

    while True:
        hour, next_run_bkk = _next_scheduled_run()
        sleep_seconds = (next_run_bkk - _bangkok_now()).total_seconds()
        time.sleep(max(sleep_seconds, 1))

        try:
            _send_scheduled(hour)
        except Exception:
            logger.exception("Scheduled Telegram push failed")
