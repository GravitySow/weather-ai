"""Minimal Telegram Bot API client for rain-alert push notifications.

See plan.md "Phase 6a"/"Phase 6b" for why each piece here exists — this used
to be a ~20-line client with no retry, no length limit, no flood control,
and no delivery record; every caller also ignored its return value.
"""

import logging
import os
import threading
import time

import requests

import weather_db

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

# A once-a-day scheduled push (06:00/21:00) or a rain alert is worth a couple
# retries on a transient network blip rather than being silently dropped —
# previously a single failed request just logged and gave up, and every
# caller ignored the False return anyway.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 2

# Telegram's 429 response includes parameters.retry_after (seconds). This
# call runs synchronously inside the /reading request path (via
# notify_rain_state_change), so cap how long a single send is allowed to
# block rather than sleeping for whatever Telegram asks under heavy flood
# control.
_MAX_RETRY_AFTER_SECONDS = 30

# Telegram caps a bot to roughly 20 messages/minute to the same chat. This
# client-side budget throttles proactively so we approach that cap gracefully
# (a short wait) instead of firing and then handling a 429 reactively. Set
# below the real ~20 for margin — matters once Phase 2's radar-advection
# alerts (still shadow mode) or future signals start pushing more often.
_RATE_LIMIT_MAX_PER_WINDOW = 15
_RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_lock = threading.Lock()
_recent_send_times = []

# Telegram's real limit is 4096 UTF-16 code units; using Python's codepoint
# count as a conservative proxy (emoji-heavy text could differ by a handful
# of units in the worst case, hence the margin below 4096 rather than
# matching it exactly).
_MAX_MESSAGE_CHARS = 4000


def _retry_after_seconds(response, attempt):
    try:
        body = response.json()
        retry_after = body.get("parameters", {}).get("retry_after")
        if retry_after:
            return min(float(retry_after), _MAX_RETRY_AFTER_SECONDS)
    except ValueError:
        pass
    return _BACKOFF_BASE_SECONDS * attempt


def _throttle_for_rate_limit():
    """Blocks briefly if sending now would push us over
    _RATE_LIMIT_MAX_PER_WINDOW sends in the trailing window."""
    wait_seconds = 0.0
    with _rate_limit_lock:
        now = time.monotonic()
        while _recent_send_times and now - _recent_send_times[0] >= _RATE_LIMIT_WINDOW_SECONDS:
            _recent_send_times.pop(0)
        if len(_recent_send_times) >= _RATE_LIMIT_MAX_PER_WINDOW:
            wait_seconds = _RATE_LIMIT_WINDOW_SECONDS - (now - _recent_send_times[0])

    if wait_seconds > 0:
        logger.info(
            "Client-side rate limit: waiting %.1fs before sending (staying under %d/min)",
            wait_seconds, _RATE_LIMIT_MAX_PER_WINDOW,
        )
        time.sleep(wait_seconds)

    with _rate_limit_lock:
        _recent_send_times.append(time.monotonic())


def _split_message(text, max_chars=_MAX_MESSAGE_CHARS):
    """Splits text into chunks under max_chars, preferring paragraph
    (\\n\\n) boundaries so a long forecast message doesn't get an ugly
    mid-sentence cut, then falling back to line boundaries, then a hard
    cut as a last resort for a single oversized line."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        # A single paragraph is itself too long — fall back to splitting on
        # lines, then hard-cut anything still too long (e.g. one giant line).
        for line in paragraph.split("\n"):
            candidate_line = f"{current}\n{line}" if current else line
            if len(candidate_line) <= max_chars:
                current = candidate_line
                continue
            if current:
                chunks.append(current)
                current = ""
            while len(line) > max_chars:
                chunks.append(line[:max_chars])
                line = line[max_chars:]
            current = line

    if current:
        chunks.append(current)

    return chunks


def _log_delivery(success, attempts, char_length):
    try:
        weather_db.insert_message_log(success=success, attempts=attempts, char_length=char_length)
    except Exception:
        logger.exception("Failed to record message_log entry")


def _send_chunk(token, chat_id, text, silent):
    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        _throttle_for_rate_limit()
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_notification": silent,
                },
                timeout=10,
            )
            if response.status_code == 429:
                wait_seconds = _retry_after_seconds(response, attempt)
                logger.warning(
                    "Telegram rate-limited us (attempt %d/%d); waiting %.1fs",
                    attempt, _MAX_ATTEMPTS, wait_seconds,
                )
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            _log_delivery(True, attempt, len(text))
            return True
        except requests.RequestException:
            logger.exception(
                "Failed to send Telegram message (attempt %d/%d)", attempt, _MAX_ATTEMPTS
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_BACKOFF_BASE_SECONDS * attempt)

    logger.error("Giving up on Telegram message after %d attempts: %s", _MAX_ATTEMPTS, text)
    _log_delivery(False, _MAX_ATTEMPTS, len(text))
    return False


def send_message(text, silent=False):
    """Sends `text`, splitting into multiple messages if it exceeds
    Telegram's length limit. `silent=True` delivers without a
    notification sound/vibration (use for routine digests; leave False for
    anything actionable like a rain alert). Returns True only if every
    chunk was delivered."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; skipping message: %s", text)
        return False

    chunks = _split_message(text)
    # A list comprehension (not a short-circuiting `all(...)` generator) so
    # every chunk is attempted even if an earlier one fails — losing chunk 1
    # to a transient blip shouldn't also silently drop chunks 2+.
    results = [_send_chunk(token, chat_id, chunk, silent) for chunk in chunks]
    return all(results)
