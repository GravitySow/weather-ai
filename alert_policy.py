"""Pure cooldown policy for rain notifications."""

from datetime import datetime, timezone


def can_notify(previous, kind, now=None, cooldown_seconds=1800):
    """Allow a notification unless the same kind was sent too recently."""
    if not previous:
        return True
    if previous.get("last_notification_kind") != kind:
        return True
    sent = previous.get("last_notification_at")
    if not sent:
        return True
    if now is None:
        now = datetime.now(timezone.utc)
    if sent.tzinfo is None:
        sent = sent.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - sent).total_seconds() >= float(cooldown_seconds)

