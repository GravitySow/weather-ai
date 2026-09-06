"""Official Thai Meteorological Department CAP warning reader.

TMD publishes a CAP/RSS index and linked CAP XML documents.  This client keeps
the source and validity times visible, matches polygons when present, and
never turns a failed fetch into a claim that no warning exists.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

TMD_CAP_INDEX_URL = os.getenv("TMD_CAP_INDEX_URL", "https://www.tmd.go.th/api/xml/CAP")
TMD_CAP_TIMEOUT_SECONDS = 8
TMD_CACHE_TTL_SECONDS = 10 * 60
TMD_MAX_CAP_ITEMS = 8

_cache = {"result": None, "fetched_at": 0.0}


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _children(element, name):
    return [child for child in list(element) if _tag(child) == name]


def _text(element, name, default=None):
    child = next(iter(_children(element, name)), None)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _parse_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _point_in_polygon(lat, lon, polygon):
    """Ray-casting test for CAP's ``lat,lon`` polygon coordinates."""
    if len(polygon) < 3:
        return False
    inside = False
    previous_lat, previous_lon = polygon[-1]
    for current_lat, current_lon in polygon:
        crosses = ((current_lon > lon) != (previous_lon > lon))
        if crosses:
            at_lat = ((previous_lat - current_lat) * (lon - current_lon) /
                      (previous_lon - current_lon) + current_lat)
            if lat < at_lat:
                inside = not inside
        previous_lat, previous_lon = current_lat, current_lon
    return inside


def _parse_polygon(value):
    points = []
    for pair in re.split(r"\s+", value or ""):
        try:
            lat, lon = pair.split(",", 1)
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        if math.isfinite(lat) and math.isfinite(lon):
            points.append((lat, lon))
    return points


def _area_matches(area_desc, polygons, latitude, longitude, hints):
    for polygon in polygons:
        if _point_in_polygon(latitude, longitude, polygon):
            return True, "polygon"
    description = (area_desc or "").lower()
    if "ประเทศไทย" in description or "ทั่วประเทศ" in description:
        return True, "nationwide_area"
    if any(hint.lower() in description for hint in hints if hint):
        return True, "configured_area_hint"
    return False, "outside_area"


def parse_cap_document(xml_bytes, source_url, latitude, longitude, now=None, area_hints=None):
    """Parse one CAP document into one warning mapping or ``None``.

    A document can contain multiple ``info`` blocks/areas; the returned item
    is included if any area covers the station.  The function is pure and is
    intentionally easy to exercise with a saved CAP fixture.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    hints = area_hints or [item.strip() for item in os.getenv(
        "TMD_AREA_HINTS", "กรุงเทพมหานคร,กรุงเทพ,นนทบุรี,ปทุมธานี,สมุทรปราการ"
    ).split(",")]
    try:
        root = ET.fromstring(xml_bytes)
    except (ET.ParseError, TypeError, ValueError):
        return None

    identifier = _text(root, "identifier")
    status = (_text(root, "status", "Unknown") or "Unknown").lower()
    msg_type = _text(root, "msgType", "Alert") or "Alert"
    sent = _parse_datetime(_text(root, "sent"))
    matched_infos = []
    for info in _children(root, "info"):
        areas = _children(info, "area")
        if not areas:
            matched_infos.append((info, [], "no_area"))
            continue
        for area in areas:
            area_desc = _text(area, "areaDesc", "") or ""
            polygons = [_parse_polygon(child.text or "") for child in _children(area, "polygon")]
            polygons = [p for p in polygons if p]
            if not polygons:
                first = _parse_polygon(_text(area, "polygon", ""))
                if first:
                    polygons = [first]
            matched, match_method = _area_matches(area_desc, polygons, latitude, longitude, hints)
            if matched:
                matched_infos.append((info, areas, match_method))
                break
    if not matched_infos:
        return None

    info, areas, match_method = matched_infos[0]
    effective = _parse_datetime(_text(info, "effective")) or sent
    expires = _parse_datetime(_text(info, "expires"))
    cancelled = msg_type.lower() == "cancel" or status == "cancelled"
    active = not cancelled and (effective is None or now >= effective) and (expires is None or now < expires)
    if cancelled:
        lifecycle = "cancelled"
    elif expires is not None and now >= expires:
        lifecycle = "expired"
    elif effective is not None and now < effective:
        lifecycle = "scheduled"
    else:
        lifecycle = "active"
    area_names = [_text(area, "areaDesc", "") for area in areas if _text(area, "areaDesc", "")]
    return {
        "id": identifier or source_url,
        "status": lifecycle,
        "msg_type": msg_type,
        "event": _text(info, "event"),
        "severity": _text(info, "severity"),
        "urgency": _text(info, "urgency"),
        "certainty": _text(info, "certainty"),
        "sender": _text(info, "senderName") or _text(root, "sender"),
        "headline": _text(info, "headline"),
        "description": (_text(info, "description") or "")[:4000],
        "instruction": (_text(info, "instruction") or "")[:2000],
        "sent": sent.isoformat() if sent else None,
        "effective": effective.isoformat() if effective else None,
        "expires": expires.isoformat() if expires else None,
        "areas": area_names,
        "area_match": match_method,
        "source_url": source_url,
        "web": _text(info, "web"),
        "active": active,
    }


def _rss_items(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
    except (ET.ParseError, TypeError, ValueError):
        return []
    result = []
    for item in root.iter():
        if _tag(item) != "item":
            continue
        result.append({
            "title": _text(item, "title", ""),
            "url": _text(item, "link"),
            "id": _text(item, "guid"),
        })
    return result


def _fetch_warnings():
    fetched_at = datetime.now(timezone.utc).isoformat()
    latitude = float(os.getenv("WEATHER_LAT", "13.8692387"))
    longitude = float(os.getenv("WEATHER_LON", "100.5180519"))
    response = requests.get(TMD_CAP_INDEX_URL, timeout=TMD_CAP_TIMEOUT_SECONDS)
    response.raise_for_status()
    warnings = []
    recent_updates = []
    seen = set()
    for item in _rss_items(response.content)[:TMD_MAX_CAP_ITEMS]:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        try:
            cap = requests.get(url, timeout=TMD_CAP_TIMEOUT_SECONDS)
            cap.raise_for_status()
            warning = parse_cap_document(cap.content, url, latitude, longitude)
        except (requests.RequestException, ValueError):
            logger.exception("Failed to fetch TMD CAP item")
            continue
        if warning:
            warning["fetched_at_utc"] = fetched_at
            if warning["active"]:
                warnings.append(warning)
            else:
                # Keep cancellation/expiry metadata auditable without placing
                # an expired alert in the active ``warnings`` list.
                recent_updates.append(warning)
    warnings.sort(key=lambda item: ({"Extreme": 0, "Severe": 1, "Moderate": 2}.get(item.get("severity"), 3),
                                    item.get("expires") or "9999"))
    return {
        "status": "ready",
        "source": "TMD_CAP",
        "source_url": TMD_CAP_INDEX_URL,
        "fetched_at_utc": fetched_at,
        "cache_age_seconds": 0.0,
        "location": {"latitude": latitude, "longitude": longitude},
        "warnings": warnings,
        "recent_updates": recent_updates[:8],
    }


def get_warnings(force=False):
    """Return active, area-matched warnings plus explicit source status."""
    now = time.monotonic()
    if not force and _cache["result"] is not None and now - _cache["fetched_at"] < TMD_CACHE_TTL_SECONDS:
        result = dict(_cache["result"])
        result["cache_age_seconds"] = round(max(0.0, now - _cache["fetched_at"]), 1)
        return result
    try:
        result = _fetch_warnings()
    except (requests.RequestException, ValueError, TypeError):
        logger.exception("TMD warning feed fetch failed")
        if _cache["result"] is not None:
            result = dict(_cache["result"])
            result.update({"status": "stale", "reason": "fetch_failed",
                           "cache_age_seconds": round(max(0.0, now - _cache["fetched_at"]), 1)})
            return result
        return {
            "status": "unavailable", "source": "TMD_CAP", "source_url": TMD_CAP_INDEX_URL,
            "fetched_at_utc": None, "cache_age_seconds": None, "warnings": [],
            "reason": "fetch_failed",
        }
    _cache.update(result=result, fetched_at=now)
    return result
