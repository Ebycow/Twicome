import html
import json
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import pytz


def seconds_to_hms(total: int) -> str:
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def seconds_to_twitch_t(total: int) -> str:
    # Twitch: ?t=1h2m3s
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    out = ""
    if h:
        out += f"{h}h"
    if m:
        out += f"{m}m"
    out += f"{s}s"
    return out


def utc_to_jst(dt: datetime) -> datetime:
    utc_tz = pytz.timezone("UTC")
    jst_tz = pytz.timezone("Asia/Tokyo")
    if dt.tzinfo is None:
        dt = utc_tz.localize(dt)
    return dt.astimezone(jst_tz)


def build_vod_link(url: Optional[str], offset_seconds: int) -> Optional[str]:
    if not url:
        return None
    # 既に ? がある場合も想定（必要なら厳密化）
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={seconds_to_twitch_t(offset_seconds)}"


def build_youtube_link(url: Optional[str], offset_seconds: int) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        params["t"] = f"{max(0, int(offset_seconds))}s"
        return urlunparse(parsed._replace(query=urlencode(params)))
    except Exception:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}t={max(0, int(offset_seconds))}s"


def split_filter_terms(raw: Optional[str]):
    if not raw:
        return []
    return [term for term in re.split(r"[\s,、]+", raw.strip()) if term]


EMOTE_URL_TEMPLATE = "https://static-cdn.jtvnw.net/emoticons/v2/{emote_id}/default/dark/{scale}"
EMOTE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def normalize_emote_id(raw_emote_id) -> Optional[str]:
    if raw_emote_id is None:
        return None
    emote_id = str(raw_emote_id).strip()
    if not emote_id:
        return None
    if not EMOTE_ID_PATTERN.fullmatch(emote_id):
        return None
    return emote_id


def parse_raw_comment(raw_json):
    if not raw_json:
        return None
    if isinstance(raw_json, dict):
        return raw_json
    try:
        return json.loads(raw_json)
    except Exception:
        return None


def render_comment_body_html(raw_json, fallback_body):
    data = parse_raw_comment(raw_json)
    fragments = None
    if isinstance(data, dict):
        msg = data.get("message") or {}
        fragments = msg.get("fragments")

    if not fragments:
        return html.escape(fallback_body or "")

    parts = []
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        text = frag.get("text") or ""
        emoticon = frag.get("emoticon") or {}
        emote_id = normalize_emote_id(emoticon.get("emoticon_id"))
        if emote_id:
            escaped = html.escape(text)
            emote_id_url = quote(emote_id, safe="")
            url1 = html.escape(EMOTE_URL_TEMPLATE.format(emote_id=emote_id_url, scale="1.0"), quote=True)
            url2 = html.escape(EMOTE_URL_TEMPLATE.format(emote_id=emote_id_url, scale="2.0"), quote=True)
            url3 = html.escape(EMOTE_URL_TEMPLATE.format(emote_id=emote_id_url, scale="3.0"), quote=True)
            parts.append(
                f'<img class="emote" src="{url1}" srcset="{url2} 2x, {url3} 3x" alt="{escaped}" title="{escaped}" loading="lazy" decoding="async">'
            )
        else:
            parts.append(html.escape(text))

    if not parts:
        return html.escape(fallback_body or "")

    return "".join(parts)


def decorate_comment(row, now):
    r = dict(row)
    raw_json = r.pop("raw_json", None)
    r["body_html"] = render_comment_body_html(raw_json, r.get("body"))
    offset_sec = int(r.get("offset_seconds") or 0)
    created_at = r.get("comment_created_at_utc")
    comment_created_at_jst = None
    relative_time = None
    is_recent = False
    if created_at:
        jst_dt = utc_to_jst(created_at)
        comment_created_at_jst = jst_dt.strftime("%Y-%m-%d %H:%M:%S")
        delta = now - created_at
        if delta.days == 0:
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            if hours > 0:
                relative_time = f"{hours}時間{minutes}分前"
            else:
                relative_time = f"{minutes}分前"
        else:
            relative_time = f"{delta.days}日前"
        is_recent = delta < timedelta(hours=24)

    r.update(
        {
            "offset_hms": seconds_to_hms(offset_sec),
            "vod_link": build_vod_link(r.get("vod_url"), offset_sec),
            "vod_jump_link": f"https://www.twitch.tv/videos/{r.get('vod_id')}?t={seconds_to_twitch_t(offset_sec)}",
            "youtube_jump_link": build_youtube_link(r.get("youtube_url"), offset_sec),
            "comment_created_at_jst": comment_created_at_jst,
            "relative_time": relative_time,
            "is_recent": is_recent,
        }
    )
    return r


# Backward-compatible aliases to keep route logic unchanged during modularization.
_split_filter_terms = split_filter_terms
_parse_raw_comment = parse_raw_comment
_render_comment_body_html = render_comment_body_html
_decorate_comment = decorate_comment
