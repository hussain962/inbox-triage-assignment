from __future__ import annotations

import re
import unicodedata
from typing import Any

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u202a\u202b\u202c\u202d\u202e"), None)
WORD = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(ZERO_WIDTH)
    text = "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t "
    )
    return text.lower().strip()


def turn_bucket(turn_index: int | None) -> str:
    idx = turn_index or 0
    if idx <= 6:
        return "early"
    if idx <= 14:
        return "mid"
    return "late"


def render_conversation(row: dict[str, Any], intent: str | None = None) -> str:
    """Flatten a conversation prefix into model text.

    Keeps speaker tags and prior action turns. Truncates very long customer
    turns so screenshot / paste dumps do not dominate the vector.
    """
    parts: list[str] = []
    context = row.get("context") or []
    for turn in context[-8:]:
        speaker = turn.get("speaker", "")
        body = normalize_text(turn.get("text") or "")
        if speaker == "customer" and len(body) > 600:
            body = body[:300] + " ... " + body[-300:]
        parts.append(f"{speaker}: {body}")

    parts.append(f"turn_bucket:{turn_bucket(row.get('turn_index'))}")
    parts.append(f"turn_index:{row.get('turn_index') or 0}")
    if intent:
        parts.append(f"intent:{intent}")
    return "\n".join(parts)


def tokens(text: str) -> list[str]:
    return WORD.findall(text)
