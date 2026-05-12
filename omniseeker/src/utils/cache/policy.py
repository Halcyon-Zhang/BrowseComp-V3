"""Cache policy definitions."""

from __future__ import annotations

from .protocol import (
    TOOL_IMAGE_UPLOAD_PUBLIC_URL,
    TOOL_JINA_READER,
    TOOL_SERPER_IMAGES,
    TOOL_SERPER_LENS,
)

NEVER_EXPIRE = -1


def get_ttl_seconds(tool_name: str, query: str, options: dict) -> int:
    """TTL for successful responses."""
    q = (query or "").lower()
    if tool_name == TOOL_IMAGE_UPLOAD_PUBLIC_URL:
        return NEVER_EXPIRE
    if tool_name in {TOOL_JINA_READER, "jina.reader.visit"}:
        return 3600
    if tool_name in {TOOL_SERPER_IMAGES, TOOL_SERPER_LENS, "serper.reverse_image", "image_search"}:
        return 86400
    if any(word in q for word in ("latest", "today", "news", "price", "weather")):
        return 900
    return NEVER_EXPIRE


def get_error_ttl_seconds(tool_name: str, query: str, options: dict) -> int:
    """Short-lived negative cache for repeated failing requests."""
    return 120
