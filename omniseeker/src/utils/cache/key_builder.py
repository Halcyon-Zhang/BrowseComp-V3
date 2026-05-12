"""Helpers to build stable cache keys from tool requests."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlparse, urlunparse


def serialize_payload(payload: dict[str, Any]) -> str:
    """Serialize payload deterministically for keying/storage."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def normalize_query(query: str) -> str:
    """Trim and collapse spaces so semantically same query maps to one key."""
    return " ".join((query or "").strip().split())


def normalize_url(url: str) -> str:
    """Normalize URL for stable cross-runtime cache identities."""
    if not isinstance(url, str):
        url = str(url)
    url = url.strip()
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.endswith(":80") and parsed.scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and parsed.scheme == "https":
            netloc = netloc[:-4]
        normalized = parsed._replace(netloc=netloc, fragment="")
        return urlunparse(normalized)
    except Exception:
        return url.strip()


def build_url_query(url: str) -> str:
    """Build stable URL-backed cache query in `url_sha256:<digest>` form."""
    normalized = normalize_url(url)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"url_sha256:{digest}"


def build_upload_query(content_sha256: str) -> str:
    """Build stable upload-backed cache query in `upload_sha256:<digest>` form."""
    digest = str(content_sha256 or "").strip().lower()
    if not digest:
        raise ValueError("content_sha256 is required for upload cache query")
    return f"upload_sha256:{digest}"


def normalize_options(
    options: dict[str, Any] | None,
    exclude_keys: set[str] | None = None,
) -> dict[str, Any]:
    """
    Drop None values, remove excluded keys, and keep deterministic key order.
    """
    if not options:
        return {}

    excluded = exclude_keys or set()
    return {
        key: options[key]
        for key in sorted(options.keys())
        if options[key] is not None and key not in excluded
    }


def build_payload(tool_name: str, query: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """
    Build normalized payload used to generate cache key.
    """
    # Cache raw responses and keep post-processing controls (e.g. topk/k)
    # outside the key so downstream slicing can reuse the same entry.
    key_options = normalize_options(options, exclude_keys={"topk", "k", "__trace"})
    return {
        "tool_name": tool_name,
        "query": normalize_query(query),
        "options": key_options,
    }


def build_cache_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return (sha256 cache_key, serialized payload json)."""
    serialized = serialize_payload(payload)
    cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return cache_key, serialized
