"""Unified cache service for wrapping tool calls."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

from .db import SQLiteCacheStore
from .key_builder import (
    build_cache_key,
    build_payload,
    normalize_options,
    normalize_query,
    serialize_payload,
)
from .policy import NEVER_EXPIRE, get_error_ttl_seconds, get_ttl_seconds


def _default_cache_db_path() -> Path:
    return Path(os.getenv("MMDEEPSEARCH_CACHE_DB", Path(__file__).resolve().parents[4] / ".cache" / "search_cache.db"))


def _is_logical_failure_dict(response: Any) -> bool:
    """
    Fetcher returned a dict that represents failure without raising.

    If we persist these as cache status=success, callers get stale errors for
    the full success TTL (e.g. Jina Reader 3600s) and never retry the network.
    """
    return isinstance(response, dict) and response.get("ok") is False


class SearchCacheService:
    """Single entry point: cache lookup, fetch fallback, and persistence."""

    def __init__(
        self,
        db_path: str = "search_cache.db",
        *,
        cleanup_every: int = 100,
        enable_touch_on_hit: bool | None = None,
        touch_batch_size: int | None = None,
        touch_flush_interval_seconds: float | None = None,
    ):
        self._store = SQLiteCacheStore(db_path=db_path)
        self._call_counter = 0
        self._cleanup_every = max(1, cleanup_every)
        self._cleanup_lock = threading.Lock()
        self._touch_lock = threading.Lock()
        self._touch_flush_event = threading.Event()
        self._touch_stop_event = threading.Event()
        self._pending_touches: dict[str, tuple[int, int]] = {}
        self._pending_touch_count = 0
        self._closed = False
        self._touch_enabled = self._resolve_touch_enabled(enable_touch_on_hit)
        self._touch_batch_size = max(
            1,
            touch_batch_size
            if touch_batch_size is not None
            else int(os.getenv("MMDEEPSEARCH_CACHE_TOUCH_BATCH_SIZE", "64")),
        )
        self._touch_flush_interval_seconds = max(
            0.1,
            touch_flush_interval_seconds
            if touch_flush_interval_seconds is not None
            else float(os.getenv("MMDEEPSEARCH_CACHE_TOUCH_FLUSH_INTERVAL_SECONDS", "2.0")),
        )
        self._touch_thread: threading.Thread | None = None
        if self._touch_enabled:
            self._touch_thread = threading.Thread(
                target=self._touch_flush_loop,
                name="search-cache-touch-flush",
                daemon=True,
            )
            self._touch_thread.start()

    @staticmethod
    def _resolve_touch_enabled(explicit_value: bool | None) -> bool:
        if explicit_value is not None:
            return explicit_value
        raw_value = os.getenv("MMDEEPSEARCH_CACHE_TOUCH_ON_HIT")
        if raw_value is None:
            return True
        return raw_value.strip().lower() not in {"", "0", "false", "no", "off"}

    @staticmethod
    def _build_request_json(tool_name: str, query: str, options: dict[str, Any] | None) -> str:
        """
        Build request payload for storage/inspection.

        Keep topk controls excluded (same as keying), but preserve "__trace"
        so raw inputs (e.g. original URL) are available for debugging.
        """
        request_payload = {
            "tool_name": tool_name,
            "query": normalize_query(query),
            "options": normalize_options(options, exclude_keys={"topk", "k"}),
        }
        return serialize_payload(request_payload)

    def _maybe_cleanup(self) -> None:
        should_cleanup = False
        with self._cleanup_lock:
            self._call_counter += 1
            should_cleanup = self._call_counter % self._cleanup_every == 0
        if should_cleanup:
            self.cleanup_expired()

    def _enqueue_touch(self, cache_key: str, now: int) -> None:
        if not self._touch_enabled or self._closed:
            return
        should_flush = False
        with self._touch_lock:
            last_accessed, hit_count = self._pending_touches.get(cache_key, (0, 0))
            self._pending_touches[cache_key] = (max(last_accessed, now), hit_count + 1)
            self._pending_touch_count += 1
            should_flush = self._pending_touch_count >= self._touch_batch_size
        if should_flush:
            self._touch_flush_event.set()

    def _drain_pending_touches(self) -> list[tuple[str, int, int]]:
        with self._touch_lock:
            if not self._pending_touches:
                return []
            pending = [
                (cache_key, last_accessed, hit_count)
                for cache_key, (last_accessed, hit_count) in self._pending_touches.items()
            ]
            self._pending_touches = {}
            self._pending_touch_count = 0
        return pending

    def _touch_flush_loop(self) -> None:
        while not self._touch_stop_event.is_set():
            self._touch_flush_event.wait(timeout=self._touch_flush_interval_seconds)
            self._touch_flush_event.clear()
            self.flush_pending_touches()
        self.flush_pending_touches()

    def flush_pending_touches(self) -> None:
        pending = self._drain_pending_touches()
        if pending:
            self._store.touch_records(pending)

    @staticmethod
    def _compute_expire_at(now: int, ttl_seconds: int) -> int:
        if ttl_seconds == NEVER_EXPIRE:
            return NEVER_EXPIRE
        return now + ttl_seconds

    @staticmethod
    def _is_record_unexpired(record: dict[str, Any], now: int) -> bool:
        expire_at = int(record["expire_at"])
        return expire_at == NEVER_EXPIRE or expire_at >= now

    def get_or_fetch(
        self,
        tool_name: str,
        query: str,
        options: dict[str, Any] | None,
        real_fetch_func: Callable[[str, dict[str, Any]], Any],
    ) -> Any:
        opts = dict(options or {})
        now = int(time.time())

        payload = build_payload(tool_name=tool_name, query=query, options=opts)
        cache_key, _ = build_cache_key(payload)
        request_json = self._build_request_json(tool_name=tool_name, query=query, options=opts)

        record = self._store.get_record(cache_key)

        if record and self._is_record_unexpired(record, now):
            if record["status"] == "error":
                error_obj = json.loads(record["response_json"])
                message = str(error_obj.get("error", "cached error"))
                raise RuntimeError(message)
            if record["status"] == "success":
                response_obj = json.loads(record["response_json"])
                self._enqueue_touch(cache_key=cache_key, now=now)
                self._maybe_cleanup()
                return response_obj

        try:
            # Never hold DB transaction/lock while executing real tool call,
            # because external API latency would block cache reads/writes.
            fresh_response = real_fetch_func(query, opts)
        except Exception as exc:
            ttl = get_error_ttl_seconds(tool_name=tool_name, query=query, options=opts)
            error_payload = {"error": str(exc)}
            self._store.upsert_record(
                cache_key=cache_key,
                tool_name=tool_name,
                query_text=query,
                request_json=request_json,
                response_json=json.dumps(error_payload, ensure_ascii=False),
                max_topk=0,
                status="error",
                created_at=now,
                expire_at=self._compute_expire_at(now, ttl),
                last_accessed=now,
            )
            self._maybe_cleanup()
            raise

        if _is_logical_failure_dict(fresh_response):
            logger.info(
                "search_cache skip_persist_logical_failure tool_name=%r cache_key=%s",
                tool_name,
                cache_key[:16] + "..." if len(cache_key) > 16 else cache_key,
            )
            self._maybe_cleanup()
            return fresh_response

        ttl = get_ttl_seconds(tool_name=tool_name, query=query, options=opts)
        self._store.upsert_record(
            cache_key=cache_key,
            tool_name=tool_name,
            query_text=query,
            request_json=request_json,
            response_json=json.dumps(fresh_response, ensure_ascii=False),
            max_topk=0,
            status="success",
            created_at=now,
            expire_at=self._compute_expire_at(now, ttl),
            last_accessed=now,
        )
        self._maybe_cleanup()
        return fresh_response

    async def aget_or_fetch(
        self,
        tool_name: str,
        query: str,
        options: dict[str, Any] | None,
        real_fetch_func: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        opts = dict(options or {})
        now = int(time.time())

        payload = build_payload(tool_name=tool_name, query=query, options=opts)
        cache_key, _ = build_cache_key(payload)
        request_json = self._build_request_json(tool_name=tool_name, query=query, options=opts)

        record = self._store.get_record(cache_key)

        if record and self._is_record_unexpired(record, now):
            if record["status"] == "error":
                error_obj = json.loads(record["response_json"])
                message = str(error_obj.get("error", "cached error"))
                raise RuntimeError(message)
            if record["status"] == "success":
                response_obj = json.loads(record["response_json"])
                self._enqueue_touch(cache_key=cache_key, now=now)
                self._maybe_cleanup()
                return response_obj

        try:
            # Never hold DB transaction/lock while executing real tool call,
            # because external API latency would block cache reads/writes.
            fresh_response = await real_fetch_func(query, opts)
        except Exception as exc:
            ttl = get_error_ttl_seconds(tool_name=tool_name, query=query, options=opts)
            error_payload = {"error": str(exc)}
            self._store.upsert_record(
                cache_key=cache_key,
                tool_name=tool_name,
                query_text=query,
                request_json=request_json,
                response_json=json.dumps(error_payload, ensure_ascii=False),
                max_topk=0,
                status="error",
                created_at=now,
                expire_at=self._compute_expire_at(now, ttl),
                last_accessed=now,
            )
            self._maybe_cleanup()
            raise

        if _is_logical_failure_dict(fresh_response):
            logger.info(
                "search_cache skip_persist_logical_failure tool_name=%r cache_key=%s",
                tool_name,
                cache_key[:16] + "..." if len(cache_key) > 16 else cache_key,
            )
            self._maybe_cleanup()
            return fresh_response

        ttl = get_ttl_seconds(tool_name=tool_name, query=query, options=opts)
        self._store.upsert_record(
            cache_key=cache_key,
            tool_name=tool_name,
            query_text=query,
            request_json=request_json,
            response_json=json.dumps(fresh_response, ensure_ascii=False),
            max_topk=0,
            status="success",
            created_at=now,
            expire_at=self._compute_expire_at(now, ttl),
            last_accessed=now,
        )
        self._maybe_cleanup()
        return fresh_response

    def cleanup_expired(self) -> int:
        now = int(time.time())
        self.flush_pending_touches()
        return self._store.delete_expired(now=now)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._touch_thread is not None:
            self._touch_stop_event.set()
            self._touch_flush_event.set()
            self._touch_thread.join(timeout=max(1.0, self._touch_flush_interval_seconds + 1.0))
        self.flush_pending_touches()
        self._store.close()


_cache_service: SearchCacheService | None = None
_cache_service_lock = threading.Lock()


def get_cache_service() -> SearchCacheService:
    global _cache_service
    if _cache_service is None:
        with _cache_service_lock:
            if _cache_service is None:
                db_path = os.getenv("MMDEEPSEARCH_CACHE_DB", "").strip()
                if not db_path:
                    db_path = str(_default_cache_db_path())
                logger.info("Search cache DB path: %s", db_path)
                _cache_service = SearchCacheService(db_path=db_path)
    return _cache_service
