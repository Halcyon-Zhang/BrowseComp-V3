"""SQLite storage for tool response cache."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .policy import NEVER_EXPIRE


class SQLiteCacheStore:
    """Persist cache entries in a local SQLite database."""

    def __init__(self, db_path: str):
        path_obj = Path(db_path).expanduser()
        if path_obj.parent and not path_obj.parent.exists():
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path_obj)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._init_db()

    def _create_connection(self, *, register: bool = True) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=3000;")
        if register:
            with self._connections_lock:
                self._connections.append(conn)
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._create_connection()
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._create_connection(register=False)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    max_topk INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expire_at INTEGER NOT NULL,
                    last_accessed INTEGER NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expire_at ON api_cache(expire_at);"
            )
            conn.commit()
        finally:
            conn.close()

    def get_record(self, cache_key: str) -> dict[str, Any] | None:
        cursor = self._get_connection().execute(
            "SELECT * FROM api_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def upsert_record(
        self,
        *,
        cache_key: str,
        tool_name: str,
        query_text: str,
        request_json: str,
        response_json: str,
        max_topk: int,
        status: str,
        created_at: int,
        expire_at: int,
        last_accessed: int,
    ) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO api_cache(
                    cache_key, tool_name, query_text, request_json, response_json,
                    max_topk, status, created_at, expire_at, last_accessed, hit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    tool_name = excluded.tool_name,
                    query_text = excluded.query_text,
                    request_json = excluded.request_json,
                    response_json = excluded.response_json,
                    max_topk = excluded.max_topk,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    expire_at = excluded.expire_at,
                    last_accessed = excluded.last_accessed
                """,
                (
                    cache_key,
                    tool_name,
                    query_text,
                    request_json,
                    response_json,
                    max_topk,
                    status,
                    created_at,
                    expire_at,
                    last_accessed,
                ),
            )

    def touch_record(self, cache_key: str, now: int) -> None:
        self.touch_records([(cache_key, now, 1)])

    def touch_records(self, records: list[tuple[str, int, int]]) -> None:
        if not records:
            return
        conn = self._get_connection()
        with conn:
            conn.executemany(
                """
                UPDATE api_cache
                SET last_accessed = CASE
                        WHEN last_accessed < ? THEN ?
                        ELSE last_accessed
                    END,
                    hit_count = hit_count + ?
                WHERE cache_key = ?
                """,
                [(last_accessed, last_accessed, hit_count, cache_key) for cache_key, last_accessed, hit_count in records],
            )

    def delete_expired(self, now: int) -> int:
        conn = self._get_connection()
        with conn:
            cursor = conn.execute(
                "DELETE FROM api_cache WHERE expire_at != ? AND expire_at < ?",
                (NEVER_EXPIRE, now),
            )
        return int(cursor.rowcount or 0)

    def close(self) -> None:
        with self._connections_lock:
            connections = self._connections
            self._connections = []
        for conn in connections:
            conn.close()
        if hasattr(self._local, "conn"):
            del self._local.conn
