"""
SQLite 冷缓存（持久化）。

目的：
- Redis 作为热缓存（TTL），SQLite 作为冷缓存（长期保存）。
- 当 Redis 不可用或 key 过期时，仍可从 SQLite 命中并回填。

保持 KISS：
- 使用标准库 sqlite3；
- 只存储必要字段：query、归一化结果 JSON、embedding（可选）、时间戳。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SqliteCacheItem:
    query: str
    payload: dict[str, Any]
    created_at: float


class SqliteCache:
    """
    SQLite 持久化缓存。
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path.as_posix())
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS searx_cache (
                  query TEXT PRIMARY KEY,
                  payload_json TEXT NOT NULL,
                  created_at REAL NOT NULL
                );
                """
            )

    def get(self, query: str) -> SqliteCacheItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, created_at FROM searx_cache WHERE query = ?",
                (query,),
            ).fetchone()
        if not row:
            return None
        payload_json, created_at = row
        return SqliteCacheItem(query=query, payload=json.loads(payload_json), created_at=float(created_at))

    def set(self, query: str, payload: dict[str, Any]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO searx_cache(query, payload_json, created_at) VALUES (?, ?, ?)",
                (query, json.dumps(payload, ensure_ascii=False), now),
            )

