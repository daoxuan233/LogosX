"""
混合搜索：SearXNG + 缓存（Redis 语义缓存 + SQLite 冷缓存）。

目标：
- 给辩手一个“像搜索一样用”的接口；
- 内部尽可能复用缓存，减少重复网络请求；
- 任何环节不可用时尽量降级，不中断辩论流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from debate_arena.search.embedder import Embedder
from debate_arena.search.redis_vector_cache import RedisVectorCache
from debate_arena.search.searxng_client import SearxngClient, SearxngSearchResponse
from debate_arena.search.sqlite_cache import SqliteCache


@dataclass(frozen=True)
class SearchOutcome:
    """
    一次检索的综合结果（包含来源信息）。

    source:
      - redis_semantic: Redis 语义命中
      - redis_exact: Redis 精确命中
      - sqlite: SQLite 命中
      - searxng: 实时搜索
      - degraded: 全部不可用
    """

    query: str
    source: str
    payload: dict[str, Any]


class HybridSearchEngine:
    """
    混合搜索引擎：优先语义命中，再实时搜索。
    """

    def __init__(
        self,
        *,
        searxng_base_url: str,
        redis_url: str,
        sqlite_path: Path,
        embedder: Embedder | None = None,
    ) -> None:
        self._client = SearxngClient(base_url=searxng_base_url)
        self._embedder: Embedder = embedder or Embedder()
        self._sqlite = SqliteCache(db_path=sqlite_path)
        self._redis = RedisVectorCache(redis_url, vector_dim=int(self._embedder.dim))

        # 尝试初始化索引（失败则不阻塞，上层会自动降级）
        try:
            self._redis.ensure_index()
        except Exception:
            pass

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def search(self, query: str) -> SearchOutcome:
        """
        搜索入口。

        1) Redis 精确命中
        2) Redis 语义命中（若可用）
        3) SQLite 冷缓存命中
        4) SearXNG 实时搜索 + 写回缓存
        5) 全部失败则降级
        """

        query = query.strip()
        if not query:
            return SearchOutcome(query=query, source="degraded", payload={"results": []})

        # 1) Redis 精确命中
        try:
            exact = self._redis.get_exact(query)
            if exact:
                return SearchOutcome(query=query, source="redis_exact", payload=exact)
        except Exception:
            pass

        # 2) Redis 语义命中
        embedding: np.ndarray | None = None
        try:
            embedding = self._embedder.embed(query)
            hits = self._redis.search_similar(embedding, k=3)
            if hits:
                best = hits[0]
                return SearchOutcome(query=query, source="redis_semantic", payload=best.payload)
        except Exception:
            pass

        # 3) SQLite 命中
        try:
            item = self._sqlite.get(query)
            if item is not None:
                return SearchOutcome(query=query, source="sqlite", payload=item.payload)
        except Exception:
            pass

        # 4) SearXNG 实时搜索
        try:
            resp: SearxngSearchResponse = self._client.search(query)
            payload = {
                "query": resp.query,
                "results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "content": r.content,
                        "engine": r.engine,
                        "score": r.score,
                    }
                    for r in resp.results
                ],
            }
            try:
                self._sqlite.set(query, payload)
            except Exception:
                pass

            try:
                if embedding is None:
                    embedding = self._embedder.embed(query)
                self._redis.set(query, payload, embedding)
            except Exception:
                pass

            return SearchOutcome(query=query, source="searxng", payload=payload)
        except Exception:
            return SearchOutcome(query=query, source="degraded", payload={"results": []})
