"""
Redis 语义缓存（Redis Stack 向量检索）。

设计要点：
- 语义缓存并不是“完全相同 query”命中，而是“相似 query”命中；
- 命中后返回已缓存的搜索结果摘要，避免重复请求 SearXNG；
- Redis 不可用时应允许降级（由上层 orchestrator 处理）。

实现策略（尽量简单且可落地）：
1) 使用 HASH 存储 item：{id} -> {query, payload_json, created_at, embedding_bytes}
2) 使用 RediSearch FT.CREATE 建立向量索引（若 Redis 端未启用 RediSearch，则跳过语义检索，退化为 key 精确命中）。
注意：不同 Redis Stack 版本与 Python 客户端能力存在差异；本模块只提供“尽最大努力”的接口，上层根据异常选择降级。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis  # pragma: no cover


@dataclass(frozen=True)
class RedisSemanticHit:
    """
    语义命中结果。
    """

    query: str
    payload: dict[str, Any]
    score: float


class RedisVectorCache:
    """
    Redis 向量语义缓存。
    """

    def __init__(
        self,
        redis_url: str,
        *,
        index_name: str = "idx:searx_cache",
        key_prefix: str = "searx_cache:",
        ttl_seconds: int = 2 * 60 * 60,
        vector_dim: int = 384,
    ) -> None:
        """
        延迟导入 redis，避免在依赖未安装时导入失败。

        依赖缺失时抛出可读错误，便于用户按文档安装。
        """

        try:
            import redis  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError("缺少依赖 redis。请使用 uv 安装项目依赖后再运行缓存相关功能。") from e

        self._redis_module = redis
        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=1.0,
            socket_timeout=2.0,
            retry_on_timeout=False,
        )
        self._index_name = index_name
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._vector_dim = vector_dim

    def ping(self) -> bool:
        return bool(self._client.ping())

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    def _make_key(self, item_id: str) -> str:
        return f"{self._key_prefix}{item_id}"

    def get_exact(self, query: str) -> dict[str, Any] | None:
        """
        精确命中：按 query 的稳定 key 查找。
        """

        key = self._make_key(self._stable_id(query))
        data = self._client.hgetall(key)
        if not data:
            return None
        payload_json = data.get(b"payload_json")
        if not payload_json:
            return None
        return json.loads(payload_json.decode("utf-8"))

    def set(self, query: str, payload: dict[str, Any], embedding: np.ndarray) -> None:
        """
        写入缓存（包含向量）。

        embedding 要求：float32、长度与 vector_dim 一致，并建议已归一化。
        """

        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        if int(embedding.shape[0]) != self._vector_dim:
            raise ValueError(f"embedding 维度不匹配：{embedding.shape[0]} != {self._vector_dim}")

        item_id = self._stable_id(query)
        key = self._make_key(item_id)
        now = time.time()

        mapping = {
            "query": query,
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "created_at": str(now),
            "embedding": embedding.tobytes(),
        }
        self._client.hset(key, mapping=mapping)
        self._client.expire(key, self._ttl_seconds)

    def search_similar(self, embedding: np.ndarray, *, k: int = 3) -> list[RedisSemanticHit]:
        """
        语义检索：返回与 embedding 最相近的 k 条缓存项。

        若 Redis 未启用 RediSearch 向量检索，则本方法会抛异常，交由上层降级。
        """

        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        if int(embedding.shape[0]) != self._vector_dim:
            raise ValueError(f"embedding 维度不匹配：{embedding.shape[0]} != {self._vector_dim}")

        q = (
            f"*=>[KNN {k} @embedding $vec AS score]"
        )
        params_dict = {"vec": embedding.tobytes()}

        res = self._client.execute_command(
            "FT.SEARCH",
            self._index_name,
            q,
            "PARAMS",
            2,
            "vec",
            params_dict["vec"],
            "RETURN",
            3,
            "query",
            "payload_json",
            "score",
            "SORTBY",
            "score",
            "DIALECT",
            2,
        )

        hits: list[RedisSemanticHit] = []
        if not res or len(res) < 2:
            return hits

        # RediSearch 返回结构：[total, doc_id, [field, value, ...], doc_id, [field, value, ...], ...]
        for i in range(1, len(res), 2):
            fields = res[i + 1]
            field_map: dict[bytes, bytes] = {}
            for j in range(0, len(fields), 2):
                field_map[fields[j]] = fields[j + 1]
            query_b = field_map.get(b"query", b"")
            payload_b = field_map.get(b"payload_json", b"{}")
            score_b = field_map.get(b"score", b"0")

            try:
                payload = json.loads(payload_b.decode("utf-8"))
            except Exception:
                payload = {}

            try:
                score = float(score_b.decode("utf-8"))
            except Exception:
                score = 0.0

            hits.append(
                RedisSemanticHit(
                    query=query_b.decode("utf-8"),
                    payload=payload,
                    score=score,
                )
            )

        return hits

    def ensure_index(self) -> None:
        """
        尝试创建 RediSearch 向量索引。

        若索引已存在会报错，我们会吞掉“索引已存在”类型错误。
        """

        try:
            self._client.execute_command(
                "FT.CREATE",
                self._index_name,
                "ON",
                "HASH",
                "PREFIX",
                1,
                self._key_prefix,
                "SCHEMA",
                "query",
                "TEXT",
                "payload_json",
                "TEXT",
                "created_at",
                "NUMERIC",
                "embedding",
                "VECTOR",
                "HNSW",
                6,
                "TYPE",
                "FLOAT32",
                "DIM",
                self._vector_dim,
                "DISTANCE_METRIC",
                "COSINE",
            )
        except self._redis_module.ResponseError as e:
            msg = str(e)
            if "Index already exists" in msg or "index already exists" in msg:
                return
            raise

    @staticmethod
    def _stable_id(query: str) -> str:
        """
        将 query 转换为稳定 id，用于精确 key。

        说明：这里使用 Python 内置 hash 不稳定，因此采用简单的 UTF-8 hex 摘要。
        """

        import hashlib

        return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
