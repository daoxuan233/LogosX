"""
向量化工具（sentence-transformers）。

说明：
- 用于“语义缓存”：对 query 与摘要文本生成 embedding，并在 Redis 向量索引中进行近邻检索。
- 为保证可维护性，该模块不做模型下载与复杂管理，仅封装最常用的 encode 入口。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Embedder:
    mode: str = "hash"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384

    def __post_init__(self) -> None:
        env_mode = os.getenv("DEBATE_ARENA_EMBEDDER", "").strip().lower()
        if env_mode:
            self.mode = env_mode
        env_dim = os.getenv("DEBATE_ARENA_EMBED_DIM", "").strip()
        if env_dim:
            try:
                self.dim = int(env_dim)
            except Exception:
                pass
        self._model = None

    def embed(self, text: str) -> np.ndarray:
        if self.mode == "hash":
            return _hash_embed(text, dim=self.dim)

        if self.mode in {"st", "sentence-transformers"}:
            return self._st_embed(text)

        try:
            return self._st_embed(text)
        except Exception:
            return _hash_embed(text, dim=self.dim)

    def _st_embed(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        vec = self._model.encode([text], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)


def _hash_embed(text: str, *, dim: int) -> np.ndarray:
    dim = int(dim) if int(dim) > 0 else 384
    v = np.zeros((dim,), dtype=np.float32)
    s = (text or "").strip()
    if not s:
        return v

    data = s.encode("utf-8", errors="ignore")
    h = hashlib.blake2b(data, digest_size=32).digest()
    seed = int.from_bytes(h[:8], byteorder="little", signed=False)

    for i, b in enumerate(data):
        seed = (seed * 6364136223846793005 + 1 + b) & 0xFFFFFFFFFFFFFFFF
        idx = int(seed % dim)
        sign = 1.0 if (seed >> 63) == 0 else -1.0
        v[idx] += sign

    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm
    return v
