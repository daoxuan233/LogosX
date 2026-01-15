"""
SearXNG 搜索客户端（HTTP JSON）。

本项目使用本地 docker 中的 SearXNG（见 searxng-docker/），通过 HTTP 调用：
- 默认地址：http://localhost:8081
- 搜索接口：GET /search?format=json&q=...

该模块只做最小封装，便于后续缓存/摘要/向量化层复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class SearxngResult:
    """
    单条搜索结果的标准化结构。
    """

    title: str
    url: str
    content: str
    engine: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class SearxngSearchResponse:
    """
    一次搜索请求的标准化响应结构。
    """

    query: str
    results: list[SearxngResult]
    raw: dict[str, Any]


class SearxngClient:
    """
    SearXNG JSON 搜索客户端。
    """

    def __init__(self, base_url: str, timeout_s: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def search(
        self,
        query: str,
        *,
        lang: str = "zh-CN",
        safe_search: int = 0,
        max_results: int = 8,
    ) -> SearxngSearchResponse:
        """
        执行搜索并返回标准化结果。

        参数说明：
        - lang: 搜索语言偏好（对应 SearXNG settings.yml 的 default_lang）
        - safe_search: 0/1/2，越大越严格
        - max_results: 最多保留多少条结果（用于控制上下文注入长度）
        """

        params = {
            "q": query,
            "format": "json",
            "language": lang,
            "safesearch": safe_search,
        }
        url = f"{self._base_url}/search"
        resp = requests.get(url, params=params, timeout=self._timeout_s)
        resp.raise_for_status()
        data = resp.json()

        normalized: list[SearxngResult] = []
        for item in data.get("results", [])[:max_results]:
            normalized.append(
                SearxngResult(
                    title=str(item.get("title") or "").strip(),
                    url=str(item.get("url") or "").strip(),
                    content=str(item.get("content") or "").strip(),
                    engine=item.get("engine"),
                    score=float(item["score"]) if "score" in item and item["score"] is not None else None,
                )
            )

        return SearxngSearchResponse(query=query, results=normalized, raw=data)

