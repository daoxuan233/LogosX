"""
LLM 工厂：统一创建对话模型实例。

该项目使用小米 mimo 的 OpenAI 兼容接口形态：
- BASE_URL=https://api.xiaomimimo.com/v1/
- API_KEY 由你手动填写
- MODEL=mimo-v2-flash
这里使用 langchain_openai.ChatOpenAI 来对接。
注意：请不要在代码/日志中输出真实 API_KEY。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def load_env() -> None:
    """
    加载本地环境变量（优先读取项目根目录 .env）。

    说明：
    - .env 不应提交到仓库
    - .env.example 作为模板
    """

    try:
        from dotenv import load_dotenv  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("缺少依赖 python-dotenv。请使用 uv 安装项目依赖后再运行。") from e

    load_dotenv(override=False)


@dataclass(frozen=True)
class _Provider:
    name: str
    api_key_env: str
    base_url_env: str
    model_env: str
    default_base_url: str
    default_model: str


class FallbackChatModel:
    def __init__(self, clients: list[tuple[str, object]]) -> None:
        self._clients = clients
        self._disabled: set[int] = set()
        self._active_idx = 0

    @property
    def active_provider(self) -> str:
        if not self._clients:
            return ""
        name, _ = self._clients[self._active_idx]
        return name

    def invoke(self, *args, **kwargs):
        if not self._clients:
            raise RuntimeError("未配置任何可用的 LLM Provider。")

        last_err: Exception | None = None
        start_idx = self._active_idx
        for offset in range(len(self._clients)):
            idx = (start_idx + offset) % len(self._clients)
            if idx in self._disabled:
                continue
            name, client = self._clients[idx]
            try:
                resp = client.invoke(*args, **kwargs)
                self._active_idx = idx
                return resp
            except Exception as e:
                last_err = e
                self._disabled.add(idx)
                continue

        raise RuntimeError(f"所有 LLM Provider 均调用失败，最后错误：{last_err}") from last_err


def make_chat_model():
    """
    创建 Chat 模型实例。

    返回：支持多 Provider 兜底的对话模型（.invoke 与 ChatOpenAI 保持兼容）
    """

    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "缺少依赖 langchain-openai。请使用 uv 安装项目依赖后再运行辩论。"
        ) from e

    providers = [
        _Provider(
            name="XIAOMIMIMO",
            api_key_env="XIAOMIMIMO_API_KEY",
            base_url_env="XIAOMIMIMO_BASE_URL",
            model_env="XIAOMIMIMO_MODEL",
            default_base_url="https://api.xiaomimimo.com/v1/",
            default_model="mimo-v2-flash",
        ),
        _Provider(
            name="DEEPSEEK",
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            model_env="DEEPSEEK_MODEL",
            default_base_url="https://api.deepseek.com/v1",
            default_model="deepseek-chat",
        ),
        _Provider(
            name="NVIDIA",
            api_key_env="NVIDIA_API_KEY",
            base_url_env="NVIDIA_BASE_URL",
            model_env="NVIDIA_MODEL",
            default_base_url="https://api.nvidia.com/v1/",
            default_model="nvidia-llama-3-1-8b-instruct",
        ),
        _Provider(
            name="OPENROUTER",
            api_key_env="OPENROUTER_API_KEY",
            base_url_env="OPENROUTER_BASE_URL",
            model_env="OPENROUTER_MODEL",
            default_base_url="https://openrouter.ai/api/v1",
            default_model="tngtech/deepseek-r1t2-chimera:free",
        ),
    ]

    clients: list[tuple[str, object]] = []
    for p in providers:
        api_key = os.getenv(p.api_key_env, "").strip()
        if not api_key:
            continue
        base_url = os.getenv(p.base_url_env, p.default_base_url).strip()
        model = os.getenv(p.model_env, p.default_model).strip()
        clients.append(
            (
                p.name,
                ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.7,
                    max_retries=2,
                    timeout=60,
                ),
            )
        )

    if not clients:
        raise RuntimeError(
            "未配置任何可用的 LLM Provider（按顺序支持：XIAOMIMIMO、DEEPSEEK、NVIDIA、OPENROUTER）。请在 .env 中至少填写一个 *_API_KEY。"
        )

    return FallbackChatModel(clients)
