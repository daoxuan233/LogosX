"""
JSON 工具：从 LLM 输出中稳健提取 JSON。

现实问题：
- 模型有时会把 JSON 包在 ```json ... ``` 代码块里；
- 或者在 JSON 前后附加解释文字；
- 我们需要尽量“宽容解析”，但在失败时要给出明确错误，便于调参与排错。
"""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """
    从文本中提取第一个 JSON 对象并解析为 dict。

    解析策略：
    1) 去掉常见代码块围栏（```json / ```）；
    2) 在文本中寻找第一个 '{' 与最后一个 '}' 的闭合区间；
    3) 尝试 json.loads；失败则抛出 ValueError。
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未在模型输出中找到 JSON 对象边界")

    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败：{e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("模型输出的 JSON 顶层必须是对象(dict)")

    return data

