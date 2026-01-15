"""
文本工具：长度裁剪、格式清洗等。
"""

from __future__ import annotations


def truncate_chars(text: str, max_chars: int) -> str:
    """
    将文本裁剪到最大字符数。

    说明：以 Python 字符计数（中文通常按一个字符计）。
    """

    text = text.strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
