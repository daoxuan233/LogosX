"""
辩论文档管理器：将全流程内容集中输出为单一中文 Markdown。

要求：
- 结构稳定：便于阅读与复盘
- 每轮固定块：主席提醒 → 正方 → 反方 → 主席评估（必要时警告）
- 最终追加：质量评估、核心洞见、未尽议题
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DebateDocument:
    """
    辩论 Markdown 文档的内存表示。

    使用 “append block” 的方式累积内容，最后一次性导出。
    """

    topic: str
    chairman_name: str
    proponent_name: str
    opponent_name: str
    rounds: int
    created_at: datetime = field(default_factory=datetime.now)
    blocks: list[str] = field(default_factory=list)
    final_blocks: list[str] = field(default_factory=list)

    def add_block(self, markdown: str) -> None:
        self.blocks.append(markdown.rstrip() + "\n")

    def add_final_block(self, markdown: str) -> None:
        self.final_blocks.append(markdown.rstrip() + "\n")

    def export_markdown(self) -> str:
        header = [
            f"# 【哲学辩论】{self.topic}",
            "",
            "## 📋 辩论基本信息",
            f"- **主席**：{self.chairman_name}",
            f"- **正方**：{self.proponent_name}",
            f"- **反方**：{self.opponent_name}",
            f"- **总轮次**：{self.rounds}轮",
            f"- **生成时间**：{self.created_at.strftime('%Y年%m月%d日 %H:%M:%S')}",
            "",
            "## ⚖️ 核心论题",
            self.topic,
            "",
            "---",
            "",
            "## 🎭 辩论实录",
            "",
        ]
        body = "".join(self.blocks)
        tail = ["", "---", "", "## 📊 辩论质量评估", ""] + self.final_blocks
        return "\n".join(header) + body + "\n".join(tail)


def format_speaker_block(side_label: str, philosopher: str, content: dict[str, str]) -> str:
    """
    将辩手结构化内容渲染为 Markdown 块。

    content 建议包含：
    - 回应对方要点
    - 本轮立论
    - 反驳点
    - 建设性论点
    - 哲学依据
    - 故事
    - 故事寓意
    """

    lines: list[str] = []
    lines.append(f"**{side_label}**（{philosopher}）：")
    lines.append(">")
    group1 = ["回应对方要点", "本轮立论", "反驳点", "建设性论点"]
    group2 = ["哲学依据", "故事", "故事寓意"]
    rendered: set[str] = set()

    has_group1 = False
    for k in group1:
        v = (content.get(k) or "").strip()
        if not v:
            continue
        lines.append(f"> **{k}**：{v}")
        rendered.add(k)
        has_group1 = True

    extra_keys: list[str] = []
    for k in group2:
        v = (content.get(k) or "").strip()
        if v:
            extra_keys.append(k)
    for k, v in content.items():
        kk = str(k)
        vv = (v or "").strip()
        if not vv:
            continue
        if kk in rendered or kk in set(group1) or kk in set(group2):
            continue
        extra_keys.append(kk)

    if extra_keys and has_group1:
        lines.append(">")

    for k in extra_keys:
        v = (content.get(k) or "").strip()
        if v:
            lines.append(f"> **{k}**：{v}")
            rendered.add(k)
    lines.append("")
    return "\n".join(lines)
