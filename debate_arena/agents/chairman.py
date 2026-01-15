"""
主席（Moderator/Chairperson）Agent：强规则 + 弱生成。

主席职责：
- 控制节奏：每轮提醒、每 5 轮阶段总结、最终质量评估；
- 跑题检测：严重偏离核心论题时警告；
- 不参与辩论：不提出观点、不反驳，不代替任何一方发言。

实现原则：
- 主席输出严格限制为模板化文本（来自配置），以降低不可控生成风险；
- 计算逻辑尽量可解释、可调参。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from debate_arena.rules_engine import OffTopicScore, compute_off_topic_score, should_warn_off_topic


@dataclass(frozen=True)
class ChairmanDecision:
    """
    主席对本轮的评估输出。

    reminder: 开场提醒（模板）
    evaluation: 本轮评估（模板/简短文本）
    warning: 可选跑题警告（模板）
    """

    reminder: str
    evaluation: str
    warning: str | None = None


class ChairmanAgent:
    """
    主席 Agent（强规则控场）。

    依赖：
    - chairman_rules.yaml（阈值、模板、评分权重等）
    """

    def __init__(self, chairman_rules: dict[str, Any]) -> None:
        self._rules = chairman_rules

    def start_round(self, round_num: int) -> str:
        """
        主席开场提醒（严格模板化）。

        注意：主席不生成观点，只提醒规则与节奏。
        """

        limits = self._rules.get("limits", {})
        max_chars = int(limits.get("max_chars_per_speech", 700))
        tmpl = (self._rules.get("templates", {}) or {}).get("round_reminder", "")
        if not tmpl:
            tmpl = "请双方围绕核心论题展开，每轮发言不超过{max_chars}字。"
        return tmpl.format(max_chars=max_chars, round=round_num)

    def evaluate_round(
        self,
        *,
        topic: str,
        round_num: int,
        pro_speech_text: str,
        opp_speech_text: str,
    ) -> tuple[ChairmanDecision, OffTopicScore, OffTopicScore]:
        """
        对本轮进行评估。

        当前以可解释的规则打分为主：关键词覆盖 + 相似度（可降级）。
        后续可接入 embedding 相似度或更强的主题一致性模型。
        """

        reminder = self.start_round(round_num)
        pro_score = compute_off_topic_score(
            topic=topic,
            speech=pro_speech_text,
            threshold_cfg=self._rules,
            similarity_hint=None,
        )
        opp_score = compute_off_topic_score(
            topic=topic,
            speech=opp_speech_text,
            threshold_cfg=self._rules,
            similarity_hint=None,
        )

        warning_parts: list[str] = []
        tmpl_warning = (self._rules.get("templates", {}) or {}).get("off_topic_warning", "")
        if not tmpl_warning:
            tmpl_warning = "警告：检测到本轮发言偏离核心论题（偏离度 {score:.0%}）。"

        if should_warn_off_topic(pro_score, self._rules):
            warning_parts.append(tmpl_warning.format(score=pro_score.score))
        if should_warn_off_topic(opp_score, self._rules):
            warning_parts.append(tmpl_warning.format(score=opp_score.score))

        warning = " ".join(warning_parts) if warning_parts else None

        # 本轮评估：保持极简与可控（主席弱生成）
        evaluation = (
            f"本轮评估：正方偏离度 {pro_score.score:.0%}（关键词覆盖 {pro_score.keyword_coverage:.0%}），"
            f"反方偏离度 {opp_score.score:.0%}（关键词覆盖 {opp_score.keyword_coverage:.0%}）。"
            "请双方下一轮继续直接回应对方要点，并补强哲学依据与故事寓意映射。"
        )

        return ChairmanDecision(reminder=reminder, evaluation=evaluation, warning=warning), pro_score, opp_score

    def stage_summary(self, *, topic: str, round_num: int) -> str | None:
        """
        每 N 轮生成阶段总结。

        由于主席弱生成，本阶段总结仍采用模板，核心内容留白为可解释字段。
        """

        stage_cfg = self._rules.get("stage_summary", {}) if isinstance(self._rules, dict) else {}
        every_n = int(stage_cfg.get("every_n_rounds", 5))
        if every_n <= 0 or round_num % every_n != 0:
            return None

        tmpl = (self._rules.get("templates", {}) or {}).get("stage_summary", "")
        if not tmpl:
            tmpl = "阶段性总结（第{round}轮）：请继续围绕“{topic}”展开。"

        # dispute/advice 先用占位，后续可用更强的摘要器从双方结构化字段提取
        return tmpl.format(round=round_num, topic=topic, n=every_n, dispute="（待提炼）", advice="（待提炼）")
