"""
辩手 Agent：正方/反方共用逻辑，通过角色 YAML 注入风格差异。

能力要求：
- 真实辩论：必须回应对方上一轮要点，且包含反驳与建设性论点；
- 哲学性：引用哲学概念与经典著作；
- 文学性：善于将道理放到故事中，且故事必须服务论点；
- 敏感内容：触发时必须用故事表达（避免直接点名敏感对象）。

实现策略（KISS）：两阶段输出
1) 计划阶段（JSON）：是否需要搜索、搜索 query、论证纲要
2) 生成阶段（JSON）：输出固定字段，便于文档拼装与规则校验
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from debate_arena.rules_engine import detect_sensitive
from debate_arena.search.hybrid_search import HybridSearchEngine
from debate_arena.utils.json_utils import extract_json_object
from debate_arena.utils.text_utils import truncate_chars


@dataclass(frozen=True)
class DebaterOutput:
    """
    辩手输出（结构化）。

    content: 包含固定字段的 dict
    raw_text: 模型原始输出（用于排错）
    """

    content: dict[str, str]
    raw_text: str


class DebaterAgent:
    """
    辩手 Agent。

    说明：
    - 该类依赖 LLM（langchain_openai.ChatOpenAI）
    - 搜索由外部 HybridSearchEngine 提供
    """

    def __init__(
        self,
        *,
        side_label: str,
        role_config: dict[str, Any],
        debate_rules: dict[str, Any],
        sensitive_config: dict[str, Any],
        chairman_rules: dict[str, Any],
        llm,
        search_engine: HybridSearchEngine,
    ) -> None:
        self._side_label = side_label
        self._role = role_config
        self._rules = debate_rules
        self._sensitive = sensitive_config
        self._chairman_rules = chairman_rules
        self._llm = llm
        self._search = search_engine

    @property
    def philosopher_name(self) -> str:
        return str(self._role.get("role_name") or "")

    def generate_turn(
        self,
        *,
        motion_topic: str,
        side_topic: str,
        stage: str,
        task_instruction: str,
        round_num: int,
        opponent_last: dict[str, str] | None,
    ) -> DebaterOutput:
        """
        生成本轮发言（结构化 JSON）。

        opponent_last: 对方上一轮结构化输出（若为首轮则可能为空）
        """

        limits = (self._chairman_rules.get("limits", {}) or {})
        max_chars = int(limits.get("max_chars_per_speech", 700))

        # 敏感触发：topic 或 opponent_last 触发都算（宁可多触发、也不漏触发）
        last_text = json.dumps(opponent_last or {}, ensure_ascii=False)
        topic_text = f"{motion_topic}\n{side_topic}\n{stage}\n{task_instruction}"
        sensitive = detect_sensitive(topic_text, self._sensitive) or detect_sensitive(last_text, self._sensitive)

        plan = self._generate_plan(
            motion_topic=motion_topic,
            side_topic=side_topic,
            stage=stage,
            task_instruction=task_instruction,
            round_num=round_num,
            opponent_last=opponent_last,
            sensitive=sensitive,
        )

        search_snippets: list[dict[str, Any]] = []
        if bool(plan.get("need_search")):
            for q in (plan.get("search_queries") or [])[:3]:
                if not isinstance(q, str) or not q.strip():
                    continue
                outcome = self._search.search(q.strip())
                search_snippets.append(
                    {
                        "query": q.strip(),
                        "source": outcome.source,
                        "results": (outcome.payload.get("results") or [])[:5],
                    }
                )

        out = self._generate_speech(
            motion_topic=motion_topic,
            side_topic=side_topic,
            stage=stage,
            task_instruction=task_instruction,
            round_num=round_num,
            opponent_last=opponent_last,
            sensitive=sensitive or bool(plan.get("sensitive")),
            plan=plan,
            search_snippets=search_snippets,
            max_chars=max_chars,
        )

        # 最终硬性长度裁剪（防止模型偶发超长）
        for k, v in list(out.content.items()):
            out.content[k] = truncate_chars(v, max_chars=max_chars)

        return out

    def _generate_plan(
        self,
        *,
        motion_topic: str,
        side_topic: str,
        stage: str,
        task_instruction: str,
        round_num: int,
        opponent_last: dict[str, str] | None,
        sensitive: bool,
    ) -> dict[str, Any]:
        """
        计划阶段：输出 JSON，用于决定是否搜索与搜索关键词。

        返回 JSON 约定字段：
        - need_search: boolean
        - search_queries: string[]
        - sensitive: boolean
        - argument_plan: string
        - story_plan: string
        """

        role_name = self.philosopher_name
        core = str(self._role.get("core_philosophy") or "")
        classics = self._role.get("classic_works") or []
        classics_text = "；".join([str(x) for x in classics]) if isinstance(classics, list) else str(classics)
        stage_text = (stage or "").strip()
        # 攻辩/质询需要“短、硬、可回答”的问题与回应；为避免叙事拖长，本环节允许不讲故事。
        story_optional = stage_text == "攻辩/质询"
        story_policy = (
            "已触发，必须用故事表达，避免直接点名敏感对象。"
            if sensitive
            else ("未触发：本环节允许不讲故事；若使用故事，必须服务论点。" if story_optional else "未触发（但仍需故事服务论点）。")
        )

        prompt = f"""
你正在扮演哲学辩手：{role_name}（{self._side_label}）。
你的核心哲学立场：{core}
可引用的经典著作候选：{classics_text}

现在辩论主题：{motion_topic}
本方选题/立场：{side_topic}
当前环节：{stage}
本轮核心任务：{task_instruction}
当前轮次：第{round_num}轮
对方上一轮（结构化摘要）：{json.dumps(opponent_last or {}, ensure_ascii=False)}

敏感/故事策略：{story_policy}

请只输出一个 JSON 对象，字段如下：
{{
  "need_search": true/false,
  "search_queries": ["..."],
  "sensitive": true/false,
  "argument_plan": "一句话概括本轮论证主线（前提→推理→结论）",
  "story_plan": "一句话概括故事场景与寓意映射（故事必须服务论点）"
}}

要求：
1) search_queries 只给 0~3 条，必须是可直接用于搜索的短 query；
2) 本轮必须回应对方上一轮要点，并包含反驳点与建设性论点；
3) 输出必须是严格 JSON，不要添加任何解释文字。
        """.strip()

        text = self._invoke_text(prompt)
        try:
            return self._extract_json_with_retry(
                text=text,
                retry_prompt=prompt
                + "\n\n补充要求：\n- 只能输出 JSON 对象本体（以 { 开始，以 } 结束），不要输出任何多余字符；\n- 不要使用单引号；key 必须使用英文双引号；\n- 每个字段之间必须用逗号分隔；\n",
                json_skeleton='{"need_search": false, "search_queries": [], "sensitive": false, "argument_plan": "", "story_plan": ""}',
            )
        except Exception:
            return {
                "need_search": False,
                "search_queries": [],
                "sensitive": bool(sensitive),
                "argument_plan": "",
                "story_plan": "",
            }

    def _generate_speech(
        self,
        *,
        motion_topic: str,
        side_topic: str,
        stage: str,
        task_instruction: str,
        round_num: int,
        opponent_last: dict[str, str] | None,
        sensitive: bool,
        plan: dict[str, Any],
        search_snippets: list[dict[str, Any]],
        max_chars: int,
    ) -> DebaterOutput:
        """
        生成阶段：输出结构化 JSON（固定字段）。

        字段来自 config/debate_rules.yaml 的 output_schema.fields。
        """

        role_name = self.philosopher_name
        core = str(self._role.get("core_philosophy") or "")
        templates = self._role.get("story_templates") or []
        story_templates = "；".join([str(x) for x in templates]) if isinstance(templates, list) else str(templates)
        classics = self._role.get("classic_works") or []
        classics_text = "；".join([str(x) for x in classics]) if isinstance(classics, list) else str(classics)

        required_fields = (((self._rules.get("output_schema") or {}).get("fields")) or [])
        required_fields_text = "、".join([str(x) for x in required_fields]) if isinstance(required_fields, list) else str(required_fields)
        stage_text = (stage or "").strip()
        # 攻辩/质询聚焦短问短答与证据核验，本环节允许不讲故事（敏感触发时仍必须故事化隐喻）。
        story_optional = stage_text == "攻辩/质询"
        story_line5 = "若讲故事，故事必须服务论点，并在“故事寓意”中明确映射关系。" if story_optional else "故事必须服务论点，并在“故事寓意”中明确映射关系"
        story_rule = (
            "敏感内容已触发：禁止直接点名敏感对象/事件，用故事隐喻表达。"
            if sensitive
            else ("本环节允许不讲故事；若讲故事，必须服务论点并在“故事寓意”中映射。" if story_optional else "即使未触发敏感，也要故事化表达。")
        )

        prompt = f"""
你正在扮演哲学辩手：{role_name}（{self._side_label}）。
核心立场：{core}
经典著作候选：{classics_text}
偏好故事开头模板：{story_templates}

主题：{motion_topic}
本方选题/立场：{side_topic}
当前环节：{stage}
本轮核心任务：{task_instruction}
轮次：第{round_num}轮
对方上一轮（结构化摘要）：{json.dumps(opponent_last or {}, ensure_ascii=False)}

本轮计划：{json.dumps(plan, ensure_ascii=False)}
可用搜索材料（供你引用与概括，勿长篇粘贴）：{json.dumps(search_snippets, ensure_ascii=False)}

 输出要求：
 1) 必须包含字段：{required_fields_text}
 2) 必须直接回应对方上一轮要点（在“回应对方要点”字段体现）
 3) 必须包含至少1个反驳点与1个建设性论点
 4) 必须给出“哲学依据”（哲学概念/原理 + 经典著作引用）
5) {story_line5}
6) {story_rule}
7) 全文尽量控制在 {max_chars} 字以内（超出会被裁剪）

请只输出一个严格 JSON 对象，每个字段对应一个字符串值，不要输出任何解释文字或代码块。
""".strip()

        raw = self._invoke_text(prompt)
        obj = self._extract_speech_with_retry(
            text=raw,
            retry_prompt=prompt,
            required_fields=required_fields if isinstance(required_fields, list) else [],
        )

        content: dict[str, str] = {}
        for f in required_fields if isinstance(required_fields, list) else []:
            key = str(f)
            val = obj.get(key, "")
            content[key] = str(val).strip()

        return DebaterOutput(content=content, raw_text=raw)

    def _extract_json_with_retry(self, *, text: str, retry_prompt: str, json_skeleton: str | None = None) -> dict[str, Any]:
        last_err: Exception | None = None
        for _ in range(3):
            try:
                return extract_json_object(text)
            except Exception as e:
                last_err = e
                fix_prompt = (
                    retry_prompt
                    + "\n\n你刚才的输出 JSON 无法解析。\n"
                    + f"解析错误：{type(e).__name__}: {e}\n"
                    + "请直接输出修复后的严格 JSON 对象，不要输出任何解释、注释或代码块。"
                )
                text = self._invoke_text(fix_prompt)
        if json_skeleton:
            skeleton_prompt = (
                retry_prompt
                + "\n\n你多次未能输出可解析的 JSON。请严格按以下 JSON 骨架输出，并仅填充字段值：\n"
                + json_skeleton
                + "\n\n只输出 JSON 对象本体。"
            )
            text2 = self._invoke_text(skeleton_prompt)
            return extract_json_object(text2)
        raise ValueError(f"JSON 解析失败（已重试）：{last_err}") from last_err

    def _extract_speech_with_retry(
        self,
        *,
        text: str,
        retry_prompt: str,
        required_fields: list[Any],
    ) -> dict[str, Any]:
        keys = [str(x) for x in required_fields if str(x).strip()]
        json_skeleton = "{" + ", ".join([f"\"{k}\": \"\"" for k in keys]) + "}"
        last_err: Exception | None = None
        for _ in range(3):
            try:
                return extract_json_object(text)
            except Exception as e:
                last_err = e
                fix_prompt = (
                    retry_prompt
                    + "\n\n你刚才的输出 JSON 无法解析。\n"
                    + f"解析错误：{type(e).__name__}: {e}\n"
                    + "请直接输出修复后的严格 JSON 对象，不要输出任何解释、注释或代码块。"
                )
                text = self._invoke_text(fix_prompt)

        try:
            text2 = self._invoke_text(
                retry_prompt
                + "\n\n你多次未能输出可解析的 JSON。请严格按以下 JSON 骨架输出，并仅填充字段值：\n"
                + json_skeleton
                + "\n\n只输出 JSON 对象本体。"
            )
            return extract_json_object(text2)
        except Exception as e:
            last_err = e

        fallback = self._extract_fields_fallback(text, keys)
        if fallback:
            return fallback

        raise ValueError(f"JSON 解析失败（已重试）：{last_err}") from last_err

    def _extract_fields_fallback(self, text: str, keys: list[str]) -> dict[str, Any]:
        t = (text or "").strip()
        if not t:
            return {}
        if not keys:
            return {}

        out: dict[str, str] = {}
        for k in keys:
            out[k] = ""

        current: str | None = None
        for line in t.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                s = s.lstrip(">").strip()
            if s.startswith("-"):
                s = s.lstrip("-").strip()

            matched_key = None
            for k in keys:
                if s.startswith(f"**{k}**：") or s.startswith(f"{k}：") or s.startswith(f"{k}:"):
                    matched_key = k
                    break
            if matched_key:
                current = matched_key
                val = s.split("：", 1)[1].strip() if "：" in s else s.split(":", 1)[1].strip()
                out[current] = (out.get(current, "") + ("\n" if out.get(current) else "") + val).strip()
                continue

            if current:
                out[current] = (out.get(current, "") + "\n" + s).strip()

        return out

    def _invoke_text(self, prompt: str) -> str:
        """
        调用 LLM 并取回纯文本内容。

        兼容不同返回类型：
        - langchain message（.content）
        - 字符串
        """

        try:
            from langchain_core.messages import HumanMessage  # type: ignore
        except ModuleNotFoundError:
            # 极端情况下（依赖异常），退化为直接字符串调用
            resp = self._llm.invoke(prompt)
            return getattr(resp, "content", str(resp))

        resp = self._llm.invoke([HumanMessage(content=prompt)])
        return getattr(resp, "content", str(resp))
