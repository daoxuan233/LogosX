from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClerkRoundOutput:
    round_num: int
    markdown: str
    raw_text: str


class ClerkAgent:
    def __init__(
        self,
        *,
        llm,
        clerk_rules: dict[str, Any],
        pro_role: dict[str, Any],
        opp_role: dict[str, Any],
    ) -> None:
        self._llm = llm
        self._rules = clerk_rules
        self._pro_role = pro_role
        self._opp_role = opp_role

    def rewrite_round(
        self,
        *,
        topic: str,
        round_num: int,
        pro_name: str,
        opp_name: str,
        pro_fields_text: str,
        opp_fields_text: str,
        round_title: str,
    ) -> ClerkRoundOutput:
        clerk = (self._rules.get("clerk") or {}) if isinstance(self._rules, dict) else {}
        prompt = (clerk.get("prompt") or {}) if isinstance(clerk, dict) else {}
        system_prompt = str(prompt.get("system") or "")
        user_template = str(prompt.get("user_template") or "")

        pro_style = self._format_role_style(self._pro_role)
        opp_style = self._format_role_style(self._opp_role)

        user_prompt = user_template.format(
            topic=topic,
            round_num=round_num,
            pro_name=pro_name,
            opp_name=opp_name,
            pro_fields=pro_fields_text,
            opp_fields=opp_fields_text,
            pro_style=pro_style,
            opp_style=opp_style,
            round_title=round_title,
        )

        text = self._invoke_text(system_prompt=system_prompt, user_prompt=user_prompt)
        missing = self._find_missing_coverage(
            output=text,
            pro_fields_text=pro_fields_text,
            opp_fields_text=opp_fields_text,
        )
        if missing:
            fix_user_prompt = (
                user_prompt
                + "\n\n【保真校验未通过：以下片段必须逐字保留并出现在输出中】\n"
                + "\n".join([f"- {m}" for m in missing[:20]])
                + "\n\n请在不改变论点内容的前提下，补齐以上片段，重新输出完整结果。"
            )
            text2 = self._invoke_text(system_prompt=system_prompt, user_prompt=fix_user_prompt)
            missing2 = self._find_missing_coverage(
                output=text2,
                pro_fields_text=pro_fields_text,
                opp_fields_text=opp_fields_text,
            )
            text = text2 if not missing2 else text

        return ClerkRoundOutput(round_num=round_num, markdown=text, raw_text=text)

    def _format_role_style(self, role: dict[str, Any]) -> str:
        if not isinstance(role, dict):
            return ""
        keys = [
            "role_name",
            "tradition",
            "core_philosophy",
            "debate_style",
            "classic_works",
            "forbidden_topics",
            "story_templates",
        ]
        parts: list[str] = []
        for k in keys:
            v = role.get(k)
            if v is None:
                continue
            parts.append(f"{k}: {v}")
        return "\n".join(parts).strip()

    def _invoke_text(self, *, system_prompt: str, user_prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = self._llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = getattr(resp, "content", resp)
        return str(content or "").strip()

    def _find_missing_coverage(self, *, output: str, pro_fields_text: str, opp_fields_text: str) -> list[str]:
        missing: list[str] = []
        out_norm = self._normalize(output)
        for source in [pro_fields_text, opp_fields_text]:
            for line in (source or "").splitlines():
                s = line.strip()
                if not s.startswith("-"):
                    continue
                piece = s.lstrip("-").strip()
                if not piece:
                    continue
                if self._covered(piece, out_norm):
                    continue
                missing.append(piece)
        return missing

    def _covered(self, piece: str, out_norm: str) -> bool:
        text = piece.split("：", 1)[1].strip() if "：" in piece else piece
        norm = self._normalize(text)
        if not norm:
            return True
        if len(norm) <= 40:
            return norm in out_norm
        head = norm[:60]
        tail = norm[-60:]
        mid = norm[len(norm) // 2 - 30 : len(norm) // 2 + 30]
        return (head in out_norm) or (mid in out_norm) or (tail in out_norm)

    def _normalize(self, s: str) -> str:
        return "".join(s.split())
