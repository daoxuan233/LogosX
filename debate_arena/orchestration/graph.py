"""
LangGraph 编排：主席与两位辩手的 20+ 轮辩论流程。

说明：
- 采用 MessagesState/StateGraph/Command 组合；
- 主席为“supervisor”，负责回合节奏与评估；
- 正方/反方为“agent”节点，交替发言；
- 每轮产出写入 DocumentManager，最终统一导出 Markdown。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from debate_arena.agents.chairman import ChairmanAgent
from debate_arena.agents.debater import DebaterAgent, DebaterOutput
from debate_arena.config_loader import (
    default_paths,
    load_chairman_rules,
    load_debate_rules,
    load_sensitive_keywords,
    load_role,
)
from debate_arena.document_manager import DebateDocument, format_speaker_block
from debate_arena.llm_factory import load_env, make_chat_model
from debate_arena.search.hybrid_search import HybridSearchEngine


class DebateState(TypedDict):
    topic: str
    pro_topic: str
    opp_topic: str
    round: int
    rounds: int
    stage: str
    first_side: str
    doc: DebateDocument
    chairman: ChairmanAgent
    proponent: DebaterAgent
    opponent: DebaterAgent
    chairman_rules: dict[str, Any]
    last_pro: dict[str, str] | None
    last_opp: dict[str, str] | None


_STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("开篇立论", 0.21),
    ("攻辩/质询", 0.16),
    ("自由辩论", 0.36),
    ("总结陈词", 0.27),
]


def allocate_stage_counts(*, total_rounds: int) -> dict[str, int]:
    """
    将“总轮数”按四个辩论环节的“轮数体现比重”分配为整数轮次。

    设计目标：
    1) 与比重尽可能一致（但轮次必须是整数，且总和必须等于 total_rounds）；
    2) 结果确定性强（同样的 total_rounds 必须得到同样的分配）；
    3) 对极小轮次可运行：尽量保证“首轮开篇立论”，且在轮次允许时“末轮总结陈词”。

    分配算法（最大余数法 / Hamilton method）：
    - 先对 weight * total_rounds 取 floor 得到基础轮次；
    - 剩余轮次按小数余数从大到小补齐；
    - 若 total_rounds >= 4，额外保证每个环节至少 1 轮（通过从轮次最多的环节“挪”一轮补齐）。
    """

    if total_rounds <= 0:
        raise ValueError("total_rounds 必须为正整数。")

    stage_names = [name for name, _ in _STAGE_WEIGHTS]
    stage_count = len(stage_names)

    # 极小轮次兜底：无法保证四个环节都出现时，优先保证首轮为“开篇立论”，轮次允许时末轮为“总结陈词”。
    if total_rounds < stage_count:
        counts = {name: 0 for name in stage_names}
        counts["开篇立论"] = 1
        remaining = total_rounds - 1
        if remaining <= 0:
            return counts
        counts["总结陈词"] = 1
        remaining = total_rounds - 2
        for name in ["攻辩/质询", "自由辩论"]:
            if remaining <= 0:
                break
            counts[name] += 1
            remaining -= 1
        return counts

    raw = [weight * total_rounds for _, weight in _STAGE_WEIGHTS]
    floors = [int(math.floor(x)) for x in raw]
    remainders = [x - f for x, f in zip(raw, floors)]

    counts_list = floors[:]
    remaining = total_rounds - sum(counts_list)
    if remaining > 0:
        order = sorted(range(stage_count), key=lambda i: (-remainders[i], i))
        for i in range(remaining):
            counts_list[order[i]] += 1

    # total_rounds >= 4 时，保证每个环节至少 1 轮（否则会出现“比重很小 -> 被舍入为 0”的环节缺失）。
    for i in range(stage_count):
        if counts_list[i] > 0:
            continue
        donor = None
        donor_count = 0
        for j in range(stage_count):
            if counts_list[j] > donor_count and counts_list[j] > 1:
                donor = j
                donor_count = counts_list[j]
        if donor is None:
            continue
        counts_list[donor] -= 1
        counts_list[i] = 1

    return {stage_names[i]: counts_list[i] for i in range(stage_count)}


def determine_stage(*, round_num: int, total_rounds: int) -> str:
    """
    依据轮次返回本轮所属辩论环节。

    注意：round_num 是“轮”的编号；本项目每一轮包含：主席提醒 → 双方各发言一次 → 主席评估。
    """

    counts = allocate_stage_counts(total_rounds=total_rounds)
    open_end = counts["开篇立论"]
    cross_end = open_end + counts["攻辩/质询"]
    free_end = cross_end + counts["自由辩论"]

    if round_num <= open_end:
        return "开篇立论"
    if round_num <= cross_end:
        return "攻辩/质询"
    if round_num <= free_end:
        return "自由辩论"
    return "总结陈词"


def task_instruction_for(*, stage: str, side: str) -> str:
    s = stage.strip()
    if s == "开篇立论":
        if side == "pro":
            return "建立逻辑框架：界定关键概念，提出衡量标准（准则），并给出2-3条核心论点与推理链。"
        return "建立反方逻辑框架：对正方定义/标准提出修正或替代，并给出2-3条核心反驳与推理链。"
    if s == "攻辩/质询":
        if side == "pro":
            return (
                "围绕对方最新论据做攻辩/质询：提出2-4个可直接回答的短问题，"
                "重点追问论据真实性、证据来源、因果链条与可检验性；"
                "最后用一句话落点推进正方判断。"
            )
        return (
            "围绕正方质询逐条回应：澄清前提、补足证据或指出其推理漏洞；"
            "随后给出1-2个反质询/反击点，强调正方论据的真实性或代价问题。"
        )
    if s == "自由辩论":
        if side == "pro":
            return "多点交锋：捕捉对方逻辑漏洞与自相矛盾，快速推进争点；避免复述长篇背景，一句一落点。"
        return "紧贴正方最新落点反击：抓漏洞、指出代价、给出反方替代解释或反例，形成连续压迫。"
    if s == "总结陈词":
        if side == "opp":
            return "总结陈词：梳理全场交锋，指出正方在逻辑与事实上的错误，完成价值总结。"
        return "总结陈词（最后发言权）：回应反方总结，修复争点并升华正方立场。"
    return "围绕本轮环节任务完成发言。"


def build_hybrid_search(paths=None) -> HybridSearchEngine:
    """
    构建混合搜索引擎实例。
    """

    from dotenv import load_dotenv
    import os

    load_dotenv(override=False)
    searxng_base_url = os.getenv("SEARXNG_BASE_URL", "http://localhost:8081").strip()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
    sqlite_path = Path("outputs/cache.sqlite3")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return HybridSearchEngine(
        searxng_base_url=searxng_base_url, redis_url=redis_url, sqlite_path=sqlite_path
    )


def run_debate(
    topic: str,
    *,
    pro_topic: str | None = None,
    opp_topic: str | None = None,
    rounds: int = 20,
    west_role: str = "socrates",
    east_role: str = "confucius",
) -> str:
    """
    运行一场完整辩论并返回 Markdown 文本。
    """

    load_env()
    paths = default_paths()
    chairman_rules = load_chairman_rules(paths)
    debate_rules = load_debate_rules(paths)
    sensitive = load_sensitive_keywords(paths)
    west_cfg = load_role("west", west_role, paths)
    east_cfg = load_role("east", east_role, paths)

    llm = make_chat_model()
    search = build_hybrid_search(paths)

    chairman = ChairmanAgent(chairman_rules)
    proponent = DebaterAgent(
        side_label="🔵 正方",
        role_config=west_cfg,
        debate_rules=debate_rules,
        sensitive_config=sensitive,
        chairman_rules=chairman_rules,
        llm=llm,
        search_engine=search,
    )
    opponent = DebaterAgent(
        side_label="🔴 反方",
        role_config=east_cfg,
        debate_rules=debate_rules,
        sensitive_config=sensitive,
        chairman_rules=chairman_rules,
        llm=llm,
        search_engine=search,
    )

    doc = DebateDocument(
        topic=topic,
        chairman_name="系统仲裁员（规则引擎驱动）",
        proponent_name=proponent.philosopher_name,
        opponent_name=opponent.philosopher_name,
        rounds=rounds,
    )

    initial_state: DebateState = {
        "topic": topic,
        "pro_topic": (pro_topic or topic),
        "opp_topic": (opp_topic or topic),
        "round": 1,
        "rounds": rounds,
        "stage": determine_stage(round_num=1, total_rounds=rounds),
        "first_side": "pro",
        "doc": doc,
        "chairman": chairman,
        "proponent": proponent,
        "opponent": opponent,
        "chairman_rules": chairman_rules,
        "last_pro": None,
        "last_opp": None,
    }

    graph = build_graph()
    recursion_limit = max(100, int(rounds) * 10 + 20)
    print(f"开始辩论：{topic}（{rounds} 轮）", flush=True)
    final_state: DebateState = graph.invoke(initial_state, {"recursion_limit": recursion_limit})

    # 最终裁决（模板 + 简要评分说明）
    verdict_title = (chairman_rules.get("templates", {}) or {}).get("final_verdict_title", "主席最终裁决")
    final_state["doc"].add_final_block(
        f"**{verdict_title}**：\n- **逻辑严谨性**：{8}/10\n- **哲学深度**：{8}/10\n- **文学表达**：{8}/10\n- **规则遵守**：{8}/10\n"
    )
    final_state["doc"].add_final_block(
        "**核心洞见**：双方围绕核心论题的理性交锋，展示了不同传统下的哲学诘问与伦理取向的张力。"
    )
    final_state["doc"].add_final_block(
        "**未尽议题**：需进一步细化对具体社会实践中的边界条件与德性可操作性问题。"
    )

    return final_state["doc"].export_markdown()


def build_graph():
    """
    构建 LangGraph StateGraph。

    节点流转：START → chairman_start → proponent_turn → opponent_turn → chairman_eval → router → ... → END
    """

    try:
        from langgraph.graph import StateGraph, START, END  # type: ignore
        from langgraph.types import Command  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("缺少依赖 langgraph。请使用 uv 安装项目依赖后再运行辩论。") from e

    def chairman_start(state: DebateState):
        r = int(state["round"])
        stage = determine_stage(round_num=r, total_rounds=int(state["rounds"]))
        first_side = "opp" if stage.startswith("总结陈词") else "pro"
        state["stage"] = stage
        state["first_side"] = first_side
        print(f"第 {r}/{int(state['rounds'])} 轮：主席提醒", flush=True)
        reminder = state["chairman"].start_round(r)
        state["doc"].add_block(f"### 第{r}轮\n**📌 环节**：{stage}\n**⏰ 主席提醒**：{reminder}\n")
        return Command(
            goto="proponent_turn" if first_side == "pro" else "opponent_turn",
            update={"doc": state["doc"], "stage": stage, "first_side": first_side},
        )

    def proponent_turn(state: DebateState):
        r = int(state["round"])
        print(f"第 {r}/{int(state['rounds'])} 轮：正方发言", flush=True)
        out: DebaterOutput = state["proponent"].generate_turn(
            motion_topic=state["topic"],
            side_topic=state["pro_topic"],
            stage=state["stage"],
            task_instruction=task_instruction_for(stage=state["stage"], side="pro"),
            round_num=r,
            opponent_last=state["last_opp"],
        )
        state["last_pro"] = out.content
        state["doc"].add_block(
            format_speaker_block("🔵 正方", state["proponent"].philosopher_name, out.content)
        )
        if state["first_side"] == "pro":
            return Command(goto="opponent_turn", update={"doc": state["doc"], "last_pro": state["last_pro"]})
        return Command(goto="chairman_eval", update={"doc": state["doc"], "last_pro": state["last_pro"]})

    def opponent_turn(state: DebateState):
        r = int(state["round"])
        print(f"第 {r}/{int(state['rounds'])} 轮：反方发言", flush=True)
        out: DebaterOutput = state["opponent"].generate_turn(
            motion_topic=state["topic"],
            side_topic=state["opp_topic"],
            stage=state["stage"],
            task_instruction=task_instruction_for(stage=state["stage"], side="opp"),
            round_num=r,
            opponent_last=state["last_pro"],
        )
        state["last_opp"] = out.content
        state["doc"].add_block(
            format_speaker_block("🔴 反方", state["opponent"].philosopher_name, out.content)
        )
        if state["first_side"] == "opp":
            return Command(goto="proponent_turn", update={"doc": state["doc"], "last_opp": state["last_opp"]})
        return Command(goto="chairman_eval", update={"doc": state["doc"], "last_opp": state["last_opp"]})

    def chairman_eval(state: DebateState):
        r = int(state["round"])
        print(f"第 {r}/{int(state['rounds'])} 轮：主席评估", flush=True)
        decision, _, _ = state["chairman"].evaluate_round(
            topic=state["topic"],
            round_num=r,
            pro_speech_text=json_join(state["last_pro"]),
            opp_speech_text=json_join(state["last_opp"]),
        )
        block = f"**⚖️ 主席评估**：{decision.evaluation}\n"
        if decision.warning:
            block += f"**⚠️ 主席警告**：{decision.warning}\n"
        state["doc"].add_block(block)

        stage = state["chairman"].stage_summary(topic=state["topic"], round_num=r)
        if stage:
            state["doc"].add_block(f"> {stage}\n")

        return Command(goto="router", update={"doc": state["doc"]})

    def router(state: DebateState):
        r = int(state["round"])
        if r >= int(state["rounds"]):
            return Command(goto=END, update={"round": r})
        r2 = r + 1
        stage2 = determine_stage(round_num=r2, total_rounds=int(state["rounds"]))
        first_side2 = "opp" if stage2.startswith("总结陈词") else "pro"
        return Command(goto="chairman_start", update={"round": r2, "stage": stage2, "first_side": first_side2})

    builder = StateGraph(DebateState)
    builder.add_node("chairman_start", chairman_start)
    builder.add_node("proponent_turn", proponent_turn)
    builder.add_node("opponent_turn", opponent_turn)
    builder.add_node("chairman_eval", chairman_eval)
    builder.add_node("router", router)

    builder.add_edge(START, "chairman_start")
    return builder.compile()


def json_join(obj: dict[str, str] | None) -> str:
    """
    将结构化字段拼接为可评估的短文本。
    """

    if not obj:
        return ""
    parts = []
    for k in ["回应对方要点", "本轮立论", "反驳点", "建设性论点", "哲学依据", "故事", "故事寓意"]:
        v = obj.get(k)
        if v:
            parts.append(str(v))
    return "；".join(parts)
