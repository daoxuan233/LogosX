"""
命令行入口（最小骨架）。

后续会逐步补齐：
- 读取 .env 与 YAML 配置
- 运行 20+ 轮辩论并输出 Markdown
"""

from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path

from debate_arena.orchestration.graph import run_debate

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="debate-arena", add_help=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="运行一场 20+ 轮哲学辩论并输出 Markdown")
    run_parser.add_argument("--motion", default="", help="本次议题/总辩题（中文）。未提供会在启动时询问")
    run_parser.add_argument("--topic", default="", help="兼容参数：等同于 --motion")
    run_parser.add_argument("--pro-topic", default="", help="正方议题/立场（中文）。未提供会在启动时询问")
    run_parser.add_argument("--opp-topic", default="", help="反方议题/立场（中文）。未提供会在启动时询问")
    run_parser.add_argument("--rounds", type=int, default=20, help="轮次数（默认 20）")
    run_parser.add_argument("--west", default="", help="西方哲学家角色文件名（不含扩展名，可选）")
    run_parser.add_argument("--east", default="", help="东方哲学家角色文件名（不含扩展名，可选）")
    run_parser.add_argument("--output", default="", help="输出 Markdown 路径（可选，默认 outputs/ 下自动命名）")
    run_parser.add_argument("--clerk-output", default="", help="书记员版输出路径（可选，默认同目录追加 _书记员版）")
    clerk_group = run_parser.add_mutually_exclusive_group()
    clerk_group.add_argument("--clerk", dest="clerk", action="store_true", help="生成书记员整理版（默认开启）")
    clerk_group.add_argument("--no-clerk", dest="clerk", action="store_false", help="不生成书记员整理版")
    run_parser.set_defaults(clerk=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            motion = (getattr(args, "motion", "") or "").strip() or (getattr(args, "topic", "") or "").strip()
            pro_topic = (getattr(args, "pro_topic", "") or "").strip()
            opp_topic = (getattr(args, "opp_topic", "") or "").strip()
            if not motion:
                motion = input("本次议题是什么？（总辩题）\n> ").strip()
            if not pro_topic:
                pro_topic = input("正方议题是什么？（正方坚持的立场/命题）\n> ").strip()
            if not opp_topic:
                opp_topic = input("反方议题是什么？（反方坚持的立场/命题）\n> ").strip()

            west = args.west or "socrates"
            east = args.east or "confucius"
            md = run_debate(
                motion,
                pro_topic=pro_topic,
                opp_topic=opp_topic,
                rounds=args.rounds,
                west_role=west,
                east_role=east,
            )
            out_dir = Path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.output:
                out_path = Path(args.output)
            else:
                from datetime import datetime

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"辩论_{ts}.md"
            out_path.write_text(md, encoding="utf-8")
            print(f"已生成辩论文档：{out_path.as_posix()}")
            if bool(getattr(args, "clerk", True)):
                clerk_out_path = _generate_clerk_version(
                    debate_md_path=out_path,
                    topic=motion,
                    west_role=west,
                    east_role=east,
                    clerk_output=args.clerk_output,
                )
                print(f"已生成书记员版：{clerk_out_path.as_posix()}")
            sys.exit(0)
        except Exception as e:
            print(f"运行失败：{e}", file=sys.stderr)
            if os.getenv("DEBATE_ARENA_DEBUG", "").strip() in {"1", "true", "True"}:
                import traceback

                traceback.print_exc()
            print("提示：若报错为鉴权/缺少依赖，请检查 uv 依赖安装与 .env 配置。", file=sys.stderr)
            sys.exit(1)

    parser.print_help()
    sys.exit(2)


def _generate_clerk_version(
    *,
    debate_md_path: Path,
    topic: str,
    west_role: str,
    east_role: str,
    clerk_output: str,
) -> Path:
    from debate_arena.agents.clerk import ClerkAgent
    from debate_arena.config_loader import load_clerk_rules, load_role
    from debate_arena.llm_factory import load_env, make_chat_model
    from debate_arena.parsing.debate_markdown import format_fields_for_prompt, parse_transcript

    load_env()
    llm = make_chat_model()

    clerk_rules = load_clerk_rules()
    pro_role = load_role("west", west_role)
    opp_role = load_role("east", east_role)

    md = debate_md_path.read_text(encoding="utf-8")
    transcript = parse_transcript(md)

    clerk = (clerk_rules.get("clerk") or {}) if isinstance(clerk_rules, dict) else {}
    output_cfg = (clerk.get("output") or {}) if isinstance(clerk, dict) else {}
    title_tmpl = str(output_cfg.get("round_title_template") or "### 第{round_num}轮（书记员整理）")

    agent = ClerkAgent(llm=llm, clerk_rules=clerk_rules, pro_role=pro_role, opp_role=opp_role)

    lines: list[str] = []
    lines.append(f"# 【书记员整理】{transcript.topic or topic}")
    lines.append("")
    lines.append("## 🎙️ 辩论整理实录")
    lines.append("")

    for r in transcript.rounds:
        if r.proponent is None or r.opponent is None:
            continue
        round_title = title_tmpl.format(round_num=r.round_num)
        pro_fields_text = format_fields_for_prompt(r.proponent.fields)
        opp_fields_text = format_fields_for_prompt(r.opponent.fields)
        out = agent.rewrite_round(
            topic=transcript.topic or topic,
            round_num=r.round_num,
            pro_name=r.proponent.philosopher or transcript.proponent_name,
            opp_name=r.opponent.philosopher or transcript.opponent_name,
            pro_fields_text=pro_fields_text,
            opp_fields_text=opp_fields_text,
            round_title=round_title,
        )
        lines.append(out.markdown.rstrip())
        lines.append("")

    clerk_md = "\n".join(lines).rstrip() + "\n"
    if clerk_output:
        out_path = Path(clerk_output)
    else:
        out_path = debate_md_path.with_name(debate_md_path.stem + "_书记员版.md")
    out_path.write_text(clerk_md, encoding="utf-8")
    return out_path
