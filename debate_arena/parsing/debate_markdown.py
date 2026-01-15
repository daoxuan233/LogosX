from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerTurn:
    side: str
    philosopher: str
    fields: dict[str, str]
    raw_markdown: str


@dataclass(frozen=True)
class DebateRound:
    round_num: int
    proponent: SpeakerTurn | None
    opponent: SpeakerTurn | None
    raw_markdown: str


@dataclass(frozen=True)
class DebateTranscript:
    topic: str
    proponent_name: str
    opponent_name: str
    rounds: list[DebateRound]


_ROUND_RE = re.compile(r"^### 第(\d+)轮\s*$", re.MULTILINE)
_SPEAKER_RE = re.compile(r"^\*\*.*?(正方|反方)\*\*（(.+?)）：\s*$")
_FIELD_START_RE = re.compile(r"^>\s*\*\*(.+?)\*\*：\s*(.*)\s*$")


def parse_transcript(markdown: str) -> DebateTranscript:
    topic = _parse_topic(markdown)
    proponent_name, opponent_name = _parse_sides(markdown)
    rounds = _parse_rounds(markdown)
    return DebateTranscript(topic=topic, proponent_name=proponent_name, opponent_name=opponent_name, rounds=rounds)


def _parse_topic(markdown: str) -> str:
    lines = markdown.splitlines()
    m = re.match(r"^#\s*【哲学辩论】\s*(.+?)\s*$", lines[0].strip()) if lines else None
    title_topic = (m.group(1).strip() if m else "").strip()

    core_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "## ⚖️ 核心论题":
            core_idx = i
            break
    if core_idx is not None:
        for j in range(core_idx + 1, min(core_idx + 6, len(lines))):
            cand = lines[j].strip()
            if cand and not cand.startswith("#") and not cand.startswith("-"):
                return cand

    return title_topic


def _parse_sides(markdown: str) -> tuple[str, str]:
    pro = ""
    opp = ""
    for line in markdown.splitlines():
        s = line.strip()
        if s.startswith("- **正方**："):
            pro = s.split("：", 1)[1].strip()
        elif s.startswith("- **反方**："):
            opp = s.split("：", 1)[1].strip()
        if pro and opp:
            break
    return pro, opp


def _parse_rounds(markdown: str) -> list[DebateRound]:
    matches = list(_ROUND_RE.finditer(markdown))
    if not matches:
        return []

    end_cut = markdown.find("## 📊 辩论质量评估")
    content = markdown if end_cut < 0 else markdown[:end_cut]

    matches = list(_ROUND_RE.finditer(content))
    rounds: list[DebateRound] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[start:end].rstrip()
        round_num = int(m.group(1))
        pro, opp = _parse_round_speakers(block)
        rounds.append(DebateRound(round_num=round_num, proponent=pro, opponent=opp, raw_markdown=block))
    return rounds


def _parse_round_speakers(round_block: str) -> tuple[SpeakerTurn | None, SpeakerTurn | None]:
    lines = round_block.splitlines()
    speaker_indices: list[int] = []
    speaker_meta: list[tuple[str, str]] = []
    for idx, line in enumerate(lines):
        m = _SPEAKER_RE.match(line.strip())
        if not m:
            continue
        side = m.group(1).strip()
        philosopher = m.group(2).strip()
        speaker_indices.append(idx)
        speaker_meta.append((side, philosopher))

    turns: dict[str, SpeakerTurn] = {}
    for si, (side, philosopher) in enumerate(speaker_meta):
        start_line = speaker_indices[si]
        end_line = speaker_indices[si + 1] if si + 1 < len(speaker_indices) else len(lines)
        raw = "\n".join(lines[start_line:end_line]).rstrip()
        fields = _extract_fields_from_speaker_block(lines[start_line:end_line])
        turns[side] = SpeakerTurn(side=side, philosopher=philosopher, fields=fields, raw_markdown=raw)

    return turns.get("正方"), turns.get("反方")


def _extract_fields_from_speaker_block(block_lines: list[str]) -> dict[str, str]:
    fields: dict[str, list[str]] = {}
    current_key: str | None = None

    for line in block_lines:
        s = line.rstrip()
        m = _FIELD_START_RE.match(s)
        if m:
            current_key = m.group(1).strip()
            v = m.group(2).strip()
            fields.setdefault(current_key, [])
            if v:
                fields[current_key].append(v)
            continue

        if current_key is None:
            continue

        if not s.lstrip().startswith(">"):
            continue

        payload = s.lstrip()[1:].lstrip()
        if payload == "":
            continue
        fields[current_key].append(payload)

    out: dict[str, str] = {}
    for k, parts in fields.items():
        text = "\n".join([p for p in parts if p.strip()]).strip()
        if text:
            out[k] = text
    return out


def format_fields_for_prompt(fields: dict[str, str]) -> str:
    order = ["回应对方要点", "本轮立论", "反驳点", "建设性论点", "哲学依据", "故事", "故事寓意"]
    lines: list[str] = []
    for k in order:
        v = (fields.get(k) or "").strip()
        if v:
            lines.append(f"- {k}：{v}")
    for k, v in fields.items():
        if k in set(order):
            continue
        vv = (v or "").strip()
        if vv:
            lines.append(f"- {k}：{vv}")
    return "\n".join(lines).strip()

