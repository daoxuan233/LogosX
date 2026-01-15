"""
规则工具：敏感内容检测、跑题检测、结构校验等。

主席“强规则 + 弱生成”的核心落在这里：
- 主席不生成辩手观点，但会对输出结构/跑题/敏感策略进行约束与警告；
- 规则参数来自 config/*.yaml，可热替换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OffTopicScore:
    """
    跑题检测分数（0~1）。

    score 越大表示越偏离。
    """

    score: float
    keyword_coverage: float
    similarity: float


def detect_sensitive(text: str, sensitive_config: dict[str, Any]) -> bool:
    """
    简单敏感检测：关键词命中即触发。

    后续可升级为语义分类模型；当前以 KISS 优先。
    """

    keywords_by_cat = (sensitive_config.get("keywords") or {}) if isinstance(sensitive_config, dict) else {}
    t = text.lower()
    for _, words in keywords_by_cat.items():
        if not isinstance(words, list):
            continue
        for w in words:
            if not w:
                continue
            if str(w).lower() in t:
                return True
    return False


def _keyword_coverage(topic: str, speech: str) -> float:
    """
    关键词覆盖率：topic 分词很难做通用，这里做简化。

    策略：
    - 抽取 topic 中引号包裹的短语（中英文引号/书名号）；
    - 粗切 topic（按空格/标点）得到长度>=2 的片段；
    - 对连续中文片段取 2/3-gram，增强对“换种说法但仍贴题”的鲁棒性；
    - 对 tokens 做长度加权覆盖，避免 2-gram 过多导致失真。
    """

    import re

    raw = topic.strip()
    if not raw:
        return 0.0

    quoted: list[str] = []
    for pattern in [r"'([^']+)'", r'"([^"]+)"', r"“([^”]+)”", r"「([^」]+)」", r"『([^』]+)』", r"《([^》]+)》"]:
        quoted.extend([m.strip() for m in re.findall(pattern, raw) if str(m).strip()])

    coarse = [t.strip() for t in re.split(r"[\s，。！？；：、,.!?;:]+", raw) if len(t.strip()) >= 2]

    cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", raw)
    ngrams: list[str] = []
    for span in cjk_spans:
        if len(span) <= 1:
            continue
        for n in (2, 3):
            if len(span) < n:
                continue
            ngrams.extend([span[i : i + n] for i in range(0, len(span) - n + 1)])

    topic_tokens = list({t for t in (quoted + coarse + ngrams) if len(t) >= 2})
    if not topic_tokens:
        return 0.0
    s = speech
    total_w = 0.0
    hit_w = 0.0
    for tok in topic_tokens:
        w = float(min(len(tok), 6))
        total_w += w
        if tok in s:
            hit_w += w
    return hit_w / max(total_w, 1.0)


def _cheap_similarity(a: str, b: str) -> float:
    """
    轻量相似度（0~1）：用于没有 embedding 依赖时的降级。

    说明：
    - 原先用 difflib.SequenceMatcher 做“字符序列相似度”，对短 topic vs 长 speech 会系统性偏低；
    - 这里改为字符 2-gram 的余弦相似度，更贴近“词面/短语重合”并对长度更鲁棒。
    """

    import math
    import re
    from collections import Counter

    def normalize(text: str) -> str:
        t = (text or "").lower()
        t = re.sub(r"[\s\r\n\t]+", " ", t)
        t = re.sub(r"[^\w\u4e00-\u9fff ]+", " ", t)
        t = re.sub(r"\s{2,}", " ", t).strip()
        return t

    def char_ngrams(text: str, n: int) -> list[str]:
        t = normalize(text).replace(" ", "")
        if len(t) < n:
            return [t] if t else []
        return [t[i : i + n] for i in range(0, len(t) - n + 1)]

    n = 2
    va = Counter(char_ngrams(a, n))
    vb = Counter(char_ngrams(b, n))
    if not va or not vb:
        return 0.0

    dot = sum(va[k] * vb.get(k, 0) for k in va.keys())
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = float(dot / (na * nb))
    return max(0.0, min(1.0, sim))


def compute_off_topic_score(
    *,
    topic: str,
    speech: str,
    threshold_cfg: dict[str, Any],
    similarity_hint: float | None = None,
) -> OffTopicScore:
    """
    综合跑题分：结合关键词覆盖率与相似度。

    similarity_hint:
    - 若上层可提供 embedding 相似度，可传入；
    - 否则自动使用 cheap similarity 降级。
    """

    coverage = _keyword_coverage(topic, speech)
    similarity = similarity_hint if similarity_hint is not None else _cheap_similarity(topic, speech)

    # 跑题度：覆盖越低、相似越低，跑题度越高
    score = 1.0 - (0.6 * similarity + 0.4 * coverage)
    score = max(0.0, min(1.0, score))

    return OffTopicScore(score=score, keyword_coverage=coverage, similarity=similarity)


def should_warn_off_topic(score: OffTopicScore, chairman_rules: dict[str, Any]) -> bool:
    """
    是否需要发出跑题警告。

    规则来自 config/chairman_rules.yaml。
    """

    cfg = chairman_rules.get("off_topic_detection") if isinstance(chairman_rules, dict) else {}
    warn_threshold = float(cfg.get("warn_threshold", 0.7))
    return score.score >= warn_threshold
