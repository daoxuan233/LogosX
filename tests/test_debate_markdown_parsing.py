import unittest


from debate_arena.parsing.debate_markdown import parse_transcript


class TestDebateMarkdownParsing(unittest.TestCase):
    def test_parse_one_round(self):
        md = """# 【哲学辩论】T

## 📋 辩论基本信息
- **主席**：系统仲裁员（规则引擎驱动）
- **正方**：甲
- **反方**：乙
- **总轮次**：1轮
- **生成时间**：2026年01月14日 19:05:49

## ⚖️ 核心论题
T2

---

## 🎭 辩论实录
### 第1轮
**⏰ 主席提醒**：R
**🔵 正方**（甲）：
>
> **回应对方要点**：A
> **本轮立论**：B
>
> **哲学依据**：C
> **故事**：D
> **故事寓意**：E

**🔴 反方**（乙）：
>
> **回应对方要点**：a
> **本轮立论**：b
> **反驳点**：c
>
> **哲学依据**：d
> **故事**：e
> **故事寓意**：f
"""

        t = parse_transcript(md)
        self.assertEqual(t.topic, "T2")
        self.assertEqual(t.proponent_name, "甲")
        self.assertEqual(t.opponent_name, "乙")
        self.assertEqual(len(t.rounds), 1)
        r1 = t.rounds[0]
        self.assertEqual(r1.round_num, 1)
        self.assertEqual(r1.proponent.philosopher, "甲")
        self.assertEqual(r1.opponent.philosopher, "乙")
        self.assertEqual(r1.proponent.fields["回应对方要点"], "A")
        self.assertEqual(r1.proponent.fields["本轮立论"], "B")
        self.assertEqual(r1.opponent.fields["反驳点"], "c")


if __name__ == "__main__":
    unittest.main()

