import unittest


from debate_arena.agents.clerk import ClerkAgent


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = outputs
        self.calls = 0

    def invoke(self, _messages):
        out = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        return _FakeResp(out)


class TestClerkAgent(unittest.TestCase):
    def test_fidelity_retry(self):
        clerk_rules = {
            "clerk": {
                "prompt": {
                    "system": "",
                    "user_template": "{round_title}\n{pro_fields}\n{opp_fields}\n",
                }
            }
        }
        llm = _FakeLLM(outputs=["### 第1轮（书记员整理）\n缺失了B\n", "### 第1轮（书记员整理）\nA\nB\na\n"])
        agent = ClerkAgent(llm=llm, clerk_rules=clerk_rules, pro_role={}, opp_role={})

        out = agent.rewrite_round(
            topic="T",
            round_num=1,
            pro_name="甲",
            opp_name="乙",
            pro_fields_text="- 本轮立论：A\n- 反驳点：B",
            opp_fields_text="- 本轮立论：a",
            round_title="### 第1轮（书记员整理）",
        )
        self.assertIn("A", out.markdown)
        self.assertIn("B", out.markdown)
        self.assertGreaterEqual(llm.calls, 2)


if __name__ == "__main__":
    unittest.main()
