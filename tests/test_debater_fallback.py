import unittest


from debate_arena.agents.debater import DebaterAgent


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


class _FakeSearchEngine:
    def search(self, _q: str):
        class _Out:
            source = "none"
            payload = {"results": []}

        return _Out()


class TestDebaterFallback(unittest.TestCase):
    def test_speech_fallback_parse_without_braces(self):
        debate_rules = {"output_schema": {"fields": ["回应对方要点", "本轮立论"]}}
        chairman_rules = {"limits": {"max_chars_per_speech": 700}}
        role = {"role_name": "X", "core_philosophy": "Y", "classic_works": []}

        llm = _FakeLLM(
            outputs=[
                '{"need_search": false, "search_queries": [], "sensitive": false, "argument_plan": "", "story_plan": ""}',
                "**回应对方要点**：A\n**本轮立论**：B\n",
            ]
        )
        agent = DebaterAgent(
            side_label="🔵 正方",
            role_config=role,
            debate_rules=debate_rules,
            sensitive_config={"categories": {}},
            chairman_rules=chairman_rules,
            llm=llm,
            search_engine=_FakeSearchEngine(),
        )

        out = agent.generate_turn(
            motion_topic="T",
            side_topic="P",
            stage="立论阶段（开篇明义）",
            task_instruction="阐述背景，给出核心定义，提出衡量标准（准则），并陈述2-3个核心论点。",
            round_num=1,
            opponent_last=None,
        )
        self.assertEqual(out.content["回应对方要点"], "A")
        self.assertEqual(out.content["本轮立论"], "B")


if __name__ == "__main__":
    unittest.main()
