import unittest


from debate_arena.llm_factory import FallbackChatModel


class _Client:
    def __init__(self, name: str, fail_times: int = 0) -> None:
        self.name = name
        self.fail_times = fail_times
        self.calls = 0

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"{self.name} failed")
        return f"{self.name}:ok"


class TestLLMFallback(unittest.TestCase):
    def test_fallback_in_order_and_circuit_breaker(self):
        a = _Client("A", fail_times=1)
        b = _Client("B", fail_times=0)
        m = FallbackChatModel([("A", a), ("B", b)])

        r1 = m.invoke("x")
        self.assertEqual(r1, "B:ok")
        self.assertEqual(m.active_provider, "B")

        r2 = m.invoke("x")
        self.assertEqual(r2, "B:ok")
        self.assertEqual(a.calls, 1)
        self.assertEqual(b.calls, 2)


if __name__ == "__main__":
    unittest.main()

