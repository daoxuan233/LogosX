import unittest


from debate_arena.document_manager import format_speaker_block
from debate_arena.orchestration.graph import allocate_stage_counts, determine_stage, task_instruction_for


class TestDebateStagesAndRendering(unittest.TestCase):
    def test_determine_stage_order(self):
        total = 6
        self.assertEqual(determine_stage(round_num=1, total_rounds=total), "开篇立论")
        self.assertEqual(determine_stage(round_num=2, total_rounds=total), "攻辩/质询")
        self.assertEqual(determine_stage(round_num=3, total_rounds=total), "自由辩论")
        self.assertEqual(determine_stage(round_num=4, total_rounds=total), "自由辩论")
        self.assertEqual(determine_stage(round_num=5, total_rounds=total), "总结陈词")
        self.assertEqual(determine_stage(round_num=6, total_rounds=total), "总结陈词")

    def test_allocate_stage_counts_sums_to_total(self):
        for total in range(4, 41):
            counts = allocate_stage_counts(total_rounds=total)
            self.assertEqual(sum(counts.values()), total)
            self.assertTrue(all(v >= 1 for v in counts.values()))

    def test_allocate_stage_counts_20_rounds(self):
        counts = allocate_stage_counts(total_rounds=20)
        self.assertEqual(counts["开篇立论"], 4)
        self.assertEqual(counts["攻辩/质询"], 3)
        self.assertEqual(counts["自由辩论"], 7)
        self.assertEqual(counts["总结陈词"], 6)

    def test_task_instruction_for_summary(self):
        stage = "总结陈词"
        self.assertIn("指出正方", task_instruction_for(stage=stage, side="opp"))
        self.assertIn("最后发言权", task_instruction_for(stage=stage, side="pro"))

    def test_document_renders_extra_fields(self):
        md = format_speaker_block(
            "🔵 正方",
            "甲",
            {
                "回应对方要点": "Q1：你是否承认…？",
                "质询问题": "Q2：如果如此，是否意味着…？",
                "哲学依据": "以德波的景观概念为框架。",
            },
        )
        self.assertIn("**质询问题**：Q2：如果如此，是否意味着…？", md)


if __name__ == "__main__":
    unittest.main()
