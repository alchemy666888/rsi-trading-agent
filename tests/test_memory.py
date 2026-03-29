import tempfile
import unittest

from btc_self_improve_agent.memory import MemoryManager


class MemoryTest(unittest.TestCase):
    def test_store_and_get_lessons(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            memory = MemoryManager(session_id="s1", db_path=f.name)
            memory.store_strategy({"rsi_buy": 30}, {"total_return": 10}, 80, "raise rsi_buy to 32")
            memory.store_strategy({"rsi_buy": 20}, {"total_return": 1}, 40, "bad")
            lessons = memory.get_relevant_lessons(min_score=70)
            self.assertIn("raise rsi_buy to 32", lessons)
            self.assertNotIn("bad", lessons)
            memory.close()

    def test_store_and_get_monthly_lessons(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            memory = MemoryManager(session_id="s2", db_path=f.name)
            memory.store_monthly_strategy(
                month_id="2024-01",
                strategy={"rsi_buy": 30},
                metrics={"total_return": 1.2, "max_dd": 5.0},
                score=72.5,
                lesson="Use stricter trend filter",
                change_log={"changes": ["trend_filter_strength: 0.6 -> 0.65"]},
                validation_flags={"max_dd_breach": False},
            )
            lessons = memory.get_recent_monthly_lessons(limit=5)
            self.assertEqual(len(lessons), 1)
            self.assertEqual(lessons[0]["month_id"], "2024-01")
            self.assertIn("trend_filter_strength", lessons[0]["change_log"]["changes"][0])
            context = memory.get_monthly_context(top_n=3)
            self.assertIn("2024-01", context)
            memory.close()


if __name__ == "__main__":
    unittest.main()
