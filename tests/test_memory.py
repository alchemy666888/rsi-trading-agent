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


if __name__ == "__main__":
    unittest.main()
