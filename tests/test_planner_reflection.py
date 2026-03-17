import unittest

from btc_self_improve_agent.planner import _extract_json_object as extract_plan_json
from btc_self_improve_agent.planner import _sanitize_strategy
from btc_self_improve_agent.reflection import _extract_json_object as extract_reflection_json


class PlannerReflectionTest(unittest.TestCase):
    def test_extract_json_from_fenced_block(self):
        payload = '```json\n{"score": 88, "lesson": "tighten risk"}\n```'
        data = extract_reflection_json(payload)
        self.assertEqual(data["score"], 88)

    def test_extract_json_from_noisy_text(self):
        payload = 'Result follows: {"rsi_buy": 25, "rsi_sell": 75} end.'
        data = extract_plan_json(payload)
        self.assertEqual(data["rsi_buy"], 25)

    def test_sanitize_strategy_bounds_and_order(self):
        strategy = _sanitize_strategy({"rsi_buy": 1, "rsi_sell": 150, "ma_short": 80, "ma_long": 20})
        self.assertGreaterEqual(strategy["rsi_buy"], 5)
        self.assertLessEqual(strategy["rsi_sell"], 95)
        self.assertLess(strategy["ma_short"], strategy["ma_long"])


if __name__ == "__main__":
    unittest.main()
