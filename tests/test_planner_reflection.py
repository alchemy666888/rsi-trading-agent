import unittest

from btc_self_improve_agent.planner import _extract_json_object as extract_plan_json
from btc_self_improve_agent.planner import _sanitize_strategy
from btc_self_improve_agent.planner import update_strategy_from_lesson
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

    def test_update_strategy_from_lesson_returns_change_log(self):
        strategy, change_log = update_strategy_from_lesson(
            {"rsi_buy": 30, "rsi_sell": 70, "max_position": 1.0},
            lesson="Drawdown too high after news shock, reduce risk.",
            monthly_metrics={"max_dd": 30.0, "win_rate": 42.0, "sharpe": 0.4},
            client=None,
        )
        self.assertIn("max_position", strategy)
        self.assertIsInstance(change_log, dict)
        self.assertIn("changes", change_log)


if __name__ == "__main__":
    unittest.main()
