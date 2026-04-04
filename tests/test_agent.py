import unittest

from btc_self_improve_agent.agent import _extract_side_trade_counts, _should_apply_monthly_update


class AgentUpdateGateTest(unittest.TestCase):
    def test_extract_side_trade_counts(self):
        simulation = {
            "trades": [
                {"side": "Long"},
                {"side": "Short"},
                {"side": "Long"},
                {"side": "long"},
                {"side": "SHORT"},
            ]
        }
        long_count, short_count = _extract_side_trade_counts(simulation)
        self.assertEqual(long_count, 3)
        self.assertEqual(short_count, 2)

    def test_skip_update_when_trade_count_insufficient(self):
        simulation = {"trade_count": 2, "trades": [{"side": "Long"}, {"side": "Short"}]}
        allow, reasons = _should_apply_monthly_update(simulation, min_trade_count=5, min_trades_per_side=1)
        self.assertFalse(allow)
        self.assertIn("insufficient_trade_count<5", reasons)

    def test_skip_update_when_one_sided_distribution(self):
        simulation = {"trade_count": 8, "trades": [{"side": "Long"} for _ in range(8)]}
        allow, reasons = _should_apply_monthly_update(simulation, min_trade_count=5, min_trades_per_side=1)
        self.assertFalse(allow)
        self.assertIn("one_sided_trade_distribution", reasons)

    def test_allow_update_when_trade_evidence_is_balanced(self):
        simulation = {
            "trade_count": 10,
            "trades": [{"side": "Long"} for _ in range(5)] + [{"side": "Short"} for _ in range(5)],
        }
        allow, reasons = _should_apply_monthly_update(simulation, min_trade_count=5, min_trades_per_side=1)
        self.assertTrue(allow)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
