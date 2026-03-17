import unittest

try:
    from btc_self_improve_agent.tools import _heuristic_sentiment, calculate_indicators, run_backtest_simulation
except ModuleNotFoundError:  # optional runtime deps not installed in CI container
    _TOOLS_AVAILABLE = False
else:
    _TOOLS_AVAILABLE = True


@unittest.skipUnless(_TOOLS_AVAILABLE, "tool runtime dependencies are not installed")
class ToolsTest(unittest.TestCase):
    def test_heuristic_sentiment_bounds(self):
        self.assertLessEqual(_heuristic_sentiment("surge rally bull"), 1.0)
        self.assertGreaterEqual(_heuristic_sentiment("ban hack crash"), -1.0)

    def test_indicator_and_backtest_pipeline(self):
        data = {
            "timestamp": [f"2024-01-{i:02d}T00:00:00+00:00" for i in range(1, 31)],
            "Open": [100 + i for i in range(30)],
            "High": [101 + i for i in range(30)],
            "Low": [99 + i for i in range(30)],
            "Close": [100 + i for i in range(30)],
            "Volume": [1000 + i * 10 for i in range(30)],
        }
        params = {"rsi_buy": 30, "rsi_sell": 70, "news_weight": 0.5}
        indicators = calculate_indicators(data, params)
        self.assertIn("RSI", indicators)
        result = run_backtest_simulation(indicators, [{"sentiment": 0.4}], params)
        for key in ["total_return", "sharpe", "max_dd", "win_rate", "profit_factor"]:
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
