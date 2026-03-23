import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from btc_self_improve_agent.tools import (
        compute_weekly_score,
        persist_trace,
        resample_features,
        run_backtest_simulation,
        run_weekly_backtest,
        write_backtest_report,
    )
except ModuleNotFoundError:  # optional runtime deps not installed in CI container
    _TOOLS_AVAILABLE = False
else:
    _TOOLS_AVAILABLE = True


@unittest.skipUnless(_TOOLS_AVAILABLE, "tool runtime dependencies are not installed")
class ToolsTest(unittest.TestCase):
    def setUp(self):
        # synthetic 15m bars for two days (fits into one ISO week)
        idx = pd.date_range("2024-01-02", periods=24 * 4 * 2, freq="15min", tz="UTC")
        base = pd.DataFrame(
            {
                "timestamp": idx,
                "Open": 100 + pd.Series(range(len(idx))) * 0.1,
                "High": 100.2 + pd.Series(range(len(idx))) * 0.1,
                "Low": 99.8 + pd.Series(range(len(idx))) * 0.1,
                "Close": 100.1 + pd.Series(range(len(idx))) * 0.1,
                "Volume": 1_000 + pd.Series(range(len(idx))) * 5,
            }
        )
        self.frames = resample_features(base)

    def test_resample_outputs_all_timeframes(self):
        for tf in ["15m", "1h", "4h", "1d"]:
            self.assertIn(tf, self.frames)
            self.assertFalse(self.frames[tf].empty)

    def test_weekly_backtest_and_score(self):
        strategy = {"entry_timeframe": "1h", "rsi_buy": 35, "rsi_sell": 65, "weight_resonance": 1.1}
        result = run_weekly_backtest(self.frames, strategy)
        metrics = result["metrics"]
        for key in ["total_return", "sharpe", "max_dd", "win_rate", "profit_factor", "costs"]:
            self.assertIn(key, metrics)

        score = compute_weekly_score(metrics)
        self.assertIsInstance(score, float)

    def test_multi_timeframe_backtest_simulation_accepts_resampled_frames(self):
        strategy = {"entry_timeframe": "auto", "rsi_buy": 35, "rsi_sell": 65, "news_weight": 0.4}
        result = run_backtest_simulation(self.frames, news=[{"sentiment": 0.2}], strategy=strategy)

        for key in ["total_return", "sharpe", "max_dd", "win_rate", "profit_factor", "trade_count", "entry_timeframe", "execution_timeframe"]:
            self.assertIn(key, result)
        self.assertEqual(result["entry_timeframe"], "auto")
        self.assertEqual(result["execution_timeframe"], "15m")

    def test_persist_trace_creates_file(self):
        metrics = {"total_return": 1.0, "sharpe": 0.5, "max_dd": 5.0, "win_rate": 55.0, "costs": 0.1}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = persist_trace("2024-W01", {"demo": True}, metrics, 42.0, trace_dir=tmpdir)
            self.assertTrue(os.path.exists(path))

    def test_backtest_report_contains_trade_details(self):
        indicators = {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "Close": [100.0, 102.0, 104.0, 103.0, 101.0],
            "RSI": [40.0, 25.0, 45.0, 75.0, 50.0],
            "MACD": [0.1, 1.2, 0.6, -1.0, -0.2],
            "MACD_signal": [0.2, 0.8, 0.7, -0.5, -0.1],
        }
        strategy = {"rsi_buy": 30, "rsi_sell": 70, "news_weight": 0.4}
        result = run_backtest_simulation(indicators, news=[{"sentiment": 0.5}], strategy=strategy)

        self.assertIn("trades", result)
        self.assertTrue(result["trades"])
        first_trade = result["trades"][0]
        for key in ["size", "entry_price", "exit_price", "entry_rationale", "exit_rationale"]:
            self.assertIn(key, first_trade)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_backtest_report(
                epoch=1,
                strategy=strategy,
                metrics=result,
                trades=result["trades"],
                report_dir=tmpdir,
                generated_at="2024-01-05T00:00:00+00:00",
            )
            self.assertTrue(os.path.exists(path))
            content = Path(path).read_text(encoding="utf-8")
            self.assertIn("# Backtest Report: Epoch 1", content)
            self.assertIn("Entry Rationale", content)
            self.assertIn("Exit Rationale", content)
            self.assertIn("Entry Price", content)
            self.assertIn("Exit Price", content)


if __name__ == "__main__":
    unittest.main()
