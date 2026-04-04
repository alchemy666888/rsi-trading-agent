from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from .config import DEFAULT_CONFIG
from .memory import MemoryManager
from .observability import trace_span, write_month_trace
from .planner import DEFAULT_STRATEGY, create_strategy_plan, update_strategy_from_lesson
from .reflection import self_reflect_month
from .tools import (
    compute_weekly_score,
    run_backtest_simulation,
    write_backtest_report,
    write_final_strategy_markdown,
    write_monthly_report,
)


def _resolve_model(model_override: str | None) -> str:
    env_model = model_override or os.getenv("CLAUDE_MODEL") or os.getenv("ANTHROPIC_MODEL")
    if env_model:
        return env_model
    return "claude-3-5-sonnet-20240620"


def _extract_side_trade_counts(simulation: dict[str, Any]) -> tuple[int, int]:
    long_count = 0
    short_count = 0
    for trade in simulation.get("trades", []) or []:
        side = str(trade.get("side", "")).lower()
        if side.startswith("long"):
            long_count += 1
        elif side.startswith("short"):
            short_count += 1
    return long_count, short_count


def _should_apply_monthly_update(
    simulation: dict[str, Any],
    *,
    recent_history: list[dict[str, Any]] | None = None,
    min_trade_count: int = 12,
    min_active_months: int = 1,
    min_trades_per_side: int = 0,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    history = [item for item in (recent_history or []) if isinstance(item, dict)]
    review_window = history + [simulation]
    aggregate_trade_count = sum(int(item.get("trade_count", 0) or 0) for item in review_window)
    active_months = sum(1 for item in review_window if int(item.get("trade_count", 0) or 0) >= 3)
    if active_months < min_active_months:
        reasons.append(f"insufficient_active_months<{min_active_months}")
    if aggregate_trade_count < min_trade_count:
        reasons.append(f"insufficient_trade_count<{min_trade_count}")

    long_count = 0
    short_count = 0
    for item in review_window:
        item_long_count, item_short_count = _extract_side_trade_counts(item)
        long_count += item_long_count
        short_count += item_short_count
    if min_trades_per_side > 0 and aggregate_trade_count >= min_trade_count and min(long_count, short_count) < min_trades_per_side:
        reasons.append("one_sided_trade_distribution")

    return len(reasons) == 0, reasons


class BTCSelfImprovingAgent:
    def __init__(self, api_key: str | None = None, session_id: str = "btc_sim", model: str | None = None):
        load_dotenv()
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.client = Anthropic(api_key=key)
        self.model = _resolve_model(model)
        self.memory = MemoryManager(session_id=session_id)
        self.session_id = session_id
        self.config = DEFAULT_CONFIG
        self.system_prompt = (
            "You are a BTC trading strategy optimizer for a historical monthly walk-forward system. "
            "Use 4h and 1d only for trend/regime context. Use 15m and 1h only for entries/exits. "
            "Use sentiment and news impact timing as risk-regime modifiers, not standalone triggers."
        )

    def run(
        self,
        user_goal: str,
        *,
        market_csv_paths: dict[str, str],
        news_json_path: str,
        epochs: int = 5,
        require_confirmation: bool = False,
    ) -> dict[str, Any]:
        del epochs
        del require_confirmation

        market_frames = self._load_market_frames(market_csv_paths)
        news_records = self._load_news(news_json_path)
        monthly_ids = self.config.month_ids

        context = self.memory.get_monthly_context(top_n=6)
        strategy = create_strategy_plan(self.client, user_goal, context, self.system_prompt, model=self.model)
        if not strategy:
            strategy = dict(DEFAULT_STRATEGY)

        monthly_results: list[dict[str, Any]] = []
        with trace_span("btc_monthly_walkforward") as trace:
            for month_id in monthly_ids:
                month_start, month_end = self._month_bounds(month_id)
                month_frames = self._slice_month_frames(market_frames, month_id)
                if not month_frames or any(frame.empty for frame in month_frames.values()):
                    continue
                month_news = self._filter_news_for_month(news_records, month_id)

                simulation = run_backtest_simulation(
                    month_frames,
                    news=month_news,
                    strategy=strategy,
                    evaluation_start=month_start,
                    # Include bars that close exactly at month boundary while still excluding next-month opens.
                    evaluation_end=month_end + timedelta(hours=1),
                )
                score = compute_weekly_score(simulation, weights={"sharpe": 25, "win_rate": 25, "max_dd": 30, "costs": 20})
                reflection = self_reflect_month(self.client, simulation, system_prompt=self.system_prompt, model=self.model)
                lesson = str(reflection.get("lesson", ""))
                combined_score = (float(reflection.get("score", 50)) + score) / 2.0

                can_update, gating_reasons = _should_apply_monthly_update(
                    simulation,
                    recent_history=[item["metrics"] for item in monthly_results[-2:]],
                    min_trade_count=12,
                    min_active_months=2,
                    min_trades_per_side=0,
                )
                if can_update:
                    updated_strategy, change_log = update_strategy_from_lesson(
                        strategy,
                        lesson=lesson,
                        monthly_metrics=simulation,
                        client=self.client,
                        model=self.model,
                    )
                else:
                    updated_strategy = dict(strategy)
                    change_log = {
                        "summary": "Skipped strategy parameter update due insufficient monthly evidence.",
                        "changes": [],
                        "gating_reasons": gating_reasons,
                    }
                validation_flags = {
                    "max_dd_breach": float(simulation.get("max_dd", 0.0)) > 25.0,
                    "low_win_rate": float(simulation.get("win_rate", 0.0)) < 40.0,
                    "low_trade_count": int(simulation.get("trade_count", 0)) < 3,
                    "update_skipped": not can_update,
                    "update_gating_reasons": gating_reasons,
                }

                monthly_report_path = write_monthly_report(
                    month_id=month_id,
                    strategy=dict(strategy),
                    metrics=simulation,
                    lesson=lesson,
                )
                month_trace_path = write_month_trace(
                    month_id=month_id,
                    payload={
                        "strategy_before": dict(strategy),
                        "strategy_after": dict(updated_strategy),
                        "metrics": simulation,
                        "score": combined_score,
                        "lesson": lesson,
                        "change_log": change_log,
                        "validation_flags": validation_flags,
                        "report_path": monthly_report_path,
                    },
                )

                self.memory.store_monthly_strategy(
                    month_id=month_id,
                    strategy=dict(strategy),
                    metrics=simulation,
                    score=combined_score,
                    lesson=lesson,
                    change_log=change_log,
                    validation_flags=validation_flags,
                )
                self.memory.store_strategy(strategy, simulation, int(max(0, min(100, combined_score))), lesson)

                monthly_results.append(
                    {
                        "month_id": month_id,
                        "score": combined_score,
                        "lesson": lesson,
                        "strategy_before": dict(strategy),
                        "strategy_after": dict(updated_strategy),
                        "metrics": simulation,
                        "report_path": monthly_report_path,
                        "trace_path": month_trace_path,
                    }
                )
                strategy = updated_strategy
                trace["steps"].append(monthly_results[-1])

        final_strategy_path = write_final_strategy_markdown(strategy=strategy, monthly_history=monthly_results)
        final_backtest = run_backtest_simulation(market_frames, news=news_records, strategy=strategy)
        final_report_path = write_backtest_report(
            epoch=999,
            strategy=strategy,
            metrics=final_backtest,
            trades=final_backtest.get("trades", []),
            title="Final Frozen Backtest Report",
        )
        stable_report_path = self._write_stable_final_backtest(final_report_path=final_report_path, metrics=final_backtest)
        result = {
            "final_strategy": strategy,
            "final_strategy_markdown_path": final_strategy_path,
            "final_backtest_report_path": stable_report_path,
            "generated_backtest_path": final_report_path,
            "final_metrics": final_backtest,
            "monthly_results": monthly_results,
        }
        self.memory.close()
        return result

    def _load_market_frames(self, csv_paths: dict[str, str]) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for tf in self.config.timeframes:
            path = csv_paths.get(tf)
            if not path:
                raise ValueError(f"Missing market csv path for timeframe: {tf}")
            df = pd.read_csv(path)
            if "timestamp" not in df.columns:
                raise ValueError(f"{path} missing timestamp column")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df.set_index("timestamp", inplace=True)
            frames[tf] = df.sort_index()
        return frames

    def _load_news(self, news_json_path: str) -> list[dict[str, Any]]:
        path = Path(news_json_path)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _month_bounds(self, month_id: str) -> tuple[datetime, datetime]:
        month_start = datetime.strptime(month_id + "-01", "%Y-%m-%d").replace(tzinfo=UTC)
        if month_start.month == 12:
            month_end = datetime(month_start.year + 1, 1, 1, tzinfo=UTC)
        else:
            month_end = datetime(month_start.year, month_start.month + 1, 1, tzinfo=UTC)
        return month_start, month_end

    def _slice_month_frames(
        self,
        frames: dict[str, pd.DataFrame],
        month_id: str,
        *,
        warmup_days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        month_start, month_end = self._month_bounds(month_id)
        window_start = month_start - timedelta(days=warmup_days)
        sliced: dict[str, pd.DataFrame] = {}
        for tf, frame in frames.items():
            sliced[tf] = frame[(frame.index >= window_start) & (frame.index < month_end)].copy()
        return sliced

    def _filter_news_for_month(
        self,
        news_records: list[dict[str, Any]],
        month_id: str,
        *,
        lookback_days: int = 7,
    ) -> list[dict[str, Any]]:
        month_start, month_end = self._month_bounds(month_id)
        window_start = month_start - timedelta(days=lookback_days)
        rows: list[dict[str, Any]] = []
        for item in news_records:
            published = item.get("published_at")
            if not published:
                continue
            try:
                dt = datetime.fromisoformat(str(published).replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                continue
            if window_start <= dt < month_end:
                rows.append(item)
        return rows

    def _write_stable_final_backtest(self, *, final_report_path: str, metrics: dict[str, Any]) -> str:
        backtest_dir = Path(__file__).resolve().parent.parent / "backtest"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        stable_path = backtest_dir / "final_strategy_backtest_report.md"
        source = Path(final_report_path)
        if source.exists():
            stable_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            fallback = [
                "# Final Frozen Backtest Report",
                "",
                f"- Generated At: {datetime.now(UTC).isoformat()}",
                f"- Total Return: {metrics.get('total_return', 'N/A')}",
            ]
            stable_path.write_text("\n".join(fallback) + "\n", encoding="utf-8")
        return str(stable_path)
