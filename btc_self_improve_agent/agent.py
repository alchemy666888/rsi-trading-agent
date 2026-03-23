from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from .memory import MemoryManager
from .observability import trace_span
from .planner import create_strategy_plan
from .reflection import self_reflect_trade
from .tools import execute_tool, write_backtest_report


def _resolve_model(model_override: str | None) -> str:
    """Pick a default Anthropic model, allowing env overrides."""

    # User-configurable env vars
    env_model = model_override or os.getenv("CLAUDE_MODEL") or os.getenv("ANTHROPIC_MODEL")
    if env_model:
        return env_model

    # Stable fallbacks (newest first). Adjust as Anthropic updates versions.
    fallbacks = [
        "claude-3-5-sonnet-20240620",
        "claude-3-sonnet-20240229",
    ]
    return fallbacks[0]


class BTCSelfImprovingAgent:
    def __init__(self, api_key: str | None = None, session_id: str = "btc_sim", model: str | None = None):
        load_dotenv()
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.client = Anthropic(api_key=key)
        self.model = _resolve_model(model)
        self.memory = MemoryManager(session_id=session_id)
        self.system_prompt = (
            "You are a BTC trading strategy optimizer. Use past lessons to improve. "
            "Prefer multi-timeframe strategies that use 15m or 1h for entries, with 4h and 1d used as the trend "
            "and confirmation layer. Treat news as a regime filter rather than a standalone trigger."
        )

    def _update_system_prompt(self, lesson: str) -> None:
        self.system_prompt += f"\nNew lesson: {lesson}"

    def run(self, user_goal: str, epochs: int = 5, require_confirmation: bool = False) -> dict[str, Any]:
        best_return = float("-inf")
        best_result: dict[str, Any] = {}

        with trace_span("btc_simulation") as trace:
            for epoch in range(1, epochs + 1):
                context = self.memory.get_relevant_lessons()
                strategy = create_strategy_plan(self.client, user_goal, context, self.system_prompt, model=self.model)

                data = execute_tool(
                    {"name": "fetch_btc_data", "args": {"timeframe": "15m", "period": "2y"}},
                    require_confirmation=False,
                )
                indicators = execute_tool({"name": "resample_features", "args": {"raw": data}}, require_confirmation=False)
                news_sentiment = execute_tool({"name": "fetch_btc_news", "args": {}}, require_confirmation=False)
                backtest_result = execute_tool(
                    {
                        "name": "run_backtest_simulation",
                        "args": {"indicators": indicators, "news": news_sentiment, "strategy": strategy},
                    },
                    require_confirmation=require_confirmation,
                )

                if "error" in backtest_result:
                    trace["steps"].append({"epoch": epoch, "error": backtest_result["error"]})
                    break

                score, lesson = self_reflect_trade(self.client, backtest_result, self.system_prompt, model=self.model)

                if backtest_result.get("max_dd", 0) > 30:
                    score = max(0, score - 10)
                    lesson = f"{lesson} Reduce drawdown below 30% by lowering risk exposure."

                report_path = write_backtest_report(
                    epoch=epoch,
                    strategy=strategy,
                    metrics=backtest_result,
                    trades=backtest_result.get("trades", []),
                )

                self.memory.store_strategy(strategy, backtest_result, score, lesson)

                trace["steps"].append(
                    {
                        "epoch": epoch,
                        "strategy": strategy,
                        "metrics": backtest_result,
                        "score": score,
                        "lesson": lesson,
                        "report_path": report_path,
                    }
                )

                if backtest_result["total_return"] > best_return:
                    best_return = backtest_result["total_return"]
                    best_result = {
                        "strategy": strategy,
                        "metrics": backtest_result,
                        "score": score,
                        "lesson": lesson,
                        "report_path": report_path,
                    }
                    self._update_system_prompt(lesson)

                print(f"Epoch {epoch}: Total Return {backtest_result['total_return']:.2f}% | Score {score}")

        self.memory.close()
        return best_result
