from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .memory import MemoryManager
from .observability import trace_span
from .planner import create_strategy_plan
from .reflection import self_reflect_trade
from .tools import execute_tool


class BTCSelfImprovingAgent:
    def __init__(self, api_key: str | None = None, session_id: str = "btc_sim"):
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        self.memory = MemoryManager(session_id=session_id)
        self.system_prompt = "You are a BTC trading strategy optimizer. Use past lessons to improve."

    def _update_system_prompt(self, lesson: str) -> None:
        self.system_prompt += f"\nNew lesson: {lesson}"

    def run(self, user_goal: str, epochs: int = 5, require_confirmation: bool = True) -> dict[str, Any]:
        best_return = float("-inf")
        best_result: dict[str, Any] = {}

        with trace_span("btc_simulation") as trace:
            for epoch in range(1, epochs + 1):
                context = self.memory.get_relevant_lessons()
                strategy = create_strategy_plan(self.client, user_goal, context, self.system_prompt)

                data = execute_tool({"name": "fetch_btc_data", "args": {"period": "2y"}}, require_confirmation=False)
                indicators = execute_tool(
                    {"name": "calculate_indicators", "args": {"data": data, "params": strategy}},
                    require_confirmation=False,
                )
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

                score, lesson = self_reflect_trade(self.client, backtest_result, self.system_prompt)

                if backtest_result.get("max_dd", 0) > 30:
                    score = max(0, score - 10)
                    lesson = f"{lesson} Reduce drawdown below 30% by lowering risk exposure."

                self.memory.store_strategy(strategy, backtest_result, score, lesson)

                trace["steps"].append(
                    {
                        "epoch": epoch,
                        "strategy": strategy,
                        "metrics": backtest_result,
                        "score": score,
                        "lesson": lesson,
                    }
                )

                if backtest_result["total_return"] > best_return:
                    best_return = backtest_result["total_return"]
                    best_result = {"strategy": strategy, "metrics": backtest_result, "score": score, "lesson": lesson}
                    self._update_system_prompt(lesson)

                print(f"Epoch {epoch}: Total Return {backtest_result['total_return']:.2f}% | Score {score}")

        self.memory.close()
        return best_result
