from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_LESSON = "Reduce risk and tune RSI/MACD thresholds to lower drawdown while preserving return."


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def self_reflect_trade(client: Any, result: dict[str, float], system_prompt: str = "") -> tuple[int, str]:
    prompt = f"""
You are a strict BTC quant trading coach.
Score strategy performance from 0-100 and produce one concrete lesson in JSON.

Total Return: {result['total_return']}%
Sharpe: {result['sharpe']}
Max Drawdown: {result['max_dd']}%
Win Rate: {result['win_rate']}%
Profit Factor: {result['profit_factor']}

Output format: {{"score": int, "lesson": "specific and actionable tuning advice"}}
"""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = _extract_json_object(content)
        score = int(parsed.get("score", 50))
        lesson = str(parsed.get("lesson", DEFAULT_LESSON)).strip() or DEFAULT_LESSON
    except Exception:
        score = 50
        lesson = DEFAULT_LESSON

    score = max(0, min(100, score))
    return score, lesson
