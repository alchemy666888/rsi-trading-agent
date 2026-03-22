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


def _extract_text_content(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part)


def self_reflect_trade(
    client: Any,
    result: dict[str, float],
    system_prompt: str = "",
    model: str = "claude-3-5-sonnet-20240620",
) -> tuple[int, str]:
    prompt = f"""
You are a strict BTC quant trading coach.
Score strategy performance from 0-100 and produce one concrete lesson in JSON.

Total Return: {result['total_return']}%
Sharpe: {result['sharpe']}
Max Drawdown: {result['max_dd']}%
Win Rate: {result['win_rate']}%
Profit Factor: {result['profit_factor']}

Output format: {{"score": int, "lesson": "specific and actionable tuning advice"}}
Do not include markdown fences or any commentary.
"""

    candidate_models: list[str] = []
    for m in [
        model,
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]:
        if m and m not in candidate_models:
            candidate_models.append(m)

    score = 50
    lesson = DEFAULT_LESSON
    last_exc: Exception | None = None

    for candidate in candidate_models:
        try:
            payload: dict[str, Any] = {
                "model": candidate,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                payload["system"] = system_prompt

            response = client.messages.create(**payload)
            content = _extract_text_content(response) or "{}"
            parsed = _extract_json_object(content)
            score = int(parsed.get("score", score))
            lesson = str(parsed.get("lesson", lesson)).strip() or lesson
            break
        except Exception as exc:  # pragma: no cover - network/LLM failures
            last_exc = exc
            continue

    if last_exc and lesson == DEFAULT_LESSON:
        print(f"[reflection] Using default score/lesson because LLM call failed: {last_exc}", flush=True)

    score = max(0, min(100, score))
    return score, lesson
