from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_STRATEGY: dict[str, float] = {
    "rsi_buy": 30,
    "rsi_sell": 70,
    "ma_short": 10,
    "ma_long": 50,
    "ema_short": 12,
    "ema_long": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2,
    "news_weight": 0.3,
}


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from plain text or fenced markdown blocks."""
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


def _sanitize_strategy(raw: dict[str, Any]) -> dict[str, float]:
    out = DEFAULT_STRATEGY.copy()

    def _num(name: str, lo: float, hi: float) -> None:
        if name in raw:
            try:
                out[name] = float(raw[name])
            except (TypeError, ValueError):
                return
            out[name] = min(hi, max(lo, out[name]))

    _num("rsi_buy", 5, 50)
    _num("rsi_sell", 50, 95)
    _num("ma_short", 2, 100)
    _num("ma_long", 5, 300)
    _num("ema_short", 2, 100)
    _num("ema_long", 5, 300)
    _num("macd_signal", 2, 50)
    _num("rsi_period", 2, 50)
    _num("bb_period", 5, 100)
    _num("bb_std", 1, 4)
    _num("news_weight", 0, 2)

    if out["ma_short"] >= out["ma_long"]:
        out["ma_short"] = min(out["ma_long"] - 1, out["ma_short"])
    if out["ema_short"] >= out["ema_long"]:
        out["ema_short"] = min(out["ema_long"] - 1, out["ema_short"])

    return out


def create_strategy_plan(client: Any, user_goal: str, context: str, system_prompt: str = "") -> dict[str, float]:
    prompt = f"""
User goal: {user_goal}
Past lessons: {context}

Return only JSON for BTC daily strategy parameters.
Example: {{"rsi_buy": 30, "rsi_sell": 70, "ma_short": 10, "ma_long": 50, "news_weight": 0.5}}
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
        raw = _extract_json_object(content)
    except Exception:
        raw = {}

    return _sanitize_strategy(raw)
