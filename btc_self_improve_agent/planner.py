from __future__ import annotations

import json
import re
from typing import Any


DEFAULT_STRATEGY: dict[str, Any] = {
    "strategy_name": "btc_multi_tf_news_regime_v1",
    "rsi_buy": 42,
    "rsi_sell": 62,
    "rsi_fast_period": 7,
    "ma_short": 10,
    "ma_long": 50,
    "sma_short": 10,
    "sma_long": 50,
    "ema_short": 12,
    "ema_long": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2,
    "ichimoku_conversion_period": 9,
    "ichimoku_base_period": 26,
    "ichimoku_span_b_period": 52,
    "vwap_window": 20,
    "volume_profile_window": 96,
    "news_weight": 0.12,
    "weight_resonance": 1.08,
    "conflict_penalty": 0.45,
    "max_position": 0.55,
    "stop_atr_multiple": 2.6,
    "take_profit_atr_multiple": 5.0,
    "cooldown_bars_after_news": 8,
    "trend_filter_strength": 0.62,
    "entry_timeframe": "auto",
    "long_score_threshold": 0.82,
    "short_score_threshold": 0.95,
    "score_hysteresis": 0.08,
    "max_hold_bars": 192,
    "breakeven_r_multiple": 1.0,
    "news_impact_cap": 0.6,
    "long_pullback_rsi_1h": 46,
    "long_pullback_rsi_15m": 44,
    "breakout_volume_surge": 1.20,
    "short_breakdown_volume_surge": 1.60,
    "short_rally_rsi_1h": 58,
    "trade_cooldown_bars": 32,
    "breakout_1h_window": 72,
    "breakout_15m_window": 32,
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


def _sanitize_strategy(raw: dict[str, Any]) -> dict[str, Any]:
    out = DEFAULT_STRATEGY.copy()

    entry_timeframe = str(raw.get("entry_timeframe", out.get("entry_timeframe", "auto"))).lower()
    out["entry_timeframe"] = entry_timeframe if entry_timeframe in {"15m", "1h", "auto"} else "auto"

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
    _num("rsi_fast_period", 2, 21)
    _num("bb_period", 5, 100)
    _num("bb_std", 1, 4)
    _num("ichimoku_conversion_period", 5, 20)
    _num("ichimoku_base_period", 10, 60)
    _num("ichimoku_span_b_period", 20, 120)
    _num("vwap_window", 5, 200)
    _num("volume_profile_window", 10, 500)
    _num("news_weight", 0, 2)
    _num("weight_resonance", 0.5, 2)
    _num("conflict_penalty", 0, 1)
    _num("max_position", 0.1, 3)
    _num("stop_atr_multiple", 0.5, 5)
    _num("take_profit_atr_multiple", 0.5, 8)
    _num("cooldown_bars_after_news", 0, 48)
    _num("trend_filter_strength", 0, 1)
    _num("long_score_threshold", 0.4, 0.95)
    _num("short_score_threshold", 0.4, 0.95)
    _num("score_hysteresis", 0.0, 0.4)
    _num("max_hold_bars", 4, 480)
    _num("breakeven_r_multiple", 0.5, 3.0)
    _num("news_impact_cap", 0.2, 2.0)
    _num("long_pullback_rsi_1h", 20, 65)
    _num("long_pullback_rsi_15m", 15, 60)
    _num("breakout_volume_surge", 1.0, 3.0)
    _num("short_breakdown_volume_surge", 1.0, 3.0)
    _num("short_rally_rsi_1h", 45, 85)
    _num("trade_cooldown_bars", 0, 96)
    _num("breakout_1h_window", 12, 240)
    _num("breakout_15m_window", 8, 192)

    if out["ma_short"] >= out["ma_long"]:
        out["ma_short"] = min(out["ma_long"] - 1, out["ma_short"])
    out["sma_short"] = out["ma_short"]
    out["sma_long"] = out["ma_long"]
    if out["ema_short"] >= out["ema_long"]:
        out["ema_short"] = min(out["ema_long"] - 1, out["ema_short"])
    if out["ichimoku_conversion_period"] >= out["ichimoku_base_period"]:
        out["ichimoku_conversion_period"] = max(5, out["ichimoku_base_period"] - 1)
    if out["ichimoku_base_period"] >= out["ichimoku_span_b_period"]:
        out["ichimoku_base_period"] = max(10, out["ichimoku_span_b_period"] - 1)
    if out["long_score_threshold"] > out["short_score_threshold"]:
        out["short_score_threshold"] = min(0.95, out["long_score_threshold"] + 0.02)

    return out


def _extract_text_content(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part)


def create_strategy_plan(
    client: Any,
    user_goal: str,
    context: str,
    system_prompt: str = "",
    model: str = "claude-3-5-sonnet-20240620",
) -> dict[str, Any]:
    prompt = f"""
User goal: {user_goal}
Past lessons: {context}

Return only JSON for BTC multi-timeframe strategy parameters.
Use 1d for macro bias, 4h for confirmation, 1h for setup confirmation, and 15m for trigger timing.
Keep BTC structurally long-biased unless the 1d macro regime is decisively bearish.
Treat news sentiment as an asymmetric risk modifier, not a standalone trigger.
Example: {{"entry_timeframe": "auto", "rsi_buy": 42, "rsi_sell": 62, "weight_resonance": 1.08, "conflict_penalty": 0.45, "news_weight": 0.12}}
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

    raw: dict[str, Any] = {}
    last_exc: Exception | None = None

    for candidate in candidate_models:
        try:
            payload: dict[str, Any] = {
                "model": candidate,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                payload["system"] = system_prompt

            response = client.messages.create(**payload)
            content = _extract_text_content(response) or "{}"
            raw = _extract_json_object(content)
            if raw:
                break
        except Exception as exc:  # pragma: no cover - network/LLM failures
            last_exc = exc
            continue

    if not raw:
        print(f"[planner] Using default strategy because LLM call failed: {last_exc}", flush=True)
        raw = {}

    return _sanitize_strategy(raw)


def update_strategy_from_lesson(
    base_strategy: dict[str, Any],
    lesson: str,
    monthly_metrics: dict[str, Any],
    client: Any | None = None,
    model: str = "claude-3-5-sonnet-20240620",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update strategy conservatively using monthly lesson and metrics."""

    baseline = _sanitize_strategy(base_strategy)
    prompt = f"""
You are updating a BTC strategy after one monthly walk-forward cycle.
Return only JSON with keys:
{{
  "updated_strategy": {{ ...strategy fields... }},
  "change_log": {{
    "summary": "one sentence",
    "changes": ["change 1", "change 2"]
  }}
}}
Rules:
- Keep changes conservative.
- Do not change more than 6 numeric fields.
- Respect multi-timeframe design: 4h/1d context, 15m/1h execution.

Current strategy:
{json.dumps(baseline, indent=2)}

Lesson:
{lesson}

Metrics:
{json.dumps(monthly_metrics, indent=2)}
"""
    parsed: dict[str, Any] = {}
    if client is not None:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}],
            )
            content = _extract_text_content(response) or "{}"
            parsed = _extract_json_object(content)
        except Exception:
            parsed = {}

    candidate = parsed.get("updated_strategy", parsed if isinstance(parsed, dict) else {})
    updated = _sanitize_strategy(candidate if isinstance(candidate, dict) else baseline)
    if not isinstance(candidate, dict) or not candidate:
        updated = _rule_based_strategy_update(baseline, lesson=lesson, metrics=monthly_metrics)

    change_log = parsed.get("change_log")
    if not isinstance(change_log, dict):
        change_log = _build_change_log(baseline, updated, lesson)
    if "changes" not in change_log or not isinstance(change_log.get("changes"), list):
        change_log["changes"] = _diff_changes(baseline, updated)
    if "summary" not in change_log:
        change_log["summary"] = "Applied conservative monthly strategy update from lesson."

    return updated, change_log


def _rule_based_strategy_update(base: dict[str, Any], lesson: str, metrics: dict[str, Any]) -> dict[str, Any]:
    updated = dict(base)
    lesson_l = lesson.lower()
    max_dd = float(metrics.get("max_dd", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    trade_count = float(metrics.get("trade_count", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    total_return = float(metrics.get("total_return", 0.0))
    trades = metrics.get("trades", []) if isinstance(metrics.get("trades", []), list) else []
    long_count = sum(1 for trade in trades if str(trade.get("side", "")).lower().startswith("long"))
    short_count = sum(1 for trade in trades if str(trade.get("side", "")).lower().startswith("short"))

    if max_dd > 25:
        updated["max_position"] = max(0.4, float(updated["max_position"]) * 0.92)
    if trade_count < 12:
        updated["long_score_threshold"] = max(0.68, float(updated["long_score_threshold"]) - 0.01)
        updated["breakout_volume_surge"] = max(1.1, float(updated["breakout_volume_surge"]) - 0.03)
        updated["max_hold_bars"] = min(144, float(updated["max_hold_bars"]) + 6)
        updated["trade_cooldown_bars"] = max(8, float(updated["trade_cooldown_bars"]) - 2)
    if total_return < 0 and profit_factor < 1.0:
        updated["news_weight"] = max(0.05, float(updated["news_weight"]) - 0.02)
        updated["short_score_threshold"] = min(0.95, float(updated["short_score_threshold"]) + 0.02)
        updated["conflict_penalty"] = min(0.8, float(updated["conflict_penalty"]) + 0.03)
        updated["trade_cooldown_bars"] = min(48, float(updated["trade_cooldown_bars"]) + 2)
        if short_count > 0:
            updated["short_breakdown_volume_surge"] = min(2.2, float(updated["short_breakdown_volume_surge"]) + 0.05)
        if long_count <= short_count:
            updated["long_score_threshold"] = max(0.68, float(updated["long_score_threshold"]) - 0.01)
    if total_return < 0 and win_rate < 42 and long_count > 0:
        updated["long_pullback_rsi_15m"] = max(30, float(updated["long_pullback_rsi_15m"]) - 1)
        updated["long_pullback_rsi_1h"] = max(34, float(updated["long_pullback_rsi_1h"]) - 1)
    if "early" in lesson_l or "late" in lesson_l:
        updated["long_score_threshold"] = max(0.68, float(updated["long_score_threshold"]) - 0.01)
        updated["score_hysteresis"] = min(0.12, float(updated["score_hysteresis"]) + 0.01)
    if "news" in lesson_l or "headline" in lesson_l:
        updated["news_weight"] = max(0.05, float(updated["news_weight"]) - 0.02)
        updated["cooldown_bars_after_news"] = min(16, float(updated["cooldown_bars_after_news"]) + 1)
    if win_rate < 45:
        updated["rsi_sell"] = min(70, float(updated["rsi_sell"]) + 1)
        updated["short_rally_rsi_1h"] = min(65, float(updated["short_rally_rsi_1h"]) + 1)
    if sharpe < 0.5:
        updated["weight_resonance"] = min(1.2, float(updated["weight_resonance"]) + 0.01)
    elif sharpe > 1.0 and total_return > 0:
        updated["max_position"] = min(1.0, float(updated["max_position"]) * 1.03)
        updated["weight_resonance"] = min(1.2, float(updated["weight_resonance"]) + 0.01)

    return _sanitize_strategy(updated)


def _diff_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            changes.append(f"{key}: {before.get(key)} -> {after.get(key)}")
    return changes


def _build_change_log(before: dict[str, Any], after: dict[str, Any], lesson: str) -> dict[str, Any]:
    return {
        "summary": f"Updated strategy from lesson: {lesson}",
        "changes": _diff_changes(before, after),
    }
