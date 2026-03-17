from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_btc_data",
            "description": "Fetch BTC historical OHLCV data from Binance",
            "parameters": {"type": "object", "properties": {"period": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_btc_news",
            "description": "Fetch BTC related news and attach a heuristic sentiment score",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_indicators",
            "description": "Calculate RSI, MACD, SMA/EMA, and Bollinger Bands",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "params": {"type": "object"},
                },
                "required": ["data", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest_simulation",
            "description": "Run vectorized backtest and return performance metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicators": {"type": "object"},
                    "news": {"type": "array"},
                    "strategy": {"type": "object"},
                },
                "required": ["indicators", "news", "strategy"],
            },
        },
    },
]


def execute_tool(tool_call: dict[str, Any], require_confirmation: bool = True) -> Any:
    """Simple local tool router for agent loop."""
    name = tool_call["name"]
    args = tool_call.get("args", {})

    if "backtest" in name and require_confirmation:
        confirm = input("⚠️ Start simulation backtest? Type Y to continue: ")
        if confirm.upper() != "Y":
            return {"error": "User cancelled"}

    if name == "fetch_btc_data":
        return fetch_btc_data(**args)
    if name == "fetch_btc_news":
        return fetch_btc_news(**args)
    if name == "calculate_indicators":
        return calculate_indicators(**args)
    if name == "run_backtest_simulation":
        return run_backtest_simulation(**args)
    return {"error": f"Unknown tool: {name}"}


def fetch_btc_data(period: str = "2y") -> dict[str, list[float]]:
    """Fetch OHLCV and return JSON-friendly column-major data."""
    import ccxt

    exchange = ccxt.binance()
    now = datetime.now(timezone.utc)
    days = 730 if period == "2y" else 365
    since = int((now - timedelta(days=days)).timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", since=since)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype(str)
    return df.to_dict(orient="list")


def _heuristic_sentiment(text: str) -> float:
    """Very lightweight deterministic sentiment heuristic in [-1, 1]."""
    positive = ["surge", "rally", "approve", "inflow", "adoption", "gain", "bull"]
    negative = ["ban", "hack", "lawsuit", "outflow", "drop", "bear", "crash"]
    lowered = text.lower()
    score = sum(0.2 for t in positive if t in lowered) - sum(0.2 for t in negative if t in lowered)
    return float(max(-1.0, min(1.0, score)))


def fetch_btc_news(limit: int = 10) -> list[dict[str, Any]]:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = [r for r in ddgs.news(keywords="Bitcoin BTC news", max_results=limit)]

    sentiments: list[dict[str, Any]] = []
    for news in results:
        title = news.get("title", "")
        body = news.get("body", "")
        sentiments.append(
            {
                "title": title,
                "date": news.get("date", ""),
                "url": news.get("url", ""),
                "sentiment": _heuristic_sentiment(f"{title} {body}"),
            }
        )
    return sentiments


def calculate_indicators(data: dict[str, Any], params: dict[str, Any]) -> dict[str, list[float]]:
    df = pd.DataFrame(data).copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["SMA_short"] = df["Close"].rolling(window=params.get("ma_short", 10)).mean()
    df["SMA_long"] = df["Close"].rolling(window=params.get("ma_long", 50)).mean()

    df["EMA_short"] = df["Close"].ewm(span=params.get("ema_short", 12), adjust=False).mean()
    df["EMA_long"] = df["Close"].ewm(span=params.get("ema_long", 26), adjust=False).mean()

    macd = df["EMA_short"] - df["EMA_long"]
    signal = macd.ewm(span=params.get("macd_signal", 9), adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = signal

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params.get("rsi_period", 14)).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    bb_period = params.get("bb_period", 20)
    df["BB_mid"] = df["Close"].rolling(window=bb_period).mean()
    df["BB_std"] = df["Close"].rolling(window=bb_period).std()
    df["BB_upper"] = df["BB_mid"] + df["BB_std"] * params.get("bb_std", 2)
    df["BB_lower"] = df["BB_mid"] - df["BB_std"] * params.get("bb_std", 2)

    return df.to_dict(orient="list")


def run_backtest_simulation(
    indicators: dict[str, Any],
    news: list[dict[str, Any]],
    strategy: dict[str, Any],
) -> dict[str, float]:
    df = pd.DataFrame(indicators).copy()
    df["signal"] = 0.0

    buy_condition = (df["RSI"] < strategy.get("rsi_buy", 30)) & (df["MACD"] > df["MACD_signal"])
    sell_condition = (df["RSI"] > strategy.get("rsi_sell", 70)) & (df["MACD"] < df["MACD_signal"])
    df.loc[buy_condition, "signal"] = 1.0
    df.loc[sell_condition, "signal"] = -1.0

    if news:
        avg_sentiment = float(np.mean([n.get("sentiment", 0.0) for n in news]))
        weight = float(strategy.get("news_weight", 0.2))
        df["signal"] = df["signal"] * (1 + avg_sentiment * weight)

    returns = pd.to_numeric(df["Close"], errors="coerce").pct_change().fillna(0) * df["signal"].shift(1).fillna(0)
    cum_returns = (1 + returns).cumprod()

    rolling_peak = cum_returns.cummax().replace(0, np.nan)
    drawdown = ((rolling_peak - cum_returns) / rolling_peak).fillna(0)
    max_dd = float(drawdown.max() * 100) if len(drawdown) else 0.0

    neg_sum = returns[returns < 0].sum()
    profit_factor = float(abs(returns[returns > 0].sum() / neg_sum)) if neg_sum != 0 else float("inf")
    volatility = float(returns.std())

    return {
        "total_return": float(returns.sum() * 100),
        "sharpe": float((returns.mean() / volatility) * np.sqrt(252)) if volatility != 0 else 0.0,
        "max_dd": max_dd,
        "win_rate": float((returns > 0).mean() * 100),
        "profit_factor": profit_factor,
    }
