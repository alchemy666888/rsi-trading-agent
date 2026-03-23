from __future__ import annotations

import json
from datetime import datetime, timezone
from math import isinf
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_COSTS = {"fee_rate": 0.0004, "slippage_rate": 0.0002, "max_loss_per_trade": 0.005}
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_btc_data",
            "description": "Fetch BTC/USDT OHLCV for a timeframe and date window (ISO strings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]},
                    "start": {"type": "string", "description": "ISO date/time inclusive (UTC)"},
                    "end": {"type": "string", "description": "ISO date/time inclusive (UTC)"},
                },
                "required": ["timeframe", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resample_features",
            "description": "Resample raw OHLCV into 15m/1h/4h/1d frames and compute core indicators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw": {"type": "array", "description": "List of OHLCV dicts or dataframe-like records"},
                    "timeframes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["raw"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_weekly_backtest",
            "description": "Backtest a single ISO week using multi-timeframe resonance and cost model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frames_by_tf": {"type": "object"},
                    "strategy_params": {"type": "object"},
                    "costs": {"type": "object"},
                },
                "required": ["frames_by_tf", "strategy_params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_weekly_score",
            "description": "Compute weighted weekly score = w1*Sharpe + w2*WinRate - w3*MaxDD - w4*Costs",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "object"},
                    "costs": {"type": "object"},
                    "weights": {"type": "object"},
                },
                "required": ["metrics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "persist_trace",
            "description": "Persist weekly trace JSON including metrics, score breakdown, and decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "week_id": {"type": "string"},
                    "strategy": {"type": "object"},
                    "metrics": {"type": "object"},
                    "score": {"type": "number"},
                    "decisions": {"type": "object"},
                    "trace_dir": {"type": "string"},
                },
                "required": ["week_id", "strategy", "metrics", "score"],
            },
        },
    },
]


def execute_tool(tool_call: dict[str, Any], require_confirmation: bool = False) -> Any:
    """Simple local tool router for agent loop."""

    name = tool_call["name"]
    args = tool_call.get("args", {})

    if name == "fetch_btc_data":
        return fetch_btc_data(**args)
    if name == "resample_features":
        return resample_features(**args)
    if name == "run_weekly_backtest":
        return run_weekly_backtest(**args)
    if name == "compute_weekly_score":
        return compute_weekly_score(**args)
    if name == "persist_trace":
        return persist_trace(**args)
    # Backward compatible V1 tool names
    if name == "fetch_btc_news":
        return fetch_btc_news(**args)
    if name == "calculate_indicators":
        return calculate_indicators(**args)
    if name == "run_backtest_simulation":
        return run_backtest_simulation(**args)
    return {"error": f"Unknown tool: {name}"}


def fetch_btc_data(timeframe: str = "1d", start: str | None = None, end: str | None = None, period: str | None = None) -> pd.DataFrame:
    """Fetch BTC/USDT OHLCV for timeframe between start and end (ISO strings, UTC) or for a period."""
    import ccxt
    from datetime import timedelta

    exchange = ccxt.binance()
    tf = timeframe

    if period:
        end_dt = datetime.now(timezone.utc)
        unit = period[-1]
        amount = int(period[:-1])
        if unit == "y":
            start_dt = end_dt - timedelta(days=365 * amount)
        elif unit == "d":
            start_dt = end_dt - timedelta(days=amount)
        elif unit == "h":
            start_dt = end_dt - timedelta(hours=amount)
        else:
            raise ValueError(f"Unsupported period unit: {unit}")
    elif start and end:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(timezone.utc)
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        raise ValueError("Either 'period' or both 'start' and 'end' must be provided.")

    since = int(start_dt.timestamp() * 1000)
    ohlcv: list[list[Any]] = []
    limit = 1000
    symbol = "BTC/USDT"

    while True:
        batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= int(end_dt.timestamp() * 1000) or len(batch) < limit:
            break
        since = last_ts + _tf_millis(tf)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
    df.set_index("timestamp", inplace=True)
    return df.sort_index()


def resample_features(raw: Iterable[dict[str, Any]] | pd.DataFrame, timeframes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Resample raw OHLCV into multiple frames and compute RSI/MACD/MA/BB/volatility per frame."""

    timeframes = timeframes or DEFAULT_TIMEFRAMES
    df = pd.DataFrame(raw).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)

    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    frames: dict[str, pd.DataFrame] = {}
    pandas_freq = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}

    for tf in timeframes:
        resampled = df.resample(pandas_freq.get(tf, tf)).agg(ohlc).dropna(how="any")
        frames[tf] = _add_indicators(resampled)

    return frames


def run_weekly_backtest(
    frames_by_tf: dict[str, pd.DataFrame],
    strategy_params: dict[str, Any],
    costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run a weekly backtest on 15m bars using multi-timeframe resonance and configurable costs."""
    result = _run_multi_timeframe_backtest(frames_by_tf=frames_by_tf, strategy=strategy_params, costs=costs)
    metrics = {key: result[key] for key in ["total_return", "sharpe", "max_dd", "win_rate", "profit_factor", "costs", "trade_count"]}
    return {"metrics": metrics, "positions": result["positions"], "returns": result["returns"]}


def compute_weekly_score(
    metrics: dict[str, float],
    costs: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted weekly score with configurable weights and cost penalty."""

    w = {"sharpe": 40.0, "win_rate": 30.0, "max_dd": 20.0, "costs": 10.0}
    w.update(weights or {})
    m_cost = metrics.get("costs", 0.0)
    m_dd = metrics.get("max_dd", 0.0)
    m_sharpe = metrics.get("sharpe", 0.0)
    m_wr = metrics.get("win_rate", 0.0)

    # Optionally strengthen drawdown penalty when past weeks exceed 30% via caller adjusting weights.
    score = (w["sharpe"] * m_sharpe) + (w["win_rate"] * m_wr) - (w["max_dd"] * m_dd) - (w["costs"] * m_cost)
    return float(score)


def persist_trace(
    week_id: str,
    strategy: dict[str, Any],
    metrics: dict[str, Any],
    score: float,
    decisions: dict[str, Any] | None = None,
    trace_dir: str = "traces",
) -> str:
    """Write weekly trace JSON to traces/week_{YYYY-WW}.json and return path."""

    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "week_id": week_id,
        "strategy": strategy,
        "metrics": metrics,
        "score": score,
        "decisions": decisions or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "rerun_with_tighter_risk" if score < 60 else "keep",
    }
    path = Path(trace_dir) / f"week_{week_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def write_backtest_report(
    *,
    epoch: int,
    strategy: dict[str, Any],
    metrics: dict[str, Any],
    trades: list[dict[str, Any]] | None = None,
    report_dir: str = "backtest",
    generated_at: str | None = None,
) -> str:
    """Write a Markdown backtest report and return the file path."""

    report_root = Path(report_dir)
    if not report_root.is_absolute():
        report_root = Path(__file__).resolve().parent.parent / report_root
    report_root.mkdir(parents=True, exist_ok=True)

    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    safe_timestamp = str(timestamp).replace(":", "-")
    path = report_root / f"epoch_{epoch:02d}_{safe_timestamp}.md"

    lines = [
        f"# Backtest Report: Epoch {epoch}",
        "",
        f"- Generated At: {generated_at or datetime.now(timezone.utc).isoformat()}",
        f"- Total Return: {_format_pct(metrics.get('total_return'))}",
        f"- Sharpe: {_format_decimal(metrics.get('sharpe'))}",
        f"- Max Drawdown: {_format_pct(metrics.get('max_dd'))}",
        f"- Win Rate: {_format_pct(metrics.get('win_rate'))}",
        f"- Profit Factor: {_format_decimal(metrics.get('profit_factor'))}",
        "",
        "## Strategy Parameters",
        "",
    ]

    for key in sorted(strategy):
        lines.append(f"- `{key}`: {strategy[key]}")

    lines.extend(
        [
            "",
            "## Trade Summary",
            "",
            f"- Trade Count: {len(trades or [])}",
        ]
    )

    if trades:
        for trade in trades:
            lines.extend(
                [
                    "",
                    f"### Trade {trade['trade_id']}: {trade['side']}",
                    "",
                    f"- Size: {_format_decimal(trade.get('size'))}",
                    f"- Entry Time: {trade.get('entry_time', 'N/A')}",
                    f"- Entry Price: {_format_price(trade.get('entry_price'))}",
                    f"- Exit Time: {trade.get('exit_time', 'OPEN')}",
                    f"- Exit Price: {_format_price(trade.get('exit_price'))}",
                    f"- Return: {_format_pct(trade.get('return_pct'))}",
                    f"- PnL: {_format_pct(trade.get('pnl_pct'))}",
                    f"- Bars Held: {trade.get('bars_held', 0)}",
                    f"- Entry Rationale: {trade.get('entry_rationale', 'N/A')}",
                    f"- Exit Rationale: {trade.get('exit_rationale', 'Position remained open at the end of the backtest.')}",
                ]
            )
    else:
        lines.extend(["", "No trades were executed in this backtest."])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --------------------
# V1 compatibility helpers (news + simple daily backtest)
# --------------------


def _heuristic_sentiment(text: str) -> float:
    positive = ["surge", "rally", "approve", "inflow", "adoption", "gain", "bull"]
    negative = ["ban", "hack", "lawsuit", "outflow", "drop", "bear", "crash"]
    lowered = text.lower()
    score = sum(0.2 for t in positive if t in lowered) - sum(0.2 for t in negative if t in lowered)
    return float(max(-1.0, min(1.0, score)))


def fetch_btc_news(limit: int = 10) -> list[dict[str, Any]]:
    # Try the renamed package first to avoid noisy runtime warnings.
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return []

    with DDGS() as ddgs:
        try:
            # ddgs >=6 renamed arg to 'query'
            results = [r for r in ddgs.news("Bitcoin BTC news", max_results=limit)]
        except TypeError:
            # fallback for duckduckgo_search API shape
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


def calculate_indicators(data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, list[float]]:
    """V1 daily indicator helper retained for compatibility."""

    _ = params  # params are unused; kept to preserve signature
    df = pd.DataFrame(data).copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    enriched = _add_indicators(df)
    return enriched.reset_index().to_dict(orient="list") if enriched.index.name else enriched.to_dict(orient="list")


def run_backtest_simulation(
    indicators: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
    strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal V1-style backtest kept to avoid breaking older entrypoints."""

    strategy = strategy or {"rsi_buy": 30, "rsi_sell": 70, "news_weight": 0.2}

    if {"15m", "1h", "4h", "1d"}.issubset(set(indicators.keys())):
        return _run_multi_timeframe_backtest(frames_by_tf=indicators, strategy=strategy, news=news)

    df = pd.DataFrame(indicators).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["signal"] = 0.0

    buy_condition = (df.get("RSI", 50) < strategy.get("rsi_buy", 30)) & (df.get("MACD", 0) > df.get("MACD_signal", 0))
    sell_condition = (df.get("RSI", 50) > strategy.get("rsi_sell", 70)) & (df.get("MACD", 0) < df.get("MACD_signal", 0))
    df.loc[buy_condition, "signal"] = 1.0
    df.loc[sell_condition, "signal"] = -1.0

    if news:
        avg_sentiment = float(np.mean([n.get("sentiment", 0.0) for n in news]))
        weight = float(strategy.get("news_weight", 0.2))
        df["signal"] = df["signal"] * (1 + avg_sentiment * weight)
    else:
        avg_sentiment = 0.0

    close = pd.to_numeric(df.get("Close", 0), errors="coerce")
    position = df["signal"].shift(1).fillna(0)
    returns = close.pct_change().fillna(0) * position
    cum_returns = (1 + returns).cumprod()
    rolling_peak = cum_returns.cummax().replace(0, np.nan)
    drawdown = ((rolling_peak - cum_returns) / rolling_peak).fillna(0)
    max_dd = float(drawdown.max() * 100) if len(drawdown) else 0.0

    neg_sum = returns[returns < 0].sum()
    profit_factor = float(abs(returns[returns > 0].sum() / neg_sum)) if neg_sum != 0 else float("inf")
    volatility = float(returns.std())
    trades = _extract_trade_details(df, position, close, returns, strategy, avg_sentiment)

    return {
        "total_return": float(returns.sum() * 100),
        "sharpe": float((returns.mean() / volatility) * np.sqrt(252)) if volatility != 0 else 0.0,
        "max_dd": max_dd,
        "win_rate": float((returns > 0).mean() * 100),
        "profit_factor": profit_factor,
        "trade_count": len(trades),
        "trades": trades,
    }


def _run_multi_timeframe_backtest(
    *,
    frames_by_tf: dict[str, Any],
    strategy: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
    costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    costs = {**DEFAULT_COSTS, **(costs or {})}
    prepared_frames = {tf: _coerce_frame(frame) for tf, frame in frames_by_tf.items()}

    entry_timeframe = str(strategy.get("entry_timeframe", "auto")).lower()
    if entry_timeframe not in {"15m", "1h", "auto"}:
        entry_timeframe = "auto"

    base_tf = "15m" if entry_timeframe == "auto" else entry_timeframe
    base = prepared_frames.get(base_tf)
    if base is None or base.empty:
        raise ValueError(f"frames_by_tf must include {base_tf} timeframe for execution")

    aligned: dict[str, pd.DataFrame] = {}
    for tf in ["15m", "1h", "4h", "1d"]:
        frame = prepared_frames.get(tf)
        if frame is None or frame.empty:
            raise ValueError(f"frames_by_tf missing timeframe: {tf}")
        aligned[tf] = frame.reindex(base.index, method="ffill")

    avg_sentiment = float(np.mean([n.get("sentiment", 0.0) for n in news])) if news else 0.0
    resonance_weight = float(strategy.get("weight_resonance", 1.1))
    conflict_penalty = float(strategy.get("conflict_penalty", 0.4))
    news_weight = float(strategy.get("news_weight", 0.3))
    max_loss = float(costs.get("max_loss_per_trade", 0.005))
    max_pos = float(strategy.get("max_position", 1.0))

    trend_4h = np.where(aligned["4h"]["EMA_short"] >= aligned["4h"]["EMA_long"], 1.0, -1.0)
    trend_1d = np.where(aligned["1d"]["EMA_short"] >= aligned["1d"]["EMA_long"], 1.0, -1.0)
    confirm_4h_long = aligned["4h"]["MACD"] >= aligned["4h"]["MACD_signal"]
    confirm_4h_short = aligned["4h"]["MACD"] <= aligned["4h"]["MACD_signal"]
    higher_tf_agree = trend_4h == trend_1d
    higher_tf_long = (trend_4h > 0) & (trend_1d > 0) & confirm_4h_long
    higher_tf_short = (trend_4h < 0) & (trend_1d < 0) & confirm_4h_short

    setup_15_long = (aligned["15m"]["RSI"] <= strategy.get("rsi_buy", 30)) & (
        aligned["15m"]["MACD"] >= aligned["15m"]["MACD_signal"]
    )
    setup_15_short = (aligned["15m"]["RSI"] >= strategy.get("rsi_sell", 70)) & (
        aligned["15m"]["MACD"] <= aligned["15m"]["MACD_signal"]
    )
    setup_1h_long = (aligned["1h"]["RSI"] <= strategy.get("rsi_buy", 30)) & (
        aligned["1h"]["MACD"] >= aligned["1h"]["MACD_signal"]
    )
    setup_1h_short = (aligned["1h"]["RSI"] >= strategy.get("rsi_sell", 70)) & (
        aligned["1h"]["MACD"] <= aligned["1h"]["MACD_signal"]
    )

    if entry_timeframe == "15m":
        long_setup = setup_15_long
        short_setup = setup_15_short
        entry_tf = np.where(long_setup | short_setup, "15m", "")
    elif entry_timeframe == "1h":
        long_setup = setup_1h_long
        short_setup = setup_1h_short
        entry_tf = np.where(long_setup | short_setup, "1h", "")
    else:
        long_setup = setup_15_long | setup_1h_long
        short_setup = setup_15_short | setup_1h_short
        entry_tf = np.where(setup_1h_long | setup_1h_short, "1h", np.where(setup_15_long | setup_15_short, "15m", ""))

    base = base.copy()
    base["trend_4h"] = trend_4h
    base["trend_1d"] = trend_1d
    base["entry_signal_tf"] = entry_tf
    base["higher_tf_aligned"] = higher_tf_agree.astype(float)
    base["signal"] = 0.0
    base.loc[long_setup & higher_tf_long, "signal"] = 1.0
    base.loc[short_setup & higher_tf_short, "signal"] = -1.0
    base.loc[higher_tf_agree, "signal"] *= resonance_weight
    base.loc[~higher_tf_agree, "signal"] *= conflict_penalty

    sentiment_bias = np.sign(avg_sentiment)
    long_regime = 1.0 + max(avg_sentiment, 0.0) * news_weight if sentiment_bias >= 0 else max(0.0, 1.0 + avg_sentiment * news_weight)
    short_regime = 1.0 + abs(min(avg_sentiment, 0.0)) * news_weight if sentiment_bias <= 0 else max(0.0, 1.0 - avg_sentiment * news_weight)
    base.loc[base["signal"] > 0, "signal"] *= long_regime
    base.loc[base["signal"] < 0, "signal"] *= short_regime
    base["news_regime_multiplier"] = np.where(base["signal"] > 0, long_regime, np.where(base["signal"] < 0, short_regime, 1.0))

    atr_pct = _atr_percentage(base)
    position: list[float] = []
    trade_costs: list[float] = []
    returns: list[float] = []
    prev_pos = 0.0

    pct_change = base["Close"].pct_change().fillna(0).to_numpy()
    signals = base["signal"].fillna(0).to_numpy()
    atr_values = atr_pct.to_numpy()

    for i, sig in enumerate(signals):
        allowed_pos = float(sig)
        if atr_values[i] > 0:
            cap = max_loss / atr_values[i]
            allowed_pos = float(np.clip(allowed_pos, -cap, cap))
        allowed_pos = float(np.clip(allowed_pos, -max_pos, max_pos))

        turnover = abs(allowed_pos - prev_pos)
        trade_cost = turnover * (costs["fee_rate"] + costs["slippage_rate"])
        ret = pct_change[i] * prev_pos - trade_cost

        position.append(allowed_pos)
        trade_costs.append(trade_cost)
        returns.append(ret)
        prev_pos = allowed_pos

    base["timestamp"] = base.index
    position_series = pd.Series(position, index=base.index, dtype=float)
    returns_series = pd.Series(returns, index=base.index, dtype=float)
    cum_returns = (1 + returns_series).cumprod()
    rolling_peak = cum_returns.cummax().replace(0, np.nan)
    drawdown = ((rolling_peak - cum_returns) / rolling_peak).fillna(0)
    neg_sum = returns_series[returns_series < 0].sum()
    profit_factor = float(abs(returns_series[returns_series > 0].sum() / neg_sum)) if neg_sum != 0 else float("inf")
    volatility = returns_series.std()
    bars_per_year = {"15m": 365 * 24 * 4, "1h": 365 * 24}.get(base_tf, 365 * 24 * 4)
    sharpe = float((returns_series.mean() / volatility) * np.sqrt(bars_per_year)) if volatility != 0 else 0.0
    trades = _extract_trade_details(base, position_series, pd.to_numeric(base["Close"], errors="coerce"), returns_series, strategy, avg_sentiment)

    return {
        "total_return": float(returns_series.sum() * 100),
        "sharpe": sharpe,
        "max_dd": float(drawdown.max() * 100) if len(drawdown) else 0.0,
        "win_rate": float((returns_series > 0).mean() * 100),
        "profit_factor": profit_factor,
        "costs": float(sum(trade_costs) * 100),
        "trade_count": len(trades),
        "trades": trades,
        "positions": position,
        "returns": returns_series.tolist(),
        "entry_timeframe": entry_timeframe,
        "execution_timeframe": base_tf,
        "news_sentiment": avg_sentiment,
    }


def _coerce_frame(frame: Any) -> pd.DataFrame:
    df = pd.DataFrame(frame).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df.sort_index()


def _tf_millis(timeframe: str) -> int:
    """Return milliseconds for a given ccxt timeframe string."""

    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    num = int(timeframe[:-1])
    unit = timeframe[-1]
    return num * multipliers[unit]


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA_short"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_long"] = out["Close"].ewm(span=26, adjust=False).mean()
    macd = out["EMA_short"] - out["EMA_long"]
    out["MACD"] = macd
    out["MACD_signal"] = macd.ewm(span=9, adjust=False).mean()

    delta = out["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    out["MA_short"] = out["Close"].rolling(window=10).mean()
    out["MA_long"] = out["Close"].rolling(window=50).mean()

    out["BB_mid"] = out["Close"].rolling(window=20).mean()
    out["BB_std"] = out["Close"].rolling(window=20).std()
    out["BB_upper"] = out["BB_mid"] + 2 * out["BB_std"]
    out["BB_lower"] = out["BB_mid"] - 2 * out["BB_std"]

    out["volatility"] = out["Close"].pct_change().rolling(window=20).std()
    return out


def _atr_percentage(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    atr_pct = (atr / close.replace(0, np.nan)).fillna(0)
    return atr_pct


def _extract_trade_details(
    df: pd.DataFrame,
    position: pd.Series,
    close: pd.Series,
    returns: pd.Series,
    strategy: dict[str, Any],
    avg_sentiment: float,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    active_trade: dict[str, Any] | None = None

    for idx in range(len(df)):
        current_pos = float(position.iloc[idx]) if idx < len(position) else 0.0
        prev_pos = float(position.iloc[idx - 1]) if idx > 0 else 0.0
        if current_pos == prev_pos:
            continue

        if prev_pos != 0 and active_trade is not None:
            exit_row = df.iloc[idx]
            exit_price = close.iloc[idx]
            cumulative_return = float((1 + returns.iloc[active_trade["start_index"] : idx + 1]).prod() - 1)
            active_trade.update(
                {
                    "exit_time": _format_timestamp(exit_row.get("timestamp")),
                    "exit_price": float(exit_price) if pd.notna(exit_price) else None,
                    "return_pct": cumulative_return * 100,
                    "pnl_pct": cumulative_return * 100 * abs(float(active_trade["size"])),
                    "bars_held": idx - active_trade["start_index"] + 1,
                    "exit_rationale": _build_exit_rationale(exit_row, current_pos, strategy),
                }
            )
            active_trade.pop("start_index", None)
            trades.append(active_trade)
            active_trade = None

        if current_pos != 0:
            entry_row = df.iloc[idx]
            entry_price = close.iloc[idx]
            side = "Long" if current_pos > 0 else "Short"
            active_trade = {
                "trade_id": len(trades) + 1,
                "side": side,
                "size": abs(float(current_pos)),
                "entry_time": _format_timestamp(entry_row.get("timestamp")),
                "entry_price": float(entry_price) if pd.notna(entry_price) else None,
                "entry_rationale": _build_entry_rationale(entry_row, current_pos, strategy, avg_sentiment),
                "start_index": idx,
            }

    if active_trade is not None:
        last_row = df.iloc[-1]
        cumulative_return = float((1 + returns.iloc[active_trade["start_index"] :]).prod() - 1)
        active_trade.update(
            {
                "exit_time": _format_timestamp(last_row.get("timestamp")),
                "exit_price": float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None,
                "return_pct": cumulative_return * 100,
                "pnl_pct": cumulative_return * 100 * abs(float(active_trade["size"])),
                "bars_held": len(df) - active_trade["start_index"],
                "exit_rationale": "Open position marked to the final available close because the backtest ended before an exit signal.",
            }
        )
        active_trade.pop("start_index", None)
        trades.append(active_trade)

    return trades


def _build_entry_rationale(row: pd.Series, position_size: float, strategy: dict[str, Any], avg_sentiment: float) -> str:
    side = "long" if position_size > 0 else "short"
    rsi = _format_decimal(row.get("RSI"))
    macd = _format_decimal(row.get("MACD"))
    macd_signal = _format_decimal(row.get("MACD_signal"))
    news_weight = _format_decimal(strategy.get("news_weight", 0.2))
    entry_tf = row.get("entry_signal_tf", "")
    trend_4h = row.get("trend_4h")
    trend_1d = row.get("trend_1d")
    regime_multiplier = _format_decimal(row.get("news_regime_multiplier", 1.0))
    if position_size > 0:
        trigger = f"RSI {rsi} below buy threshold {strategy.get('rsi_buy', 30)} with MACD {macd} above signal {macd_signal}"
    else:
        trigger = f"RSI {rsi} above sell threshold {strategy.get('rsi_sell', 70)} with MACD {macd} below signal {macd_signal}"
    if entry_tf:
        trigger = f"{entry_tf} entry trigger fired when {trigger}"
    if trend_4h in (1, 1.0, -1, -1.0) and trend_1d in (1, 1.0, -1, -1.0):
        trend_text = f" Higher-timeframe trend was {'bullish' if float(trend_4h) > 0 else 'bearish'} on 4h and {'bullish' if float(trend_1d) > 0 else 'bearish'} on 1d."
    else:
        trend_text = ""
    return (
        f"Opened {side} exposure when {trigger}. "
        f"Average news sentiment was {_format_decimal(avg_sentiment)}, scaled by news weight {news_weight} "
        f"for a regime multiplier of {regime_multiplier}, resulting in position size {_format_decimal(abs(position_size))}."
        f"{trend_text}"
    )


def _build_exit_rationale(row: pd.Series, next_position: float, strategy: dict[str, Any]) -> str:
    if next_position == 0:
        return "Exited because the strategy signal returned to neutral on the next bar."
    next_side = "long" if next_position > 0 else "short"
    return (
        f"Closed the prior trade because the strategy flipped to {next_side}. "
        f"At exit, RSI was {_format_decimal(row.get('RSI'))} and MACD was {_format_decimal(row.get('MACD'))} "
        f"versus signal {_format_decimal(row.get('MACD_signal'))}."
    )


def _format_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _format_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_decimal(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if isinf(numeric):
        return "inf"
    return f"{numeric:.4f}"


def _format_price(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
