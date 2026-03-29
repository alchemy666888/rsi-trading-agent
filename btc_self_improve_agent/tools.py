from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG


DEFAULT_COSTS = {"fee_rate": 0.0004, "slippage_rate": 0.0002, "max_loss_per_trade": 0.005}
DEFAULT_TIMEFRAMES = list(DEFAULT_CONFIG.timeframes)

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
            "description": "Resample raw OHLCV into 15m/1h/4h/1d frames and compute indicators.",
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
            "description": "Backtest a single period using multi-timeframe trend/execution hierarchy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frames_by_tf": {"type": "object"},
                    "strategy_params": {"type": "object"},
                    "costs": {"type": "object"},
                    "news": {"type": "array"},
                },
                "required": ["frames_by_tf", "strategy_params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_weekly_score",
            "description": "Compute weighted score = w1*Sharpe + w2*WinRate - w3*MaxDD - w4*Costs",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {"type": "object"},
                    "weights": {"type": "object"},
                },
                "required": ["metrics"],
            },
        },
    },
]


def execute_tool(tool_call: dict[str, Any], require_confirmation: bool = False) -> Any:
    del require_confirmation
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

    # Backward-compatible names
    if name == "fetch_btc_news":
        return fetch_btc_news(**args)
    if name == "calculate_indicators":
        return calculate_indicators(**args)
    if name == "run_backtest_simulation":
        return run_backtest_simulation(**args)
    return {"error": f"Unknown tool: {name}"}


def fetch_btc_data(
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
) -> pd.DataFrame:
    """Fetch BTC/USDT OHLCV for timeframe between start and end (UTC)."""
    import ccxt
    from datetime import timedelta

    exchange = ccxt.binance({"enableRateLimit": True})
    tf = timeframe

    if period:
        # Compatibility path; explicit range is preferred.
        end_dt = datetime.now(UTC)
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
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(UTC)
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00")).astimezone(UTC)
    else:
        raise ValueError("Either 'period' or both 'start' and 'end' must be provided.")

    since = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    ohlcv: list[list[Any]] = []
    limit = 1000

    while True:
        batch = exchange.fetch_ohlcv(DEFAULT_CONFIG.symbol_exchange, tf, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        last_ts = int(batch[-1][0])
        if last_ts >= end_ms or len(batch) < limit:
            break
        since = last_ts + _tf_millis(tf)

    cols = [
        "open_time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    if not ohlcv:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # fetch_ohlcv usually returns 6 fields; enrich schema for downstream consistency.
    frame = pd.DataFrame(ohlcv, columns=["open_time", "Open", "High", "Low", "Close", "Volume"])
    frame["close_time"] = frame["open_time"] + _tf_millis(tf) - 1
    frame["quote_asset_volume"] = frame["Close"] * frame["Volume"]
    frame["number_of_trades"] = np.nan
    frame["taker_buy_base_asset_volume"] = np.nan
    frame["taker_buy_quote_asset_volume"] = np.nan
    frame["ignore"] = np.nan
    frame = frame[cols]
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame[(frame["open_time"] >= start_dt) & (frame["open_time"] <= end_dt)].copy()
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    frame.rename(columns={"open_time": "timestamp"}, inplace=True)
    frame.set_index("timestamp", inplace=True)
    return frame


def fetch_btc_data_bundle(
    start: str | None = None,
    end: str | None = None,
    timeframes: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    start = start or DEFAULT_CONFIG.start_ts
    end = end or DEFAULT_CONFIG.end_ts
    for timeframe in list(timeframes or DEFAULT_TIMEFRAMES):
        frame = fetch_btc_data(timeframe=timeframe, start=start, end=end)
        validate_ohlcv_continuity(frame, timeframe=timeframe)
        frames[timeframe] = frame
    return frames


def validate_ohlcv_continuity(df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    if df.empty:
        return {"ok": False, "missing_count": 0, "duplicate_count": 0, "rows": 0}
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV frame index must be DatetimeIndex")
    expected = pd.date_range(df.index.min(), df.index.max(), freq=_pandas_freq(timeframe), tz="UTC")
    missing = expected.difference(df.index)
    duplicate_count = int(df.index.duplicated().sum())
    return {
        "ok": len(missing) == 0 and duplicate_count == 0,
        "missing_count": int(len(missing)),
        "duplicate_count": duplicate_count,
        "rows": int(len(df)),
    }


def resample_features(raw: Iterable[dict[str, Any]] | pd.DataFrame, timeframes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Resample raw OHLCV into multiple frames and compute full feature set."""
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    df = pd.DataFrame(raw).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing OHLCV column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ohlc = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    frames: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        resampled = df.resample(_pandas_freq(tf)).agg(ohlc).dropna(how="any")
        frames[tf] = _add_indicators(resampled, timeframe=tf)
    return frames


def run_weekly_backtest(
    frames_by_tf: dict[str, pd.DataFrame],
    strategy_params: dict[str, Any],
    costs: dict[str, float] | None = None,
    news: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = _run_multi_timeframe_backtest(frames_by_tf=frames_by_tf, strategy=strategy_params, costs=costs, news=news)
    metrics = {
        key: result[key]
        for key in [
            "total_return",
            "sharpe",
            "max_dd",
            "win_rate",
            "profit_factor",
            "costs",
            "trade_count",
        ]
    }
    return {"metrics": metrics, "positions": result["positions"], "returns": result["returns"], "trades": result["trades"]}


def compute_weekly_score(metrics: dict[str, float], costs: dict[str, float] | None = None, weights: dict[str, float] | None = None) -> float:
    del costs
    w = {"sharpe": 40.0, "win_rate": 30.0, "max_dd": 20.0, "costs": 10.0}
    w.update(weights or {})
    score = (w["sharpe"] * float(metrics.get("sharpe", 0.0))) + (w["win_rate"] * float(metrics.get("win_rate", 0.0)))
    score -= (w["max_dd"] * float(metrics.get("max_dd", 0.0)))
    score -= (w["costs"] * float(metrics.get("costs", 0.0)))
    return float(score)


def persist_trace(
    week_id: str,
    strategy: dict[str, Any],
    metrics: dict[str, Any],
    score: float,
    decisions: dict[str, Any] | None = None,
    trace_dir: str = "traces",
) -> str:
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    payload = {
        "week_id": week_id,
        "strategy": strategy,
        "metrics": metrics,
        "score": score,
        "decisions": decisions or {},
        "timestamp": datetime.now(UTC).isoformat(),
        "action": "rerun_with_tighter_risk" if score < DEFAULT_CONFIG.monthly_score_threshold else "keep",
    }
    path = Path(trace_dir) / f"week_{week_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def fetch_btc_news(limit: int = 10) -> list[dict[str, Any]]:
    # V1 compatibility API retained.
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return []
    with DDGS() as ddgs:
        try:
            results = [r for r in ddgs.news("Bitcoin BTC crypto web3 news", max_results=limit)]
        except TypeError:
            results = [r for r in ddgs.news(keywords="Bitcoin BTC crypto web3 news", max_results=limit)]
    enriched: list[dict[str, Any]] = []
    for row in results:
        text = f"{row.get('title', '')} {row.get('body', '')}"
        sentiment = _heuristic_sentiment(text)
        enriched.append(
            {
                "title": row.get("title", ""),
                "date": row.get("date", ""),
                "url": row.get("url", ""),
                "sentiment": sentiment,
                "sentiment_strength": abs(sentiment),
                "impact_lag_bucket": "0-1h",
                "impact_duration_bucket": "1-3d",
                "impacted_timeframe": "1h",
            }
        )
    return enriched


def enrich_news_records(news_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in news_records:
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        text = f"{title} {snippet}".strip()
        sentiment = float(item.get("sentiment", _heuristic_sentiment(text)))
        polarity = "neutral"
        if sentiment > 0.15:
            polarity = "bullish"
        elif sentiment < -0.15:
            polarity = "bearish"
        strength = min(1.0, abs(sentiment) + (0.15 if _contains_event_keywords(text) else 0.0))

        published = _parse_datetime(item.get("published_at") or item.get("date") or item.get("search_window_start"))
        if published is None:
            published = datetime.now(UTC)
        lag, duration, impacted_tf = _infer_impact_profile(text=text, polarity=polarity, strength=strength)
        lag_start, lag_end = _bucket_to_timedelta(lag)
        dur_start, dur_end = _bucket_to_timedelta(duration)

        enriched.append(
            {
                **item,
                "published_at": published.astimezone(UTC).isoformat(),
                "sentiment": sentiment,
                "sentiment_polarity": polarity,
                "sentiment_strength": strength,
                "sentiment_confidence": min(1.0, 0.4 + strength * 0.6),
                "impact_lag_bucket": lag,
                "impact_duration_bucket": duration,
                "impacted_timeframe": impacted_tf,
                "impact_start": (published + lag_start).astimezone(UTC).isoformat(),
                "impact_end": (published + lag_end + dur_end).astimezone(UTC).isoformat(),
                "impact_peak_start": (published + lag_start + dur_start).astimezone(UTC).isoformat(),
            }
        )
    return sorted(enriched, key=lambda x: x.get("published_at", ""))


def calculate_indicators(data: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, list[float]]:
    del params
    df = pd.DataFrame(data).copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    enriched = _add_indicators(df, timeframe="1d")
    if isinstance(enriched.index, pd.DatetimeIndex):
        return enriched.reset_index().to_dict(orient="list")
    return enriched.to_dict(orient="list")


def run_backtest_simulation(
    indicators: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
    strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    strategy = strategy or {}
    if {"15m", "1h", "4h", "1d"}.issubset(set(indicators.keys())):
        return _run_multi_timeframe_backtest(frames_by_tf=indicators, strategy=strategy, news=news)

    df = pd.DataFrame(indicators).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)
    if "RSI" not in df.columns or "MACD" not in df.columns or "MACD_signal" not in df.columns:
        df = _add_indicators(df, timeframe="1d")
    return _run_single_frame_backtest(df, strategy=strategy, news=news)


def write_backtest_report(
    *,
    epoch: int,
    strategy: dict[str, Any],
    metrics: dict[str, Any],
    trades: list[dict[str, Any]] | None = None,
    report_dir: str = "backtest",
    generated_at: str | None = None,
    title: str | None = None,
) -> str:
    report_root = Path(report_dir)
    if not report_root.is_absolute():
        report_root = Path(__file__).resolve().parent.parent / report_root
    report_root.mkdir(parents=True, exist_ok=True)

    timestamp = generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = report_root / f"epoch_{epoch:02d}_{str(timestamp).replace(':', '-')}.md"
    month_rows = metrics.get("monthly_breakdown", [])
    regime_rows = metrics.get("regime_breakdown", [])

    lines = [
        f"# {title or f'Backtest Report: Epoch {epoch}'}",
        "",
        f"- Generated At: {generated_at or datetime.now(UTC).isoformat()}",
        f"- Total Return: {_format_pct(metrics.get('total_return'))}",
        f"- Annualized Return: {_format_pct(metrics.get('annualized_return'))}",
        f"- Sharpe: {_format_decimal(metrics.get('sharpe'))}",
        f"- Sortino: {_format_decimal(metrics.get('sortino'))}",
        f"- Max Drawdown: {_format_pct(metrics.get('max_dd'))}",
        f"- Calmar: {_format_decimal(metrics.get('calmar'))}",
        f"- Win Rate: {_format_pct(metrics.get('win_rate'))}",
        f"- Profit Factor: {_format_decimal(metrics.get('profit_factor'))}",
        f"- Expectancy: {_format_pct(metrics.get('expectancy'))}",
        f"- Trade Count: {metrics.get('trade_count', len(trades or []))}",
        "",
        "## Strategy Parameters",
        "",
    ]
    for key in sorted(strategy):
        lines.append(f"- `{key}`: {strategy[key]}")
    lines.extend(["", "## Monthly Breakdown", ""])
    if month_rows:
        for row in month_rows:
            lines.append(
                f"- `{row.get('month_id')}` return={_format_pct(row.get('total_return'))}, "
                f"max_dd={_format_pct(row.get('max_dd'))}, trades={row.get('trade_count', 0)}"
            )
    else:
        lines.append("- Not available.")
    lines.extend(["", "## Regime Breakdown", ""])
    if regime_rows:
        for row in regime_rows:
            lines.append(
                f"- `{row.get('regime')}` trades={row.get('trade_count', 0)}, "
                f"win_rate={_format_pct(row.get('win_rate'))}, return={_format_pct(row.get('total_return'))}"
            )
    else:
        lines.append("- Not available.")
    lines.extend(["", "## Trade Details", ""])
    if trades:
        for trade in trades:
            lines.extend(
                [
                    f"### Trade {trade.get('trade_id', '?')} ({trade.get('side', 'N/A')})",
                    "",
                    f"- Entry Time: {trade.get('entry_time', 'N/A')}",
                    f"- Exit Time: {trade.get('exit_time', 'N/A')}",
                    f"- Entry Price: {_format_price(trade.get('entry_price'))}",
                    f"- Exit Price: {_format_price(trade.get('exit_price'))}",
                    f"- Size: {_format_decimal(trade.get('size'))}",
                    f"- Return: {_format_pct(trade.get('return_pct'))}",
                    f"- PnL: {_format_pct(trade.get('pnl_pct'))}",
                    f"- Bars Held: {trade.get('bars_held', 0)}",
                    f"- Higher TF Context: {trade.get('higher_tf_context_summary', 'N/A')}",
                    f"- Lower TF Trigger: {trade.get('lower_tf_trigger_summary', 'N/A')}",
                    f"- News Filter: {trade.get('news_filter_summary', 'N/A')}",
                    f"- Entry Rationale: {trade.get('entry_rationale', 'N/A')}",
                    f"- Exit Rationale: {trade.get('exit_rationale', 'N/A')}",
                    "",
                ]
            )
    else:
        lines.append("No trades executed.")

    lines.extend(
        [
            "",
            "## Narrative Analysis",
            "",
            str(metrics.get("narrative", "No narrative summary provided.")),
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_monthly_report(
    *,
    month_id: str,
    strategy: dict[str, Any],
    metrics: dict[str, Any],
    lesson: str,
    report_dir: str = "backtest",
) -> str:
    report_root = Path(report_dir)
    if not report_root.is_absolute():
        report_root = Path(__file__).resolve().parent.parent / report_root
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / f"monthly_{month_id}.md"
    lines = [
        f"# Monthly Paper Trading Report: {month_id}",
        "",
        f"- Total Return: {_format_pct(metrics.get('total_return'))}",
        f"- Sharpe: {_format_decimal(metrics.get('sharpe'))}",
        f"- Max Drawdown: {_format_pct(metrics.get('max_dd'))}",
        f"- Win Rate: {_format_pct(metrics.get('win_rate'))}",
        f"- Profit Factor: {_format_decimal(metrics.get('profit_factor'))}",
        f"- Trade Count: {metrics.get('trade_count', 0)}",
        "",
        "## Lesson Learned",
        "",
        lesson,
        "",
        "## Strategy Snapshot",
        "",
    ]
    for key in sorted(strategy.keys()):
        lines.append(f"- `{key}`: {strategy[key]}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_final_strategy_markdown(
    *,
    strategy: dict[str, Any],
    monthly_history: list[dict[str, Any]],
    output_path: str = "docs/final-btc-strategy.md",
) -> str:
    path = Path(output_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final BTC Strategy",
        "",
        "- Asset: `BTCUSDT`",
        "- Timeframes: `4h` + `1d` for trend context; `15m` + `1h` for entries/exits",
        "- Data Range: `2023-01-01` to `2025-12-31`",
        "",
        "## Rules",
        "",
        "- Trend context must align on `4h` and `1d` before entries.",
        "- Entries are triggered on `15m` or `1h` using RSI/MACD and confirmation rules.",
        "- News sentiment is used as a regime modifier with impact-lag windows.",
        "- Risk controls use ATR-based stop and capped position sizing.",
        "",
        "## Final Parameters",
        "",
    ]
    for key in sorted(strategy):
        lines.append(f"- `{key}`: {strategy[key]}")
    lines.extend(["", "## Monthly Learning Summary", ""])
    for item in monthly_history:
        lines.append(f"- `{item.get('month_id')}` score={_format_decimal(item.get('score'))}: {item.get('lesson', '')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _run_single_frame_backtest(df: pd.DataFrame, strategy: dict[str, Any], news: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    frame = df.copy()
    frame = _add_indicators(frame, timeframe="1d")
    signal = np.zeros(len(frame), dtype=float)
    rsi_buy = float(strategy.get("rsi_buy", 30))
    rsi_sell = float(strategy.get("rsi_sell", 70))
    long_cond = (frame["RSI"] <= rsi_buy) & (frame["MACD"] >= frame["MACD_signal"])
    short_cond = (frame["RSI"] >= rsi_sell) & (frame["MACD"] <= frame["MACD_signal"])
    signal[long_cond.to_numpy()] = 1.0
    signal[short_cond.to_numpy()] = -1.0
    frame["signal"] = signal
    returns, position, costs = _simulate_positions(frame, signal=signal, costs=DEFAULT_COSTS)
    return _finalize_backtest_result(
        frame=frame,
        position=position,
        returns=returns,
        costs=costs,
        strategy=strategy,
        news=news or [],
        entry_timeframe="auto",
        execution_timeframe="1d",
    )


def _run_multi_timeframe_backtest(
    *,
    frames_by_tf: dict[str, Any],
    strategy: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
    costs: dict[str, float] | None = None,
) -> dict[str, Any]:
    costs = {**DEFAULT_COSTS, **(costs or {})}
    prepared = {tf: _coerce_frame(frame) for tf, frame in frames_by_tf.items()}
    for tf in ["15m", "1h", "4h", "1d"]:
        if tf not in prepared or prepared[tf].empty:
            raise ValueError(f"frames_by_tf missing timeframe: {tf}")

    entry_timeframe = str(strategy.get("entry_timeframe", "auto")).lower()
    if entry_timeframe not in {"15m", "1h", "auto"}:
        entry_timeframe = "auto"
    base_tf = "15m" if entry_timeframe == "auto" else entry_timeframe
    base = prepared[base_tf].copy()
    aligned = {tf: prepared[tf].reindex(base.index, method="ffill") for tf in ["15m", "1h", "4h", "1d"]}

    rsi_buy = float(strategy.get("rsi_buy", 30))
    rsi_sell = float(strategy.get("rsi_sell", 70))
    resonance = float(strategy.get("weight_resonance", 1.1))
    conflict_penalty = float(strategy.get("conflict_penalty", 0.4))
    trend_strength = float(strategy.get("trend_filter_strength", 0.6))
    max_pos = float(strategy.get("max_position", 1.0))

    trend_4h = np.where(
        (aligned["4h"]["EMA_short"] >= aligned["4h"]["EMA_long"]) & (aligned["4h"]["Close"] >= aligned["4h"]["SMA_50"]),
        1.0,
        -1.0,
    )
    trend_1d = np.where(
        (aligned["1d"]["EMA_short"] >= aligned["1d"]["EMA_long"]) & (aligned["1d"]["Close"] >= aligned["1d"]["SMA_50"]),
        1.0,
        -1.0,
    )
    higher_aligned = trend_4h == trend_1d
    higher_long = (trend_4h > 0) & (trend_1d > 0) & (aligned["4h"]["MACD"] >= aligned["4h"]["MACD_signal"])
    higher_short = (trend_4h < 0) & (trend_1d < 0) & (aligned["4h"]["MACD"] <= aligned["4h"]["MACD_signal"])

    long_15 = (aligned["15m"]["RSI"] <= rsi_buy) & (aligned["15m"]["MACD"] >= aligned["15m"]["MACD_signal"])
    short_15 = (aligned["15m"]["RSI"] >= rsi_sell) & (aligned["15m"]["MACD"] <= aligned["15m"]["MACD_signal"])
    long_1h = (aligned["1h"]["RSI"] <= rsi_buy) & (aligned["1h"]["MACD"] >= aligned["1h"]["MACD_signal"])
    short_1h = (aligned["1h"]["RSI"] >= rsi_sell) & (aligned["1h"]["MACD"] <= aligned["1h"]["MACD_signal"])

    if entry_timeframe == "15m":
        long_setup, short_setup = long_15, short_15
        entry_tf = np.where(long_15 | short_15, "15m", "")
    elif entry_timeframe == "1h":
        long_setup, short_setup = long_1h, short_1h
        entry_tf = np.where(long_1h | short_1h, "1h", "")
    else:
        long_setup = long_1h | long_15
        short_setup = short_1h | short_15
        entry_tf = np.where(long_1h | short_1h, "1h", np.where(long_15 | short_15, "15m", ""))

    signal = np.zeros(len(base), dtype=float)
    long_idx = (long_setup & higher_long).to_numpy()
    short_idx = (short_setup & higher_short).to_numpy()
    signal[long_idx] = 1.0
    signal[short_idx] = -1.0
    signal[np.where(higher_aligned.to_numpy(), True, False)] *= resonance
    signal[np.where(~higher_aligned.to_numpy(), True, False)] *= conflict_penalty

    # News impact: use enriched historical events with impact windows.
    impact_series = compute_news_impact_series(base.index, enrich_news_records(news or []))
    news_weight = float(strategy.get("news_weight", 0.3))
    signal = signal * (1.0 + (impact_series.to_numpy() * news_weight))

    # Downweight when higher-timeframe trend is weak.
    weak_trend = np.abs(aligned["4h"]["MACD"].to_numpy()) < trend_strength * np.nanstd(aligned["4h"]["MACD"].to_numpy())
    signal = np.where(weak_trend, signal * conflict_penalty, signal)
    signal = np.clip(signal, -max_pos, max_pos)

    base["trend_4h"] = trend_4h
    base["trend_1d"] = trend_1d
    base["entry_signal_tf"] = entry_tf
    base["signal"] = signal
    base["news_impact"] = impact_series.to_numpy()
    returns, position, per_bar_costs = _simulate_positions(base, signal=signal, costs=costs)
    return _finalize_backtest_result(
        frame=base,
        position=position,
        returns=returns,
        costs=per_bar_costs,
        strategy=strategy,
        news=news or [],
        entry_timeframe=entry_timeframe,
        execution_timeframe=base_tf,
    )


def _simulate_positions(frame: pd.DataFrame, signal: np.ndarray, costs: dict[str, float]) -> tuple[pd.Series, pd.Series, list[float]]:
    atr = _atr_percentage(frame).to_numpy()
    pct_change = pd.to_numeric(frame["Close"], errors="coerce").pct_change().fillna(0).to_numpy()
    max_loss = float(costs.get("max_loss_per_trade", 0.005))
    max_pos = float(np.nanmax(np.abs(signal))) if len(signal) else 0.0

    position: list[float] = []
    returns: list[float] = []
    trade_costs: list[float] = []
    prev_pos = 0.0
    for i, sig in enumerate(signal):
        allowed = float(sig)
        if i < len(atr) and atr[i] > 0:
            cap = max_loss / atr[i]
            allowed = float(np.clip(allowed, -cap, cap))
        allowed = float(np.clip(allowed, -max(1e-9, max_pos), max(1e-9, max_pos)))
        turnover = abs(allowed - prev_pos)
        trade_cost = turnover * (float(costs["fee_rate"]) + float(costs["slippage_rate"]))
        ret = pct_change[i] * prev_pos - trade_cost
        position.append(allowed)
        returns.append(ret)
        trade_costs.append(trade_cost)
        prev_pos = allowed
    return pd.Series(returns, index=frame.index), pd.Series(position, index=frame.index), trade_costs


def _finalize_backtest_result(
    *,
    frame: pd.DataFrame,
    position: pd.Series,
    returns: pd.Series,
    costs: list[float],
    strategy: dict[str, Any],
    news: list[dict[str, Any]],
    entry_timeframe: str,
    execution_timeframe: str,
) -> dict[str, Any]:
    cum = (1 + returns).cumprod()
    peak = cum.cummax().replace(0, np.nan)
    drawdown = ((peak - cum) / peak).fillna(0)
    neg_sum = returns[returns < 0].sum()
    profit_factor = float(abs(returns[returns > 0].sum() / neg_sum)) if neg_sum != 0 else float("inf")
    vol = float(returns.std())
    bars_per_year = {"15m": 365 * 24 * 4, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}.get(execution_timeframe, 365)
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(bars_per_year)) if vol != 0 else 0.0
    downside = returns[returns < 0].std()
    sortino = float((returns.mean() / downside) * np.sqrt(bars_per_year)) if downside not in (0.0, np.nan) and not math.isnan(float(downside)) else 0.0
    total_return = float(returns.sum() * 100)
    annualized = float(((1 + (returns.sum())) ** (bars_per_year / max(1, len(returns))) - 1) * 100) if len(returns) > 0 else 0.0
    max_dd = float(drawdown.max() * 100) if len(drawdown) else 0.0
    calmar = float((annualized / max_dd) if max_dd > 0 else 0.0)
    expectancy = float(returns.mean() * 100)
    trades = _extract_trade_details(frame, position, pd.to_numeric(frame["Close"], errors="coerce"), returns, strategy, news)
    regime_breakdown = _regime_breakdown(trades)
    monthly_breakdown = _monthly_breakdown(returns, drawdown, trades)
    narrative = (
        "The strategy uses 4h and 1d trend alignment to gate directional bias, then executes with 15m/1h entries. "
        "News impact windows modulate exposure rather than acting as standalone triggers."
    )
    return {
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": float((returns > 0).mean() * 100),
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "costs": float(sum(costs) * 100),
        "trade_count": len(trades),
        "trades": trades,
        "positions": position.tolist(),
        "returns": returns.tolist(),
        "entry_timeframe": entry_timeframe,
        "execution_timeframe": execution_timeframe,
        "news_event_count": len(news),
        "monthly_breakdown": monthly_breakdown,
        "regime_breakdown": regime_breakdown,
        "narrative": narrative,
    }


def compute_news_impact_series(index: pd.DatetimeIndex, news_events: list[dict[str, Any]]) -> pd.Series:
    impact = pd.Series(0.0, index=index, dtype=float)
    if len(index) == 0:
        return impact
    for event in news_events:
        start = _parse_datetime(event.get("impact_start") or event.get("published_at"))
        end = _parse_datetime(event.get("impact_end") or event.get("published_at"))
        if start is None or end is None:
            continue
        if end < index.min().to_pydatetime().replace(tzinfo=UTC) or start > index.max().to_pydatetime().replace(tzinfo=UTC):
            continue
        sentiment = float(event.get("sentiment", 0.0))
        strength = float(event.get("sentiment_strength", abs(sentiment)))
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        impact.loc[mask] += sentiment * max(0.1, strength)
    return impact.clip(lower=-1.5, upper=1.5)


def _heuristic_sentiment(text: str) -> float:
    positive = ["surge", "rally", "approve", "inflow", "adoption", "gain", "bull", "etf", "institutional"]
    negative = ["ban", "hack", "lawsuit", "outflow", "drop", "bear", "crash", "liquidation", "default"]
    lowered = text.lower()
    score = sum(0.18 for t in positive if t in lowered) - sum(0.18 for t in negative if t in lowered)
    return float(max(-1.0, min(1.0, score)))


def _contains_event_keywords(text: str) -> bool:
    text = text.lower()
    return any(token in text for token in ["etf", "sec", "regulation", "hack", "liquidation", "fed", "cpi"])


def _infer_impact_profile(text: str, polarity: str, strength: float) -> tuple[str, str, str]:
    t = text.lower()
    if "etf" in t or "sec" in t or "regulation" in t:
        return ("1-4h", "3-7d", "4h")
    if "hack" in t or "liquidation" in t:
        return ("0-1h", "4-12h", "15m")
    if "fed" in t or "inflation" in t or "cpi" in t:
        return ("1-4h", "1-3d", "1h")
    if polarity == "neutral":
        return ("4-12h", "12-24h", "1h")
    if strength >= 0.7:
        return ("0-1h", "1-3d", "1h")
    return ("1-4h", "12-24h", "1h")


def _bucket_to_timedelta(bucket: str) -> tuple[pd.Timedelta, pd.Timedelta]:
    lookup = {
        "0-1h": (pd.Timedelta(hours=0), pd.Timedelta(hours=1)),
        "1-4h": (pd.Timedelta(hours=1), pd.Timedelta(hours=4)),
        "4-12h": (pd.Timedelta(hours=4), pd.Timedelta(hours=12)),
        "12-24h": (pd.Timedelta(hours=12), pd.Timedelta(hours=24)),
        "1-3d": (pd.Timedelta(days=1), pd.Timedelta(days=3)),
        "3-7d": (pd.Timedelta(days=3), pd.Timedelta(days=7)),
    }
    return lookup.get(bucket, (pd.Timedelta(hours=0), pd.Timedelta(hours=12)))


def _coerce_frame(frame: Any) -> pd.DataFrame:
    df = pd.DataFrame(frame).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index().dropna(subset=["Open", "High", "Low", "Close", "Volume"], how="any")
    return _add_indicators(df, timeframe="15m")


def _tf_millis(timeframe: str) -> int:
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    num = int(timeframe[:-1])
    unit = timeframe[-1]
    return num * multipliers[unit]


def _pandas_freq(timeframe: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    return mapping[timeframe]


def _add_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    volume = pd.to_numeric(out["Volume"], errors="coerce")

    out["EMA_short"] = close.ewm(span=12, adjust=False).mean()
    out["EMA_long"] = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA_short"] - out["EMA_long"]
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    gain_fast = delta.where(delta > 0, 0.0).rolling(window=7).mean()
    loss_fast = (-delta.where(delta < 0, 0.0)).rolling(window=7).mean()
    rs_fast = gain_fast / loss_fast.replace(0, np.nan)
    out["RSI_fast"] = 100 - (100 / (1 + rs_fast))

    out["SMA_10"] = close.rolling(window=10).mean()
    out["SMA_20"] = close.rolling(window=20).mean()
    out["SMA_50"] = close.rolling(window=50).mean()
    out["SMA_200"] = close.rolling(window=200).mean()
    out["MA_short"] = out["SMA_10"]
    out["MA_long"] = out["SMA_50"]

    out["BB_mid"] = close.rolling(window=20).mean()
    out["BB_std"] = close.rolling(window=20).std()
    out["BB_upper"] = out["BB_mid"] + 2 * out["BB_std"]
    out["BB_lower"] = out["BB_mid"] - 2 * out["BB_std"]
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / out["BB_mid"].replace(0, np.nan)

    # Ichimoku
    conversion = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    base = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    span_a = ((conversion + base) / 2).shift(26)
    span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    lagging = close.shift(-26)
    out["ICHIMOKU_tenkan"] = conversion
    out["ICHIMOKU_kijun"] = base
    out["ICHIMOKU_span_a"] = span_a
    out["ICHIMOKU_span_b"] = span_b
    out["ICHIMOKU_chikou"] = lagging
    out["ICHIMOKU_cloud_bias"] = np.where(span_a >= span_b, 1.0, -1.0)

    # VWAP
    typical = (high + low + close) / 3.0
    tpv = (typical * volume).fillna(0)
    cum_tpv = tpv.cumsum()
    cum_vol = volume.replace(0, np.nan).cumsum()
    out["VWAP"] = cum_tpv / cum_vol
    out["VWAP_rolling"] = (typical * volume).rolling(window=20).sum() / volume.replace(0, np.nan).rolling(window=20).sum()

    # Volume profile approximations
    vp_window = 96 if timeframe == "15m" else 60 if timeframe == "1h" else 40
    out["VP_POC"] = close.rolling(window=vp_window).median()
    out["VP_VAL"] = close.rolling(window=vp_window).quantile(0.3)
    out["VP_VAH"] = close.rolling(window=vp_window).quantile(0.7)
    vol_q75 = volume.rolling(window=vp_window).quantile(0.75)
    vol_q25 = volume.rolling(window=vp_window).quantile(0.25)
    out["VP_HVN_FLAG"] = (volume >= vol_q75).astype(float)
    out["VP_LVN_FLAG"] = (volume <= vol_q25).astype(float)

    out["volatility"] = close.pct_change().rolling(window=20).std()
    out["trend_slope_20"] = out["SMA_20"].diff(5)
    out["volume_surge_ratio"] = volume / volume.rolling(window=20).mean().replace(0, np.nan)
    out["body_to_range"] = (close - out["Open"]).abs() / (high - low).replace(0, np.nan)
    out["trend_alignment_score"] = np.where(
        (out["EMA_short"] >= out["EMA_long"]) & (out["SMA_20"] >= out["SMA_50"]),
        1.0,
        -1.0,
    )
    return out


def _atr_percentage(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return (atr / close.replace(0, np.nan)).fillna(0)


def _extract_trade_details(
    df: pd.DataFrame,
    position: pd.Series,
    close: pd.Series,
    returns: pd.Series,
    strategy: dict[str, Any],
    news_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for idx in range(len(df)):
        cur = float(position.iloc[idx]) if idx < len(position) else 0.0
        prev = float(position.iloc[idx - 1]) if idx > 0 else 0.0
        if cur == prev:
            continue
        row = df.iloc[idx]
        ts = _format_timestamp(row.name if not isinstance(row.get("timestamp"), pd.Timestamp) else row.get("timestamp"))
        if prev != 0 and active is not None:
            cumulative_return = float((1 + returns.iloc[active["start_index"] : idx + 1]).prod() - 1)
            active.update(
                {
                    "exit_time": ts,
                    "exit_price": float(close.iloc[idx]) if pd.notna(close.iloc[idx]) else None,
                    "return_pct": cumulative_return * 100,
                    "pnl_pct": cumulative_return * 100 * abs(float(active["size"])),
                    "bars_held": idx - active["start_index"] + 1,
                    "exit_rationale": _build_exit_rationale(row, cur),
                }
            )
            active.pop("start_index", None)
            trades.append(active)
            active = None
        if cur != 0:
            side = "Long" if cur > 0 else "Short"
            active = {
                "trade_id": len(trades) + 1,
                "side": side,
                "size": abs(float(cur)),
                "entry_time": ts,
                "entry_price": float(close.iloc[idx]) if pd.notna(close.iloc[idx]) else None,
                "higher_tf_context_summary": _higher_tf_summary(row),
                "lower_tf_trigger_summary": _lower_tf_summary(row),
                "news_filter_summary": _news_summary(news_events),
                "entry_rationale": _build_entry_rationale(row, cur, strategy),
                "start_index": idx,
            }
    if active is not None:
        cumulative_return = float((1 + returns.iloc[active["start_index"] :]).prod() - 1)
        active.update(
            {
                "exit_time": _format_timestamp(df.index[-1]),
                "exit_price": float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None,
                "return_pct": cumulative_return * 100,
                "pnl_pct": cumulative_return * 100 * abs(float(active["size"])),
                "bars_held": len(df) - active["start_index"],
                "exit_rationale": "Marked to close at end of backtest window.",
            }
        )
        active.pop("start_index", None)
        trades.append(active)
    return trades


def _higher_tf_summary(row: pd.Series) -> str:
    trend_4h = row.get("trend_4h")
    trend_1d = row.get("trend_1d")
    if trend_4h in (-1.0, 1.0) and trend_1d in (-1.0, 1.0):
        return f"4h={'bullish' if trend_4h > 0 else 'bearish'}, 1d={'bullish' if trend_1d > 0 else 'bearish'}"
    return "Higher timeframe trend unavailable"


def _lower_tf_summary(row: pd.Series) -> str:
    return f"Entry trigger timeframe: {row.get('entry_signal_tf', 'N/A')}, RSI={_format_decimal(row.get('RSI'))}, MACD={_format_decimal(row.get('MACD'))}"


def _news_summary(news_events: list[dict[str, Any]]) -> str:
    if not news_events:
        return "No active news events applied."
    avg_sent = float(np.mean([float(n.get("sentiment", 0.0)) for n in news_events]))
    return f"{len(news_events)} enriched news events, average sentiment {_format_decimal(avg_sent)}"


def _build_entry_rationale(row: pd.Series, position_size: float, strategy: dict[str, Any]) -> str:
    side = "long" if position_size > 0 else "short"
    return (
        f"Opened {side} position because 4h/1d trend filter allowed the direction, "
        f"lower timeframe trigger ({row.get('entry_signal_tf', 'N/A')}) confirmed RSI/MACD setup, "
        f"and signal strength after news/regime adjustment was {_format_decimal(position_size)}."
    )


def _build_exit_rationale(row: pd.Series, next_position: float) -> str:
    if next_position == 0:
        return "Exited because signal returned to neutral."
    return f"Exited because strategy flipped to {'long' if next_position > 0 else 'short'}."


def _monthly_breakdown(returns: pd.Series, drawdown: pd.Series, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if returns.empty:
        return []
    monthly_returns = returns.resample("MS").sum()
    monthly_dd = drawdown.resample("MS").max()
    trade_counts: dict[str, int] = {}
    for trade in trades:
        month_id = str(trade.get("entry_time", ""))[:7]
        trade_counts[month_id] = trade_counts.get(month_id, 0) + 1
    rows: list[dict[str, Any]] = []
    for idx, value in monthly_returns.items():
        month = idx.strftime("%Y-%m")
        rows.append(
            {
                "month_id": month,
                "total_return": float(value * 100),
                "max_dd": float(monthly_dd.get(idx, 0.0) * 100),
                "trade_count": trade_counts.get(month, 0),
            }
        )
    return rows


def _regime_breakdown(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[float]] = {"long": [], "short": []}
    for trade in trades:
        side = "long" if str(trade.get("side", "")).lower().startswith("long") else "short"
        buckets[side].append(float(trade.get("return_pct", 0.0)))
    rows: list[dict[str, Any]] = []
    for regime, vals in buckets.items():
        if not vals:
            rows.append({"regime": regime, "trade_count": 0, "win_rate": 0.0, "total_return": 0.0})
            continue
        arr = np.array(vals, dtype=float)
        rows.append(
            {
                "regime": regime,
                "trade_count": int(len(arr)),
                "win_rate": float((arr > 0).mean() * 100),
                "total_return": float(arr.sum()),
            }
        )
    return rows


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue
    return None


def _format_timestamp(value: Any) -> str:
    if value is None:
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
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(val):
        return "inf"
    return f"{val:.4f}"


def _format_price(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)
