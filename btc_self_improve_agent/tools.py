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
    score = (w["sharpe"] * float(metrics.get("sharpe", 0.0))) + (w["win_rate"] * (float(metrics.get("win_rate", 0.0)) / 100.0))
    score -= (w["max_dd"] * (float(metrics.get("max_dd", 0.0)) / 100.0))
    score -= (w["costs"] * (float(metrics.get("costs", 0.0)) / 100.0))
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
        event_type = _infer_event_type(text)
        published_conf = str(item.get("published_at_confidence", "unknown")).lower()
        confidence_factor = {"high": 1.0, "medium": 0.8, "unknown": 0.6, "low": 0.35}.get(published_conf, 0.6)

        published = _parse_datetime(item.get("published_at") or item.get("date") or item.get("search_window_start"))
        if published is None:
            # Keep deterministic fallback for legacy records, but mark as low-confidence.
            published = datetime.now(UTC)
            if published_conf == "unknown":
                published_conf = "low"
                confidence_factor = 0.35
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
                "sentiment_confidence": min(1.0, (0.35 + strength * 0.5) * confidence_factor + 0.2),
                "published_at_confidence": published_conf,
                "event_type": event_type,
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
    df = pd.DataFrame(data).copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    enriched = _add_indicators(df, timeframe="1d", strategy=params)
    if isinstance(enriched.index, pd.DatetimeIndex):
        return enriched.reset_index().to_dict(orient="list")
    return enriched.to_dict(orient="list")


def run_backtest_simulation(
    indicators: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
    strategy: dict[str, Any] | None = None,
    evaluation_start: datetime | str | None = None,
    evaluation_end: datetime | str | None = None,
) -> dict[str, Any]:
    strategy = strategy or {}
    if {"15m", "1h", "4h", "1d"}.issubset(set(indicators.keys())):
        return _run_multi_timeframe_backtest(
            frames_by_tf=indicators,
            strategy=strategy,
            news=news,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )

    df = pd.DataFrame(indicators).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)
    if "RSI" not in df.columns or "MACD" not in df.columns or "MACD_signal" not in df.columns:
        df = _add_indicators(df, timeframe="1d", strategy=strategy)
    return _run_single_frame_backtest(
        df,
        strategy=strategy,
        news=news,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )


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
        f"- Average Win: {_format_pct(metrics.get('avg_win'))}",
        f"- Average Loss: {_format_pct(metrics.get('avg_loss'))}",
        f"- Average Holding Bars: {_format_decimal(metrics.get('avg_holding_bars'))}",
        f"- Trade Count: {metrics.get('trade_count', len(trades or []))}",
        f"- Reliability Flags: {', '.join(metrics.get('reliability_flags', [])) if metrics.get('reliability_flags') else 'none'}",
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
        f"- Expectancy: {_format_pct(metrics.get('expectancy'))}",
        f"- Trade Count: {metrics.get('trade_count', 0)}",
        f"- Reliability Flags: {', '.join(metrics.get('reliability_flags', [])) if metrics.get('reliability_flags') else 'none'}",
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
        "- `1d` defines the macro bias with a slow SMA-driven bull/bear filter and `4h` confirms direction.",
        "- `1h` confirms the setup family and `15m` fine-tunes trigger timing for live entries and exits.",
        "- BTC is treated as structurally long-biased unless the daily macro regime is decisively bearish.",
        "- News sentiment is used as an asymmetric risk modifier and cooldown filter, not a standalone trigger.",
        "- Risk controls use ATR-based stop logic, capped position sizing, and setup-aware target management.",
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


def _run_single_frame_backtest(
    df: pd.DataFrame,
    strategy: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
    evaluation_start: datetime | str | None = None,
    evaluation_end: datetime | str | None = None,
) -> dict[str, Any]:
    frame = df.copy()
    required = {"RSI", "MACD", "MACD_signal"}
    if not required.issubset(frame.columns):
        frame = _add_indicators(frame, timeframe="1d", strategy=strategy)
    if "MACD_hist" not in frame.columns:
        frame["MACD_hist"] = pd.to_numeric(frame["MACD"], errors="coerce") - pd.to_numeric(frame["MACD_signal"], errors="coerce")
    if "RSI_fast" not in frame.columns:
        frame["RSI_fast"] = pd.to_numeric(frame["RSI"], errors="coerce")
    signal = np.zeros(len(frame), dtype=float)
    rsi_buy = float(strategy.get("rsi_buy", 30))
    rsi_sell = float(strategy.get("rsi_sell", 70))
    long_cond = ((frame["RSI"] <= rsi_buy) | ((frame["RSI"].shift(1) <= rsi_buy) & (frame["RSI"] > rsi_buy))) & (frame["MACD"] >= frame["MACD_signal"])
    short_cond = ((frame["RSI"] >= rsi_sell) | ((frame["RSI"].shift(1) >= rsi_sell) & (frame["RSI"] < rsi_sell))) & (frame["MACD"] <= frame["MACD_signal"])
    signal[long_cond.to_numpy()] = 1.0
    signal[short_cond.to_numpy()] = -1.0
    eval_start = _parse_datetime(evaluation_start)
    if eval_start is not None:
        signal = np.where(frame.index < pd.Timestamp(eval_start), 0.0, signal)
    frame["signal"] = signal
    frame["trend_4h"] = np.sign(signal)
    frame["trend_1d"] = np.sign(signal)
    frame["entry_signal_tf"] = "1d"
    frame["news_impact"] = 0.0
    frame["active_news_count"] = 0
    frame["news_cooldown"] = 0.0
    returns, position, costs, event_notes = _simulate_positions(frame, signal=signal, costs=DEFAULT_COSTS, strategy=strategy)
    return _finalize_backtest_result(
        frame=frame,
        position=position,
        returns=returns,
        costs=costs,
        event_notes=event_notes,
        strategy=strategy,
        news=news or [],
        entry_timeframe="auto",
        execution_timeframe="1d",
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )


def _run_multi_timeframe_backtest(
    *,
    frames_by_tf: dict[str, Any],
    strategy: dict[str, Any],
    news: list[dict[str, Any]] | None = None,
    costs: dict[str, float] | None = None,
    evaluation_start: datetime | str | None = None,
    evaluation_end: datetime | str | None = None,
) -> dict[str, Any]:
    costs = {**DEFAULT_COSTS, **(costs or {})}
    prepared = {tf: _coerce_frame(frame, timeframe=tf, strategy=strategy) for tf, frame in frames_by_tf.items()}
    for tf in ["15m", "1h", "4h", "1d"]:
        if tf not in prepared or prepared[tf].empty:
            raise ValueError(f"frames_by_tf missing timeframe: {tf}")

    entry_timeframe = str(strategy.get("entry_timeframe", "auto")).lower()
    if entry_timeframe not in {"15m", "1h", "auto"}:
        entry_timeframe = "auto"
    base_tf = "15m"
    base = prepared[base_tf].copy()
    aligned = {tf: prepared[tf].reindex(base.index, method="ffill") for tf in ["15m", "1h", "4h", "1d"]}

    rsi_buy = float(strategy.get("rsi_buy", 30))
    rsi_sell = float(strategy.get("rsi_sell", 70))
    resonance = float(strategy.get("weight_resonance", 1.1))
    conflict_penalty = float(strategy.get("conflict_penalty", 0.4))
    trend_strength = float(strategy.get("trend_filter_strength", 0.6))
    max_pos = float(strategy.get("max_position", 1.0))
    long_score_threshold = float(strategy.get("long_score_threshold", 0.72))
    short_score_threshold = float(strategy.get("short_score_threshold", 0.72))
    score_hysteresis = float(strategy.get("score_hysteresis", 0.05))
    news_impact_cap = float(strategy.get("news_impact_cap", 0.8))
    trend_strength = min(0.8, max(0.55, trend_strength))
    breakout_volume_surge = float(strategy.get("breakout_volume_surge", 1.30))
    short_breakdown_volume_surge = float(strategy.get("short_breakdown_volume_surge", breakout_volume_surge + 0.15))
    long_pullback_rsi_1h = float(strategy.get("long_pullback_rsi_1h", min(46.0, rsi_buy + 4.0)))
    long_pullback_rsi_15m = float(strategy.get("long_pullback_rsi_15m", min(44.0, rsi_buy + 2.0)))
    short_rally_rsi_1h = float(strategy.get("short_rally_rsi_1h", max(56.0, rsi_sell - 4.0)))
    breakout_1h_window = max(12, int(round(float(strategy.get("breakout_1h_window", 72)))))
    breakout_15m_window = max(8, int(round(float(strategy.get("breakout_15m_window", 32)))))

    def _as_series(values: Any) -> pd.Series:
        if isinstance(values, pd.Series):
            return values.reindex(base.index).fillna(False)
        return pd.Series(values, index=base.index)

    def _score_series(components: list[Any]) -> pd.Series:
        return sum(_as_series(component).astype(float) for component in components) / len(components)

    daily_close = aligned["1d"]["Close"]
    daily_sma_long = aligned["1d"]["SMA_long"].fillna(daily_close)
    daily_sma_200 = aligned["1d"]["SMA_200"].fillna(daily_sma_long)
    daily_bull_score = _score_series(
        [
            daily_close >= daily_sma_200,
            daily_close >= daily_sma_long,
            aligned["1d"]["EMA_short"] >= aligned["1d"]["EMA_long"],
            aligned["1d"]["MACD"] >= aligned["1d"]["MACD_signal"],
            aligned["1d"]["ICHIMOKU_cloud_bias"] > 0,
            aligned["1d"]["trend_slope_20"].fillna(0) >= 0,
        ]
    )
    daily_bear_score = _score_series(
        [
            daily_close <= (daily_sma_200 * 0.998),
            daily_close <= daily_sma_long,
            aligned["1d"]["EMA_short"] <= aligned["1d"]["EMA_long"],
            aligned["1d"]["MACD"] <= aligned["1d"]["MACD_signal"],
            aligned["1d"]["ICHIMOKU_cloud_bias"] < 0,
            aligned["1d"]["trend_slope_20"].fillna(0) <= 0,
        ]
    )
    daily_bear_threshold = min(0.92, max(0.78, trend_strength + 0.14))
    macro_bull = (daily_close >= daily_sma_200) & (daily_bull_score >= trend_strength)
    macro_bear = (daily_close <= (daily_sma_200 * 0.998)) & (daily_bear_score >= daily_bear_threshold)

    h4_close = aligned["4h"]["Close"]
    h4_sma_long = aligned["4h"]["SMA_long"].fillna(h4_close)
    h4_vwap = aligned["4h"]["VWAP_rolling"].fillna(h4_close)
    trend_4h_bull_score = _score_series(
        [
            h4_close >= aligned["4h"]["EMA_short"],
            h4_close >= h4_sma_long,
            h4_close >= h4_vwap,
            aligned["4h"]["MACD"] >= aligned["4h"]["MACD_signal"],
            aligned["4h"]["ICHIMOKU_cloud_bias"] > 0,
        ]
    )
    trend_4h_bear_score = _score_series(
        [
            h4_close <= aligned["4h"]["EMA_short"],
            h4_close <= h4_sma_long,
            h4_close <= h4_vwap,
            aligned["4h"]["MACD"] <= aligned["4h"]["MACD_signal"],
            aligned["4h"]["ICHIMOKU_cloud_bias"] < 0,
        ]
    )

    trend_1d = pd.Series(np.where(macro_bull, 1.0, np.where(macro_bear, -1.0, 0.0)), index=base.index)
    trend_4h = pd.Series(
        np.where(
            trend_4h_bull_score >= max(0.55, trend_strength - 0.05),
            1.0,
            np.where(trend_4h_bear_score >= max(0.68, daily_bear_threshold - 0.05), -1.0, 0.0),
        ),
        index=base.index,
    )

    higher_aligned = (trend_4h == trend_1d) & (trend_1d != 0.0)
    higher_long_score = ((0.65 * daily_bull_score) + (0.35 * trend_4h_bull_score)).clip(0.0, 1.0)
    higher_short_score = ((0.70 * daily_bear_score) + (0.30 * trend_4h_bear_score)).clip(0.0, 1.0)

    bull_regime = macro_bull & (trend_4h_bull_score >= 0.45)
    bear_regime = macro_bear & (trend_4h_bear_score >= 0.60)
    range_regime = ~(bull_regime | bear_regime)

    h1_close = aligned["1h"]["Close"]
    h1_vwap = aligned["1h"]["VWAP_rolling"].fillna(h1_close)
    h1_sma_200 = aligned["1h"]["SMA_200"].fillna(aligned["1h"]["SMA_long"]).fillna(h1_close)
    setup_1h_long_score = _score_series(
        [
            h1_close >= aligned["1h"]["EMA_short"],
            h1_close >= h1_vwap,
            aligned["1h"]["MACD"] >= aligned["1h"]["MACD_signal"],
            aligned["1h"]["RSI_fast"] >= aligned["1h"]["RSI"],
            aligned["1h"]["ICHIMOKU_cloud_bias"] >= 0,
        ]
    )
    setup_1h_short_score = _score_series(
        [
            h1_close <= aligned["1h"]["EMA_short"],
            h1_close <= h1_vwap,
            aligned["1h"]["MACD"] <= aligned["1h"]["MACD_signal"],
            aligned["1h"]["RSI_fast"] <= aligned["1h"]["RSI"],
            aligned["1h"]["ICHIMOKU_cloud_bias"] <= 0,
            h1_close <= h1_sma_200,
        ]
    )

    recent_high_1h = aligned["1h"]["High"].rolling(window=breakout_1h_window).max().shift(1)
    recent_low_1h = aligned["1h"]["Low"].rolling(window=breakout_1h_window).min().shift(1)

    m15_close = aligned["15m"]["Close"]
    m15_vwap = aligned["15m"]["VWAP_rolling"].fillna(m15_close)
    recent_high_15m = aligned["15m"]["High"].rolling(window=breakout_15m_window).max().shift(1)
    recent_low_15m = aligned["15m"]["Low"].rolling(window=breakout_15m_window).min().shift(1)

    allow_long = bull_regime & higher_aligned & (higher_long_score >= max(0.62, long_score_threshold - 0.06))
    allow_short = bear_regime & higher_aligned & (daily_bear_score >= max(0.90, short_score_threshold))

    long_breakout_1h = (
        allow_long
        & (h1_close >= recent_high_1h)
        & (aligned["1h"]["volume_surge_ratio"] >= max(1.05, breakout_volume_surge - 0.20))
        & (aligned["1h"]["MACD"] >= aligned["1h"]["MACD_signal"])
        & (h1_close >= h1_vwap)
    ).fillna(False)
    long_breakout_15 = (
        allow_long
        & (m15_close.shift(1) < recent_high_15m.shift(1))
        & (m15_close >= recent_high_15m)
        & (aligned["15m"]["volume_surge_ratio"] >= breakout_volume_surge)
        & (aligned["15m"]["MACD"] >= aligned["15m"]["MACD_signal"])
        & (aligned["15m"]["MACD_hist"] >= aligned["15m"]["MACD_hist"].shift(1))
        & (m15_close >= m15_vwap)
    ).fillna(False)

    short_breakdown_1h = (
        allow_short
        & (h1_close <= recent_low_1h)
        & (aligned["1h"]["volume_surge_ratio"] >= max(1.15, short_breakdown_volume_surge - 0.10))
        & (aligned["1h"]["MACD"] <= aligned["1h"]["MACD_signal"])
        & (h1_close <= h1_vwap)
        & (h1_close <= h1_sma_200)
    ).fillna(False)
    short_breakdown_15 = (
        allow_short
        & (m15_close.shift(1) > recent_low_15m.shift(1))
        & (m15_close <= recent_low_15m)
        & (aligned["15m"]["volume_surge_ratio"] >= short_breakdown_volume_surge)
        & (aligned["15m"]["MACD"] <= aligned["15m"]["MACD_signal"])
        & (aligned["15m"]["MACD_hist"] <= aligned["15m"]["MACD_hist"].shift(1))
        & (m15_close <= m15_vwap)
    ).fillna(False)

    long_event = (long_breakout_1h & long_breakout_15).fillna(False)
    short_event = (short_breakdown_1h & short_breakdown_15).fillna(False)
    long_pass = long_event & ~long_event.shift(1, fill_value=False)
    short_pass = short_event & ~short_event.shift(1, fill_value=False)

    long_score = (
        (0.55 * higher_long_score)
        + (0.15 * setup_1h_long_score)
        + (0.15 * _as_series(long_breakout_1h).astype(float))
        + (0.15 * _as_series(long_breakout_15).astype(float))
    ).clip(0.0, 1.0)
    short_score = (
        (0.60 * higher_short_score)
        + (0.15 * setup_1h_short_score)
        + (0.125 * _as_series(short_breakdown_1h).astype(float))
        + (0.125 * _as_series(short_breakdown_15).astype(float))
    ).clip(0.0, 1.0)

    alignment_scale = pd.Series(np.where(higher_aligned, resonance, max(0.65, conflict_penalty)), index=base.index)
    short_alignment_scale = pd.Series(
        np.where(higher_aligned, min(1.0, resonance * 0.95), max(0.40, conflict_penalty * 0.85)),
        index=base.index,
    )
    long_score = (long_score * alignment_scale).clip(0.0, 1.0)
    short_score = (short_score * short_alignment_scale).clip(0.0, 1.0)

    long_score_arr = long_score.to_numpy(dtype=float)
    short_score_arr = short_score.to_numpy(dtype=float)
    long_threshold = long_score_threshold + (0.03 if entry_timeframe == "1h" else 0.0)
    short_threshold = max(0.92, short_score_threshold) + (0.03 if entry_timeframe == "1h" else 0.0)
    long_active = np.asarray(long_pass, dtype=bool) & (long_score_arr >= long_threshold) & ((long_score_arr - short_score_arr) >= score_hysteresis)
    short_active = np.asarray(short_pass, dtype=bool) & (short_score_arr >= short_threshold) & ((short_score_arr - long_score_arr) >= max(score_hysteresis + 0.04, 0.12))

    signal = np.zeros(len(base), dtype=float)
    signal[long_active] = min(max_pos, 1.0)
    signal[short_active] = -min(max_pos, 1.0)

    bull_regime_arr = bull_regime.to_numpy(dtype=bool)
    bear_regime_arr = bear_regime.to_numpy(dtype=bool)
    long_size_bias = np.where(bull_regime_arr, 1.0, 0.0)
    short_size_bias = np.where(bear_regime_arr, 0.25, 0.0)
    signal = np.where(signal > 0, signal * long_size_bias, np.where(signal < 0, signal * short_size_bias, signal))

    enriched_news = enrich_news_records(news or [])
    cooldown_bars = int(float(strategy.get("cooldown_bars_after_news", 0)))
    news_context = compute_news_context(base.index, enriched_news, cooldown_bars=cooldown_bars)
    impact_series = news_context["impact"].clip(lower=-news_impact_cap, upper=news_impact_cap)
    news_weight = float(strategy.get("news_weight", 0.3))
    aligned_impact = np.sign(signal) * impact_series.to_numpy()
    positive_news_cap = min(0.25, news_impact_cap * 0.35)
    directional_factor = np.ones(len(base), dtype=float)
    directional_factor += np.clip(aligned_impact, 0.0, positive_news_cap) * (news_weight * 0.5)
    directional_factor += np.clip(aligned_impact, -news_impact_cap, 0.0) * news_weight
    directional_factor = np.where(np.abs(impact_series.to_numpy()) >= 0.15, directional_factor, np.minimum(directional_factor, 1.0))
    directional_factor = np.clip(directional_factor, 1.0 - news_impact_cap, 1.0 + positive_news_cap)
    signal = signal * directional_factor
    hard_cooldown = news_context.get("hard_cooldown", news_context["cooldown"]).to_numpy()
    soft_cooldown = news_context.get("soft_cooldown", pd.Series(False, index=base.index, dtype=bool)).to_numpy()
    signal = np.where(soft_cooldown, signal * 0.5, signal)
    signal = np.where(hard_cooldown, 0.0, signal)

    weak_long_trend = higher_long_score.to_numpy(dtype=float) < max(0.62, trend_strength)
    weak_short_trend = higher_short_score.to_numpy(dtype=float) < max(0.88, daily_bear_threshold)
    weak_trend = np.where(signal > 0, weak_long_trend, np.where(signal < 0, weak_short_trend, False))
    weak_trend_penalty = np.where(signal > 0, max(0.82, conflict_penalty), max(0.65, conflict_penalty))
    signal = np.where(weak_trend, signal * weak_trend_penalty, signal)
    eval_start = _parse_datetime(evaluation_start)
    if eval_start is not None:
        signal = np.where(base.index < pd.Timestamp(eval_start), 0.0, signal)
    signal = np.clip(signal, -max_pos, max_pos)
    entry_tf = np.where(signal != 0.0, "1h+15m", "")
    setup_family = np.full(len(base), "no_trade", dtype=object)
    setup_family[signal > 0] = "long_breakout"
    setup_family[signal < 0] = "short_breakdown"

    regime = np.where(bull_regime_arr, "bull", np.where(bear_regime_arr, "bear", "range"))

    base["trend_4h"] = trend_4h.to_numpy(dtype=float)
    base["trend_1d"] = trend_1d.to_numpy(dtype=float)
    base["entry_signal_tf"] = entry_tf
    base["setup_family"] = setup_family
    base["regime"] = regime
    base["long_score"] = long_score_arr
    base["short_score"] = short_score_arr
    base["signal"] = signal
    base["news_impact"] = impact_series.to_numpy()
    base["active_news_count"] = news_context["active_count"].to_numpy()
    base["news_hard_cooldown"] = np.asarray(hard_cooldown, dtype=float)
    base["news_soft_cooldown"] = np.asarray(soft_cooldown, dtype=float)
    base["news_cooldown"] = np.asarray(hard_cooldown, dtype=float)
    returns, position, per_bar_costs, event_notes = _simulate_positions(base, signal=signal, costs=costs, strategy=strategy)
    return _finalize_backtest_result(
        frame=base,
        position=position,
        returns=returns,
        costs=per_bar_costs,
        event_notes=event_notes,
        strategy=strategy,
        news=news or [],
        entry_timeframe=entry_timeframe,
        execution_timeframe=base_tf,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )


def _simulate_positions(
    frame: pd.DataFrame,
    signal: np.ndarray,
    costs: dict[str, float],
    strategy: dict[str, Any],
) -> tuple[pd.Series, pd.Series, list[float], list[str]]:
    atr = pd.to_numeric(frame.get("ATR_pct", _atr_percentage(frame)), errors="coerce").fillna(0).to_numpy()
    close = pd.to_numeric(frame["Close"], errors="coerce").ffill().to_numpy()
    high = pd.to_numeric(frame["High"], errors="coerce").ffill().to_numpy()
    low = pd.to_numeric(frame["Low"], errors="coerce").ffill().to_numpy()
    pct_change = pd.Series(close, index=frame.index).pct_change().fillna(0).to_numpy()
    fee_rate = float(costs["fee_rate"])
    slippage_rate = float(costs["slippage_rate"])
    max_loss = float(costs.get("max_loss_per_trade", 0.005))
    max_pos = float(strategy.get("max_position", 1.0))
    stop_mult = float(strategy.get("stop_atr_multiple", 1.8))
    take_profit_mult = float(strategy.get("take_profit_atr_multiple", 2.8))
    max_hold_bars = max(4, int(round(float(strategy.get("max_hold_bars", 96)))))
    breakeven_r_multiple = float(strategy.get("breakeven_r_multiple", 1.0))
    trade_cooldown_bars = max(0, int(round(float(strategy.get("trade_cooldown_bars", 0)))))

    position: list[float] = []
    returns: list[float] = []
    trade_costs: list[float] = []
    event_notes: list[str] = []
    prev_pos = 0.0
    setup_family = frame.get("setup_family", pd.Series("unknown_setup", index=frame.index))
    active: dict[str, Any] | None = None
    cooldown_remaining = 0

    for i, sig in enumerate(signal):
        desired = float(sig)
        prev_close = close[i - 1] if i > 0 else close[i]
        next_pos = prev_pos
        note = ""
        exited_this_bar = False
        exit_price_override: float | None = None
        atr_pct = max(float(atr[i]) if i < len(atr) and np.isfinite(atr[i]) else 0.0, 1e-6)
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if active is not None and prev_pos != 0.0:
            active["bars_held"] = active.get("bars_held", 0.0) + 1.0
            risk_distance = float(active.get("risk_distance", 0.0))
            if prev_pos > 0:
                if (
                    not bool(active.get("breakeven_armed", False))
                    and risk_distance > 0
                    and high[i] >= float(active["entry_price"]) * (1.0 + (breakeven_r_multiple * risk_distance))
                ):
                    active["stop_price"] = max(float(active["stop_price"]), float(active["entry_price"]))
                    active["breakeven_armed"] = True
                trailing_stop = close[i] * (1.0 - (stop_mult * atr_pct))
                active["stop_price"] = max(float(active["stop_price"]), trailing_stop)
                stop_hit = low[i] <= float(active["stop_price"])
                target_hit = high[i] >= float(active["target_price"])
                trend_invalid = float(frame["trend_4h"].iloc[i]) < 0 or float(frame["trend_1d"].iloc[i]) < 0
            else:
                if (
                    not bool(active.get("breakeven_armed", False))
                    and risk_distance > 0
                    and low[i] <= float(active["entry_price"]) * (1.0 - (breakeven_r_multiple * risk_distance))
                ):
                    active["stop_price"] = min(float(active["stop_price"]), float(active["entry_price"]))
                    active["breakeven_armed"] = True
                trailing_stop = close[i] * (1.0 + (stop_mult * atr_pct))
                active["stop_price"] = min(float(active["stop_price"]), trailing_stop)
                stop_hit = high[i] >= float(active["stop_price"])
                target_hit = low[i] <= float(active["target_price"])
                trend_invalid = float(frame["trend_4h"].iloc[i]) > 0 or float(frame["trend_1d"].iloc[i]) > 0

            if stop_hit:
                next_pos = 0.0
                note = "stop_loss_hit"
                exited_this_bar = True
                exit_price_override = float(active["stop_price"])
                active = None
                cooldown_remaining = max(cooldown_remaining, trade_cooldown_bars)
            elif target_hit:
                next_pos = 0.0
                note = "take_profit_hit"
                exited_this_bar = True
                exit_price_override = float(active["target_price"])
                active = None
                cooldown_remaining = max(cooldown_remaining, max(1, trade_cooldown_bars // 2))
            elif trend_invalid:
                next_pos = 0.0
                note = "higher_timeframe_invalidation"
                exited_this_bar = True
                active = None
                cooldown_remaining = max(cooldown_remaining, trade_cooldown_bars)
            elif desired != 0.0 and np.sign(desired) != np.sign(prev_pos):
                next_pos = 0.0
                note = "signal_flip"
                exited_this_bar = True
                active = None
                cooldown_remaining = max(cooldown_remaining, trade_cooldown_bars)
            elif float(active.get("bars_held", 0.0)) >= float(active.get("max_hold_bars", max_hold_bars)):
                next_pos = 0.0
                note = "max_holding_reached"
                exited_this_bar = True
                active = None
                cooldown_remaining = max(cooldown_remaining, max(1, trade_cooldown_bars // 2))

        if prev_pos == 0.0 and cooldown_remaining == 0 and not exited_this_bar and desired != 0.0:
            unit_risk = max(stop_mult * atr_pct, 1e-6)
            size_cap = max_loss / unit_risk
            size = min(max_pos, abs(desired), size_cap)
            if size > 0:
                next_pos = float(np.sign(desired) * size)
                entry_price = close[i]
                family = str(setup_family.iloc[i]) if i < len(setup_family) else "unknown_setup"
                target_multiplier = take_profit_mult
                family_max_hold = max_hold_bars
                if family == "long_pullback":
                    target_multiplier *= 0.80
                    family_max_hold = max(12, int(round(max_hold_bars * 0.75)))
                elif family == "long_breakout":
                    target_multiplier *= 1.10
                    family_max_hold = max(12, int(round(max_hold_bars * 1.20)))
                elif family == "long_trend_continuation":
                    target_multiplier *= 0.95
                elif family == "short_breakdown":
                    target_multiplier *= 0.85
                    family_max_hold = max(12, int(round(max_hold_bars * 0.90)))
                elif family == "short_trend_continuation":
                    target_multiplier *= 0.70
                    family_max_hold = max(12, int(round(max_hold_bars * 0.70)))
                stop_distance = stop_mult * atr_pct
                target_distance = target_multiplier * atr_pct
                if next_pos > 0:
                    active = {
                        "entry_price": entry_price,
                        "stop_price": entry_price * (1.0 - stop_distance),
                        "target_price": entry_price * (1.0 + target_distance),
                        "risk_distance": stop_distance,
                        "breakeven_armed": False,
                        "bars_held": 0.0,
                        "max_hold_bars": float(family_max_hold),
                        "setup_family": family,
                    }
                else:
                    active = {
                        "entry_price": entry_price,
                        "stop_price": entry_price * (1.0 + stop_distance),
                        "target_price": entry_price * (1.0 - target_distance),
                        "risk_distance": stop_distance,
                        "breakeven_armed": False,
                        "bars_held": 0.0,
                        "max_hold_bars": float(family_max_hold),
                        "setup_family": family,
                    }
                note = "entry_opened"

        turnover = abs(next_pos - prev_pos)
        trade_cost = turnover * (fee_rate + slippage_rate)

        if prev_pos != 0.0 and note in {"stop_loss_hit", "take_profit_hit"}:
            exit_price = exit_price_override if exit_price_override is not None else close[i]
            bar_return = ((exit_price / prev_close) - 1.0) * prev_pos if prev_close else 0.0
        else:
            bar_return = pct_change[i] * prev_pos

        ret = float(bar_return - trade_cost)
        position.append(next_pos)
        returns.append(ret)
        trade_costs.append(trade_cost)
        event_notes.append(note)
        prev_pos = next_pos

    return pd.Series(returns, index=frame.index), pd.Series(position, index=frame.index), trade_costs, event_notes


def _finalize_backtest_result(
    *,
    frame: pd.DataFrame,
    position: pd.Series,
    returns: pd.Series,
    costs: list[float],
    event_notes: list[str],
    strategy: dict[str, Any],
    news: list[dict[str, Any]],
    entry_timeframe: str,
    execution_timeframe: str,
    evaluation_start: datetime | str | None = None,
    evaluation_end: datetime | str | None = None,
) -> dict[str, Any]:
    event_note_series = pd.Series(event_notes, index=frame.index)
    mask = _window_mask(frame.index, evaluation_start=evaluation_start, evaluation_end=evaluation_end)
    frame = frame.loc[mask].copy()
    position = position.loc[mask].copy()
    returns = returns.loc[mask].copy()
    event_note_series = event_note_series.loc[mask].reindex(frame.index, fill_value="")
    costs_array = np.asarray(costs, dtype=float)
    window_costs = costs_array[mask] if len(costs_array) == len(mask) else costs_array

    cum = (1 + returns).cumprod()
    peak = cum.cummax().replace(0, np.nan)
    drawdown = ((peak - cum) / peak).fillna(0)
    bars_per_year = {"15m": 365 * 24 * 4, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}.get(execution_timeframe, 365)
    vol = float(returns.std())
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(bars_per_year)) if vol != 0 else 0.0
    downside = returns[returns < 0].std()
    sortino = float((returns.mean() / downside) * np.sqrt(bars_per_year)) if downside not in (0.0, np.nan) and not math.isnan(float(downside)) else 0.0
    total_return = float(((cum.iloc[-1] - 1.0) * 100) if not cum.empty else 0.0)
    annualized = float(((cum.iloc[-1]) ** (bars_per_year / max(1, len(returns))) - 1) * 100) if not cum.empty else 0.0
    max_dd = float(drawdown.max() * 100) if len(drawdown) else 0.0
    calmar = float((annualized / max_dd) if max_dd > 0 else 0.0)
    trades = _extract_trade_details(
        frame,
        position,
        pd.to_numeric(frame["Close"], errors="coerce"),
        returns,
        strategy,
        news,
        event_note_series,
    )
    trade_returns = np.array([float(trade.get("return_pct", 0.0)) / 100.0 for trade in trades], dtype=float)
    pos_sum = float(trade_returns[trade_returns > 0].sum()) if len(trade_returns) else 0.0
    neg_sum = float(trade_returns[trade_returns < 0].sum()) if len(trade_returns) else 0.0
    profit_factor = float(abs(pos_sum / neg_sum)) if neg_sum != 0 else (float("inf") if pos_sum > 0 else 0.0)
    win_rate = float((trade_returns > 0).mean() * 100) if len(trade_returns) else 0.0
    expectancy = float(trade_returns.mean() * 100) if len(trade_returns) else 0.0
    avg_win = float(trade_returns[trade_returns > 0].mean() * 100) if np.any(trade_returns > 0) else 0.0
    avg_loss = float(trade_returns[trade_returns < 0].mean() * 100) if np.any(trade_returns < 0) else 0.0
    avg_holding_bars = float(np.mean([float(trade.get("bars_held", 0)) for trade in trades])) if trades else 0.0
    regime_breakdown = _regime_breakdown(trades)
    monthly_breakdown = _monthly_breakdown(
        returns,
        drawdown,
        trades,
        execution_timeframe=execution_timeframe,
    )
    long_count = sum(1 for trade in trades if str(trade.get("side", "")).lower().startswith("long"))
    short_count = sum(1 for trade in trades if str(trade.get("side", "")).lower().startswith("short"))
    reliability_flags: list[str] = []
    if len(trades) < 30:
        reliability_flags.append("low_sample_size")
    if len(trades) > 0 and (long_count == 0 or short_count == 0):
        reliability_flags.append("one_sided_exposure")
    if len(trades) < 10 and (win_rate >= 95.0 or win_rate <= 5.0):
        reliability_flags.append("extreme_win_rate_small_sample")
    if math.isinf(profit_factor) and len(trades) < 20:
        reliability_flags.append("profit_factor_unstable")
    if sharpe < 0:
        reliability_flags.append("negative_sharpe")
    if profit_factor < 1.0:
        reliability_flags.append("profit_factor_below_one")
    if total_return < 0:
        reliability_flags.append("negative_total_return")

    narrative = (
        "The strategy uses a slow 1d macro bias with 4h confirmation, then confirms setups on 1h and executes on 15m. "
        "News impact windows modulate risk asymmetrically rather than acting as standalone triggers."
    )
    if reliability_flags:
        narrative += f" Reliability caution: {', '.join(reliability_flags)}."
    return {
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_holding_bars": avg_holding_bars,
        "costs": float(sum(window_costs) * 100),
        "trade_count": len(trades),
        "trades": trades,
        "positions": position.tolist(),
        "returns": returns.tolist(),
        "entry_timeframe": entry_timeframe,
        "execution_timeframe": execution_timeframe,
        "news_event_count": len(news),
        "monthly_breakdown": monthly_breakdown,
        "regime_breakdown": regime_breakdown,
        "reliability_flags": reliability_flags,
        "narrative": narrative,
    }


def compute_news_context(
    index: pd.DatetimeIndex,
    news_events: list[dict[str, Any]],
    *,
    cooldown_bars: int = 0,
) -> dict[str, pd.Series]:
    impact = pd.Series(0.0, index=index, dtype=float)
    active_count = pd.Series(0, index=index, dtype=int)
    hard_cooldown = pd.Series(False, index=index, dtype=bool)
    soft_cooldown = pd.Series(False, index=index, dtype=bool)
    if len(index) == 0:
        return {
            "impact": impact,
            "active_count": active_count,
            "hard_cooldown": hard_cooldown,
            "soft_cooldown": soft_cooldown,
            "cooldown": hard_cooldown,
        }

    for event in news_events:
        start = _parse_datetime(event.get("impact_start") or event.get("published_at"))
        end = _parse_datetime(event.get("impact_end") or event.get("published_at"))
        if start is None or end is None:
            continue
        if end < index.min().to_pydatetime().replace(tzinfo=UTC) or start > index.max().to_pydatetime().replace(tzinfo=UTC):
            continue

        sentiment = float(event.get("sentiment", 0.0))
        strength = float(event.get("sentiment_strength", abs(sentiment)))
        published_conf = str(event.get("published_at_confidence", "unknown")).lower()
        confidence_factor = {"high": 1.0, "medium": 0.8, "unknown": 0.6, "low": 0.35}.get(published_conf, 0.6)
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        impact.loc[mask] += sentiment * max(0.1, strength) * confidence_factor
        active_count.loc[mask] += 1

        if cooldown_bars > 0:
            start_idx = int(index.searchsorted(pd.Timestamp(start), side="left"))
            end_idx = min(len(index), start_idx + cooldown_bars)
            if start_idx < len(index):
                event_type = str(event.get("event_type", "")).lower()
                is_major_event = event_type in {"regulation", "etf", "hack", "liquidation", "macro"}
                if confidence_factor >= 0.75 and strength >= 0.6 and is_major_event:
                    hard_cooldown.iloc[start_idx:end_idx] = True
                elif confidence_factor >= 0.55 and strength >= 0.45:
                    soft_cooldown.iloc[start_idx:end_idx] = True

    return {
        "impact": impact.clip(lower=-1.5, upper=1.5),
        "active_count": active_count,
        "hard_cooldown": hard_cooldown,
        "soft_cooldown": soft_cooldown,
        "cooldown": hard_cooldown,
    }


def compute_news_impact_series(index: pd.DatetimeIndex, news_events: list[dict[str, Any]]) -> pd.Series:
    return compute_news_context(index, news_events)["impact"]


def _window_mask(
    index: pd.DatetimeIndex,
    *,
    evaluation_start: datetime | str | None = None,
    evaluation_end: datetime | str | None = None,
) -> np.ndarray:
    start = _parse_datetime(evaluation_start)
    end = _parse_datetime(evaluation_end)
    mask = np.ones(len(index), dtype=bool)
    if start is not None:
        mask &= index >= pd.Timestamp(start)
    if end is not None:
        mask &= index < pd.Timestamp(end)
    return mask


def _heuristic_sentiment(text: str) -> float:
    positive = ["surge", "rally", "approve", "inflow", "adoption", "gain", "bull", "etf", "institutional"]
    negative = ["ban", "hack", "lawsuit", "outflow", "drop", "bear", "crash", "liquidation", "default"]
    lowered = text.lower()
    score = sum(0.18 for t in positive if t in lowered) - sum(0.18 for t in negative if t in lowered)
    return float(max(-1.0, min(1.0, score)))


def _contains_event_keywords(text: str) -> bool:
    text = text.lower()
    return any(token in text for token in ["etf", "sec", "regulation", "hack", "liquidation", "fed", "cpi"])


def _infer_event_type(text: str) -> str:
    t = text.lower()
    if any(token in t for token in ["hack", "exploit", "drain", "breach"]):
        return "hack"
    if any(token in t for token in ["liquidation", "liquidated"]):
        return "liquidation"
    if any(token in t for token in ["sec", "regulation", "law", "policy"]):
        return "regulation"
    if "etf" in t:
        return "etf"
    if any(token in t for token in ["fed", "cpi", "inflation", "rate hike", "fomc"]):
        return "macro"
    if any(token in t for token in ["mining", "hashrate", "difficulty"]):
        return "mining"
    if any(token in t for token in ["stablecoin", "usdt", "usdc"]):
        return "stablecoin"
    return "other"


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


def _fill_missing_ohlcv_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    expected_index = pd.date_range(df.index.min(), df.index.max(), freq=_pandas_freq(timeframe), tz="UTC")
    out = df.reindex(expected_index)

    close = pd.to_numeric(out.get("Close"), errors="coerce")
    close = close.ffill().bfill()
    out["Close"] = close
    out["Open"] = pd.to_numeric(out.get("Open"), errors="coerce").fillna(close.shift(1)).fillna(close)
    oc_max = pd.concat([out["Open"], out["Close"]], axis=1).max(axis=1)
    oc_min = pd.concat([out["Open"], out["Close"]], axis=1).min(axis=1)
    out["High"] = pd.to_numeric(out.get("High"), errors="coerce").fillna(oc_max)
    out["Low"] = pd.to_numeric(out.get("Low"), errors="coerce").fillna(oc_min)
    out["Volume"] = pd.to_numeric(out.get("Volume"), errors="coerce").fillna(0.0)
    return out.dropna(subset=["Open", "High", "Low", "Close"], how="any")


def _coerce_frame(frame: Any, timeframe: str, strategy: dict[str, Any] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(frame).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], how="any")
    df = _fill_missing_ohlcv_gaps(df, timeframe=timeframe)
    df.index = df.index + pd.to_timedelta(_tf_millis(timeframe), unit="ms")
    return _add_indicators(df, timeframe=timeframe, strategy=strategy)


def _tf_millis(timeframe: str) -> int:
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    num = int(timeframe[:-1])
    unit = timeframe[-1]
    return num * multipliers[unit]


def _pandas_freq(timeframe: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
    return mapping[timeframe]


def _add_indicators(df: pd.DataFrame, timeframe: str, strategy: dict[str, Any] | None = None) -> pd.DataFrame:
    out = df.copy()
    strategy = strategy or {}
    close = pd.to_numeric(out["Close"], errors="coerce")
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    volume = pd.to_numeric(out["Volume"], errors="coerce")

    ema_short = max(2, int(round(float(strategy.get("ema_short", 12)))))
    ema_long = max(ema_short + 1, int(round(float(strategy.get("ema_long", 26)))))
    macd_signal = max(2, int(round(float(strategy.get("macd_signal", 9)))))
    rsi_period = max(2, int(round(float(strategy.get("rsi_period", 14)))))
    rsi_fast_period = max(2, int(round(float(strategy.get("rsi_fast_period", 7)))))
    sma_short = max(2, int(round(float(strategy.get("sma_short", strategy.get("ma_short", 10))))))
    sma_long = max(sma_short + 1, int(round(float(strategy.get("sma_long", strategy.get("ma_long", 50))))))
    bb_period = max(5, int(round(float(strategy.get("bb_period", 20)))))
    bb_std = float(strategy.get("bb_std", 2.0))
    ichimoku_conversion = max(5, int(round(float(strategy.get("ichimoku_conversion_period", 9)))))
    ichimoku_base = max(ichimoku_conversion + 1, int(round(float(strategy.get("ichimoku_base_period", 26)))))
    ichimoku_span_b = max(ichimoku_base + 1, int(round(float(strategy.get("ichimoku_span_b_period", 52)))))
    vwap_window = max(5, int(round(float(strategy.get("vwap_window", 20)))))
    default_vp_window = 96 if timeframe == "15m" else 60 if timeframe == "1h" else 40
    vp_window = max(10, int(round(float(strategy.get("volume_profile_window", default_vp_window)))))

    out["EMA_short"] = close.ewm(span=ema_short, adjust=False).mean()
    out["EMA_long"] = close.ewm(span=ema_long, adjust=False).mean()
    out["MACD"] = out["EMA_short"] - out["EMA_long"]
    out["MACD_signal"] = out["MACD"].ewm(span=macd_signal, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    avg_gain_fast = gain.ewm(alpha=1 / rsi_fast_period, adjust=False, min_periods=rsi_fast_period).mean()
    avg_loss_fast = loss.ewm(alpha=1 / rsi_fast_period, adjust=False, min_periods=rsi_fast_period).mean()
    rs_fast = avg_gain_fast / avg_loss_fast.replace(0, np.nan)
    out["RSI_fast"] = 100 - (100 / (1 + rs_fast))

    out["SMA_10"] = close.rolling(window=10).mean()
    out["SMA_20"] = close.rolling(window=20).mean()
    out["SMA_50"] = close.rolling(window=50).mean()
    out["SMA_200"] = close.rolling(window=200).mean()
    out["SMA_short"] = close.rolling(window=sma_short).mean()
    out["SMA_long"] = close.rolling(window=sma_long).mean()
    out["MA_short"] = out["SMA_short"]
    out["MA_long"] = out["SMA_long"]

    out["BB_mid"] = close.rolling(window=bb_period).mean()
    out["BB_std"] = close.rolling(window=bb_period).std()
    out["BB_upper"] = out["BB_mid"] + bb_std * out["BB_std"]
    out["BB_lower"] = out["BB_mid"] - bb_std * out["BB_std"]
    out["BB_width"] = (out["BB_upper"] - out["BB_lower"]) / out["BB_mid"].replace(0, np.nan)

    # Ichimoku
    conversion = (high.rolling(window=ichimoku_conversion).max() + low.rolling(window=ichimoku_conversion).min()) / 2
    base = (high.rolling(window=ichimoku_base).max() + low.rolling(window=ichimoku_base).min()) / 2
    span_a = ((conversion + base) / 2).shift(ichimoku_base)
    span_b = ((high.rolling(window=ichimoku_span_b).max() + low.rolling(window=ichimoku_span_b).min()) / 2).shift(ichimoku_base)
    lagging = close.shift(ichimoku_base)
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
    out["VWAP_rolling"] = (typical * volume).rolling(window=vwap_window).sum() / volume.replace(0, np.nan).rolling(window=vwap_window).sum()

    # Volume profile approximations
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
        (out["EMA_short"] >= out["EMA_long"]) & (out["SMA_short"] >= out["SMA_long"]),
        1.0,
        -1.0,
    )
    out["ATR_pct"] = _atr_percentage(out)
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
    event_notes: pd.Series | None = None,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    note_series = event_notes if event_notes is not None else pd.Series("", index=df.index)
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
                    "exit_rationale": _build_exit_rationale(row, cur, str(note_series.iloc[idx])),
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
                "news_filter_summary": _news_summary(row),
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
    return (
        f"Regime: {row.get('regime', 'N/A')}, "
        f"Entry workflow: {row.get('entry_signal_tf', 'N/A')}, "
        f"setup={row.get('setup_family', 'N/A')}, "
        f"score(L/S)={_format_decimal(row.get('long_score'))}/{_format_decimal(row.get('short_score'))}, "
        f"RSI={_format_decimal(row.get('RSI'))}, MACD={_format_decimal(row.get('MACD'))}"
    )


def _news_summary(row: pd.Series) -> str:
    active_count = int(row.get("active_news_count", 0) or 0)
    impact = float(row.get("news_impact", 0.0) or 0.0)
    hard_cooldown = bool(row.get("news_hard_cooldown", row.get("news_cooldown", 0)))
    soft_cooldown = bool(row.get("news_soft_cooldown", 0))
    if active_count <= 0 and abs(impact) < 1e-9:
        return "No active news events applied."
    direction = "bullish" if impact > 0 else "bearish" if impact < 0 else "neutral"
    if hard_cooldown:
        cooldown_text = ", hard cooldown active"
    elif soft_cooldown:
        cooldown_text = ", soft cooldown active"
    else:
        cooldown_text = ""
    return f"{active_count} active news events, net {direction} impact {_format_decimal(impact)}{cooldown_text}"


def _build_entry_rationale(row: pd.Series, position_size: float, strategy: dict[str, Any]) -> str:
    side = "long" if position_size > 0 else "short"
    setup_family = row.get("setup_family", "unknown_setup")
    return (
        f"Opened {side} position because the 1d macro bias and 4h confirmation allowed the direction, "
        f"the lower-timeframe workflow ({row.get('entry_signal_tf', 'N/A')}) confirmed {setup_family}, "
        f"and signal strength after news/regime adjustment was {_format_decimal(position_size)}."
    )


def _build_exit_rationale(row: pd.Series, next_position: float, event_note: str = "") -> str:
    event_map = {
        "stop_loss_hit": "Exited because ATR stop-loss was hit.",
        "take_profit_hit": "Exited because ATR take-profit was hit.",
        "higher_timeframe_invalidation": "Exited because 4h/1d regime invalidated the trade.",
        "signal_flip": "Exited because the lower-timeframe signal flipped direction.",
        "signal_neutral": "Exited because the tactical trigger faded after the minimum hold period.",
        "max_holding_reached": "Exited because max holding duration was reached.",
    }
    if event_note in event_map:
        return event_map[event_note]
    if next_position == 0:
        return "Exited because signal returned to neutral."
    return f"Exited because strategy flipped to {'long' if next_position > 0 else 'short'}."


def _monthly_breakdown(
    returns: pd.Series,
    drawdown: pd.Series,
    trades: list[dict[str, Any]],
    *,
    execution_timeframe: str = "1d",
) -> list[dict[str, Any]]:
    if returns.empty:
        return []
    index_shift = pd.to_timedelta(_tf_millis(execution_timeframe), unit="ms")
    aligned_returns = returns.copy()
    aligned_drawdown = drawdown.copy()
    aligned_returns.index = aligned_returns.index - index_shift
    aligned_drawdown.index = aligned_drawdown.index - index_shift

    monthly_returns = aligned_returns.resample("MS").apply(lambda x: float((1 + x).prod() - 1))
    monthly_dd = aligned_drawdown.resample("MS").max()
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
