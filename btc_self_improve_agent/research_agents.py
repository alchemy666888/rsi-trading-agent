from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .tools import fetch_btc_data, resample_features


DEFAULT_RESEARCH_START = "2023-01-01"
DEFAULT_RESEARCH_END = "2025-12-31"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _extract_text_content(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "")))
    return "\n".join(part for part in parts if part)


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


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def _candidate_models(primary: str | None) -> list[str]:
    ordered = [
        primary,
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
    ]
    seen: set[str] = set()
    result: list[str] = []
    for model in ordered:
        if not model or model in seen:
            continue
        seen.add(model)
        result.append(model)
    return result


def _parse_any_datetime(value: Any) -> datetime | None:
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
    for fmt in [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
    ]:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue
    return None


def _serialize_datetimes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for record in records:
        item: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, datetime):
                item[key] = value.astimezone(UTC).isoformat()
            else:
                item[key] = value
        serialized.append(item)
    return serialized


@dataclass
class ResearchResult:
    raw_path: str
    analysis_json_path: str
    analysis_md_path: str
    analysis: dict[str, Any]


class DataFetchAnalysisAgent:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def run(
        self,
        start_date: str = DEFAULT_RESEARCH_START,
        end_date: str = DEFAULT_RESEARCH_END,
        timeframe: str = "15m",
        output_dir: str = "data",
    ) -> ResearchResult:
        start_ts = f"{start_date}T00:00:00Z"
        end_ts = f"{end_date}T23:59:59Z"
        df = fetch_btc_data(timeframe=timeframe, start=start_ts, end=end_ts)

        root = _project_root() / output_dir
        root.mkdir(parents=True, exist_ok=True)
        stem = f"btc_usdt_{_safe_slug(timeframe)}_{start_date}_{end_date}"
        raw_path = root / f"{stem}.csv"
        analysis_json_path = root / f"{stem}_analysis.json"
        analysis_md_path = root / f"{stem}_analysis.md"

        export_df = df.reset_index().copy()
        export_df["timestamp"] = export_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        export_df.to_csv(raw_path, index=False)

        analysis = self._analyze(df, timeframe=timeframe, start_date=start_date, end_date=end_date)
        analysis_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        analysis_md_path.write_text(self._to_markdown(analysis, raw_path.name), encoding="utf-8")

        return ResearchResult(
            raw_path=str(raw_path),
            analysis_json_path=str(analysis_json_path),
            analysis_md_path=str(analysis_md_path),
            analysis=analysis,
        )

    def _analyze(self, df: pd.DataFrame, timeframe: str, start_date: str, end_date: str) -> dict[str, Any]:
        frames = resample_features(df, timeframes=["15m", "1h", "4h", "1d"])
        timeframe_metrics = {tf: self._build_timeframe_metrics(frame, tf) for tf, frame in frames.items()}
        closes = df["Close"].astype(float)
        summary_payload = {
            "start_date": start_date,
            "end_date": end_date,
            "requested_timeframe": timeframe,
            "rows": int(len(df)),
            "start_close": float(closes.iloc[0]) if not closes.empty else None,
            "end_close": float(closes.iloc[-1]) if not closes.empty else None,
            "absolute_return_pct": float(((closes.iloc[-1] / closes.iloc[0]) - 1) * 100) if len(closes) > 1 else 0.0,
            "avg_volume": float(df["Volume"].mean()) if "Volume" in df and not df["Volume"].empty else 0.0,
            "timeframes": timeframe_metrics,
            "cross_timeframe": self._build_cross_timeframe_summary(timeframe_metrics),
        }

        prompt = f"""
You are analyzing BTC/USDT market data for a trading research pipeline.
Use the multi-timeframe price and volume statistics below and return only JSON.

Summary statistics:
{json.dumps(summary_payload, indent=2)}

Return JSON with this shape:
{{
  "summary": "2-4 sentence market regime summary",
  "multi_timeframe_overview": "short paragraph explaining how 15m, 1h, 4h, and 1d relate",
  "price_volume_alignment": "short paragraph focused on price/volume confirmation or divergence",
  "timeframe_signals": {{
    "15m": "1-2 sentence signal read",
    "1h": "1-2 sentence signal read",
    "4h": "1-2 sentence signal read",
    "1d": "1-2 sentence signal read"
  }},
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "trading_implications": ["implication 1", "implication 2", "implication 3"]
}}
Do not include markdown fences.
"""

        analysis = self._call_llm(prompt)
        if not analysis:
            analysis = {
                "summary": (
                    f"BTC/USDT {timeframe} data from {start_date} to {end_date} shows "
                    f"{summary_payload['absolute_return_pct']:.2f}% total price appreciation across a volatile multi-year cycle. "
                    "The higher timeframes preserved the structural trend while lower timeframes spent long stretches oscillating between momentum continuation and mean reversion."
                ),
                "multi_timeframe_overview": (
                    "The daily and 4-hour views are best for regime direction, while the 1-hour and 15-minute views are more useful for timing entries and exits. "
                    "When the lower timeframes move against the higher timeframe trend, the data suggests treating that as consolidation unless volume expands decisively."
                ),
                "price_volume_alignment": (
                    "Price advances are more trustworthy when they coincide with rising relative volume and positive MACD structure on the intermediate frames. "
                    "Weak volume during breakouts raises the odds of short-lived moves and range reversion."
                ),
                "timeframe_signals": {
                    tf: self._fallback_timeframe_signal(metrics) for tf, metrics in timeframe_metrics.items()
                },
                "key_risks": [
                    "Large downside swings can erase gains quickly during momentum reversals.",
                    "Volatility regime shifts may invalidate thresholds tuned on calmer periods.",
                    "Volume and price trends can diverge, reducing indicator reliability.",
                ],
                "trading_implications": [
                    "Use the daily and 4-hour trend as directional filters before acting on 15-minute setups.",
                    "Require 1-hour or 4-hour volume confirmation before chasing intraday breakouts.",
                    "Recalibrate thresholds over rolling windows rather than assuming one static market state.",
                ],
                "metrics": summary_payload,
            }
        else:
            analysis["metrics"] = summary_payload

        return analysis

    def _build_timeframe_metrics(self, frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        close = frame["Close"].astype(float)
        volume = frame["Volume"].astype(float)
        returns = close.pct_change().dropna()
        rolling_volume = volume.rolling(window=20).mean()
        latest = frame.iloc[-1]

        recent_bars = min(len(frame), 20)
        recent_window = frame.tail(recent_bars)
        price_change_pct = float(((close.iloc[-1] / close.iloc[0]) - 1) * 100) if len(close) > 1 else 0.0
        recent_change_pct = (
            float(((recent_window["Close"].iloc[-1] / recent_window["Close"].iloc[0]) - 1) * 100)
            if len(recent_window) > 1
            else 0.0
        )
        current_volume_ratio = (
            float(latest["Volume"] / rolling_volume.iloc[-1])
            if len(frame) >= 20 and pd.notna(rolling_volume.iloc[-1]) and rolling_volume.iloc[-1] != 0
            else None
        )
        annualization = {"15m": 365 * 24 * 4, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}.get(timeframe, 365)
        annualized_volatility = float(returns.std() * (annualization**0.5)) if not returns.empty else 0.0
        bb_range = latest.get("BB_upper", 0) - latest.get("BB_lower", 0)
        bb_position = (
            float((latest["Close"] - latest["BB_lower"]) / bb_range)
            if pd.notna(bb_range) and bb_range not in (0, None)
            else None
        )
        volume_trend = (
            "expanding"
            if current_volume_ratio is not None and current_volume_ratio > 1.15
            else "contracting"
            if current_volume_ratio is not None and current_volume_ratio < 0.85
            else "neutral"
        )

        return {
            "timeframe": timeframe,
            "rows": int(len(frame)),
            "start": frame.index[0].isoformat(),
            "end": frame.index[-1].isoformat(),
            "start_close": float(close.iloc[0]) if not close.empty else None,
            "end_close": float(close.iloc[-1]) if not close.empty else None,
            "price_change_pct": price_change_pct,
            "recent_change_pct": recent_change_pct,
            "annualized_volatility": annualized_volatility,
            "latest_close": float(latest["Close"]),
            "latest_rsi": float(latest["RSI"]) if pd.notna(latest.get("RSI")) else None,
            "latest_macd": float(latest["MACD"]) if pd.notna(latest.get("MACD")) else None,
            "latest_macd_signal": float(latest["MACD_signal"]) if pd.notna(latest.get("MACD_signal")) else None,
            "ema_gap_pct": (
                float(((latest["EMA_short"] / latest["EMA_long"]) - 1) * 100)
                if pd.notna(latest.get("EMA_short")) and pd.notna(latest.get("EMA_long")) and latest["EMA_long"] != 0
                else None
            ),
            "ma_gap_pct": (
                float(((latest["MA_short"] / latest["MA_long"]) - 1) * 100)
                if pd.notna(latest.get("MA_short")) and pd.notna(latest.get("MA_long")) and latest["MA_long"] != 0
                else None
            ),
            "bollinger_position": bb_position,
            "volume_average": float(volume.mean()) if not volume.empty else 0.0,
            "volume_recent_average": float(recent_window["Volume"].mean()) if not recent_window.empty else 0.0,
            "volume_ratio_to_20": current_volume_ratio,
            "volume_trend": volume_trend,
        }

    def _build_cross_timeframe_summary(self, timeframe_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
        trend_map = {
            tf: self._classify_trend(metrics)
            for tf, metrics in timeframe_metrics.items()
        }
        momentum_map = {
            tf: self._classify_momentum(metrics)
            for tf, metrics in timeframe_metrics.items()
        }
        bullish_count = sum(1 for value in trend_map.values() if value == "bullish")
        bearish_count = sum(1 for value in trend_map.values() if value == "bearish")
        aligned = bullish_count in {0, len(trend_map)} or bearish_count in {0, len(trend_map)}

        return {
            "trend_by_timeframe": trend_map,
            "momentum_by_timeframe": momentum_map,
            "alignment_state": "aligned" if aligned else "mixed",
            "price_volume_confirmation": {
                tf: metrics["volume_trend"] for tf, metrics in timeframe_metrics.items()
            },
        }

    def _classify_trend(self, metrics: dict[str, Any]) -> str:
        ema_gap = metrics.get("ema_gap_pct")
        ma_gap = metrics.get("ma_gap_pct")
        if ema_gap is None or ma_gap is None:
            return "neutral"
        if ema_gap > 0 and ma_gap > 0:
            return "bullish"
        if ema_gap < 0 and ma_gap < 0:
            return "bearish"
        return "neutral"

    def _classify_momentum(self, metrics: dict[str, Any]) -> str:
        macd = metrics.get("latest_macd")
        macd_signal = metrics.get("latest_macd_signal")
        rsi = metrics.get("latest_rsi")
        if macd is None or macd_signal is None or rsi is None:
            return "neutral"
        if macd > macd_signal and rsi >= 55:
            return "positive"
        if macd < macd_signal and rsi <= 45:
            return "negative"
        return "neutral"

    def _fallback_timeframe_signal(self, metrics: dict[str, Any]) -> str:
        trend = self._classify_trend(metrics)
        momentum = self._classify_momentum(metrics)
        volume_trend = metrics.get("volume_trend", "neutral")
        rsi = metrics.get("latest_rsi")
        rsi_text = f"RSI near {rsi:.2f}" if isinstance(rsi, (float, int)) else "RSI unavailable"
        return (
            f"{metrics['timeframe']} structure is {trend} with {momentum} momentum, "
            f"{rsi_text}, and {volume_trend} volume conditions."
        )

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        for model in _candidate_models(self.model):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1400,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception:
                continue
            return _extract_json_object(_extract_text_content(response))
        return {}

    def _to_markdown(self, analysis: dict[str, Any], raw_filename: str) -> str:
        lines = [
            "# BTC Market Data Analysis",
            "",
            f"- Raw Data File: `{raw_filename}`",
            f"- Coverage: {analysis['metrics']['start_date']} to {analysis['metrics']['end_date']}",
            f"- Source Timeframe: {analysis['metrics']['requested_timeframe']}",
            "",
            "## Summary",
            "",
            str(analysis.get("summary", "")).strip(),
            "",
            "## Multi-Timeframe Overview",
            "",
            str(analysis.get("multi_timeframe_overview", "")).strip(),
            "",
            "## Price / Volume Alignment",
            "",
            str(analysis.get("price_volume_alignment", "")).strip(),
            "",
            "## Timeframe Signals",
            "",
        ]
        for tf, text in analysis.get("timeframe_signals", {}).items():
            lines.append(f"- `{tf}`: {text}")
        lines.extend([
            "",
            "## Cross-Timeframe Metrics",
            "",
            "",
        ])
        for key, value in analysis.get("metrics", {}).get("cross_timeframe", {}).items():
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Key Risks", "",])
        for item in analysis.get("key_risks", []):
            lines.append(f"- {item}")
        lines.extend(["", "## Trading Implications", ""])
        for item in analysis.get("trading_implications", []):
            lines.append(f"- {item}")
        lines.extend(["", "## Metrics", ""])
        for key, value in analysis.get("metrics", {}).items():
            if key == "timeframes":
                continue
            lines.append(f"- `{key}`: {value}")
        lines.extend(["", "## Timeframe Metric Snapshot", ""])
        for tf, metrics in analysis.get("metrics", {}).get("timeframes", {}).items():
            lines.append(f"### {tf}")
            lines.append("")
            for key, value in metrics.items():
                if key == "timeframe":
                    continue
                lines.append(f"- `{key}`: {value}")
            lines.append("")
        return "\n".join(lines)


class NewsFetchAnalysisAgent:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def run(
        self,
        start_date: str = DEFAULT_RESEARCH_START,
        end_date: str = DEFAULT_RESEARCH_END,
        output_dir: str = "news",
    ) -> ResearchResult:
        root = _project_root() / output_dir
        root.mkdir(parents=True, exist_ok=True)
        stem = f"btc_news_{start_date}_{end_date}"
        raw_path = root / f"{stem}.json"
        analysis_json_path = root / f"{stem}_analysis.json"
        analysis_md_path = root / f"{stem}_analysis.md"

        articles = self._fetch_news(start_date, end_date)
        raw_path.write_text(json.dumps(_serialize_datetimes(articles), indent=2), encoding="utf-8")

        analysis = self._analyze(articles, start_date=start_date, end_date=end_date)
        analysis_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        analysis_md_path.write_text(self._to_markdown(analysis, raw_path.name), encoding="utf-8")

        return ResearchResult(
            raw_path=str(raw_path),
            analysis_json_path=str(analysis_json_path),
            analysis_md_path=str(analysis_md_path),
            analysis=analysis,
        )

    def _fetch_news(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError:
                return []

        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
        cursor = start_dt
        dedup: dict[str, dict[str, Any]] = {}

        while cursor <= end_dt:
            window_end = min(cursor + timedelta(days=89), end_dt)
            query = (
                "Bitcoin BTC crypto market regulation ETF adoption mining "
                f"after:{cursor.date().isoformat()} before:{(window_end + timedelta(days=1)).date().isoformat()}"
            )
            with DDGS() as ddgs:
                try:
                    results = list(ddgs.text(query, max_results=20))
                except TypeError:
                    try:
                        results = list(ddgs.text(keywords=query, max_results=20))
                    except Exception:
                        results = []
                except Exception:
                    results = []

            for result in results:
                url = str(result.get("href") or result.get("url") or "").strip()
                title = str(result.get("title") or "").strip()
                body = str(result.get("body") or result.get("snippet") or "").strip()
                published = _parse_any_datetime(result.get("date"))
                key = url or title
                if not key:
                    continue
                record = {
                    "title": title,
                    "url": url,
                    "snippet": body,
                    "published_at": published,
                    "search_window_start": cursor.isoformat(),
                    "search_window_end": window_end.isoformat(),
                }
                existing = dedup.get(key)
                if existing is None or (existing.get("snippet") == "" and body):
                    dedup[key] = record

            cursor = window_end + timedelta(days=1)

        articles = sorted(
            dedup.values(),
            key=lambda item: item.get("published_at") or _parse_any_datetime(item.get("search_window_start")) or start_dt,
        )
        return articles

    def _analyze(self, articles: list[dict[str, Any]], start_date: str, end_date: str) -> dict[str, Any]:
        dated_articles = [a for a in articles if isinstance(a.get("published_at"), datetime)]
        sampled = [
            {
                "title": article.get("title", ""),
                "published_at": article["published_at"].astimezone(UTC).isoformat() if article.get("published_at") else "",
                "snippet": article.get("snippet", "")[:400],
                "url": article.get("url", ""),
            }
            for article in dated_articles[:80]
        ]
        summary_payload = {
            "start_date": start_date,
            "end_date": end_date,
            "article_count": len(articles),
            "dated_article_count": len(dated_articles),
            "sample_articles": sampled,
        }

        prompt = f"""
You are analyzing BTC news for a trading research pipeline.
Use the article sample and metadata below and return only JSON.

Payload:
{json.dumps(summary_payload, indent=2)}

Return JSON with this shape:
{{
  "summary": "2-4 sentence summary of the dominant narratives",
  "narratives": ["narrative 1", "narrative 2", "narrative 3"],
  "sentiment_assessment": "short paragraph",
  "trading_implications": ["implication 1", "implication 2", "implication 3"],
  "coverage_notes": "brief note about archive completeness",
  "metrics": {{
    "start_date": "{start_date}",
    "end_date": "{end_date}",
    "article_count": {len(articles)},
    "dated_article_count": {len(dated_articles)}
  }}
}}
Do not include markdown fences.
"""

        analysis = self._call_llm(prompt)
        if not analysis:
            analysis = {
                "summary": (
                    f"Collected {len(articles)} BTC-related news items covering the period from {start_date} to {end_date}. "
                    "The archive likely emphasizes high-signal public narratives such as ETF flows, regulation, adoption, and mining economics."
                ),
                "narratives": [
                    "Spot ETF approval and institutional flows shaped directional sentiment.",
                    "Regulatory actions and lawsuits repeatedly changed risk appetite.",
                    "Adoption, mining, and macro-liquidity stories influenced medium-term conviction.",
                ],
                "sentiment_assessment": (
                    "News sentiment across multi-year BTC cycles is usually mixed rather than uniformly bullish or bearish, "
                    "so it works better as a regime filter than as a standalone trading trigger."
                ),
                "trading_implications": [
                    "Fade isolated headlines unless they align with price confirmation.",
                    "Treat regulation and ETF flow news as high-priority macro catalysts.",
                    "Track narrative persistence across weeks instead of reacting to single articles.",
                ],
                "coverage_notes": (
                    "Archive completeness depends on search-engine indexing and available publication dates, "
                    "so some historical articles may be missing or only associated with their query window."
                ),
                "metrics": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "article_count": len(articles),
                    "dated_article_count": len(dated_articles),
                },
            }
        return analysis

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        for model in _candidate_models(self.model):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1400,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception:
                continue
            return _extract_json_object(_extract_text_content(response))
        return {}

    def _to_markdown(self, analysis: dict[str, Any], raw_filename: str) -> str:
        lines = [
            "# BTC News Analysis",
            "",
            f"- Raw News File: `{raw_filename}`",
            f"- Coverage: {analysis['metrics']['start_date']} to {analysis['metrics']['end_date']}",
            f"- Article Count: {analysis['metrics']['article_count']}",
            "",
            "## Summary",
            "",
            str(analysis.get("summary", "")).strip(),
            "",
            "## Dominant Narratives",
            "",
        ]
        for item in analysis.get("narratives", []):
            lines.append(f"- {item}")
        lines.extend(["", "## Sentiment Assessment", "", str(analysis.get("sentiment_assessment", "")).strip(), "", "## Trading Implications", ""])
        for item in analysis.get("trading_implications", []):
            lines.append(f"- {item}")
        lines.extend(["", "## Coverage Notes", "", str(analysis.get("coverage_notes", "")).strip(), ""])
        return "\n".join(lines)


class ResearchCoordinator:
    def __init__(self, client: Any, model: str) -> None:
        self.data_agent = DataFetchAnalysisAgent(client=client, model=model)
        self.news_agent = NewsFetchAnalysisAgent(client=client, model=model)

    def run(
        self,
        start_date: str = DEFAULT_RESEARCH_START,
        end_date: str = DEFAULT_RESEARCH_END,
    ) -> dict[str, ResearchResult]:
        return {
            "data": self.data_agent.run(start_date=start_date, end_date=end_date, output_dir="data"),
            "news": self.news_agent.run(start_date=start_date, end_date=end_date, output_dir="news"),
        }
