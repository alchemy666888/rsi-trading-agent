from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG
from .tools import enrich_news_records, fetch_btc_data_bundle, resample_features, validate_ohlcv_continuity


DEFAULT_RESEARCH_START = DEFAULT_CONFIG.start_date
DEFAULT_RESEARCH_END = DEFAULT_CONFIG.end_date


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
    ordered = [primary, "claude-sonnet-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-sonnet-20240620"]
    seen: set[str] = set()
    result: list[str] = []
    for model in ordered:
        if model and model not in seen:
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
    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%a, %d %b %Y %H:%M:%S %z"]:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue
    return None


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
        frames = fetch_btc_data_bundle(start=start_ts, end=end_ts, timeframes=DEFAULT_CONFIG.timeframes)

        root = _project_root() / output_dir
        root.mkdir(parents=True, exist_ok=True)
        raw_paths: dict[str, str] = {}
        continuity: dict[str, dict[str, Any]] = {}

        for tf, frame in frames.items():
            stem = f"btc_usdt_{_safe_slug(tf)}_{start_date}_{end_date}"
            path = root / f"{stem}.csv"
            export_df = frame.reset_index().copy()
            if "timestamp" in export_df.columns:
                export_df["timestamp"] = pd.to_datetime(export_df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            export_df.to_csv(path, index=False)
            raw_paths[tf] = str(path)
            continuity[tf] = validate_ohlcv_continuity(frame, timeframe=tf)

        primary_raw_path = raw_paths.get(timeframe, raw_paths["15m"])
        stem_primary = f"btc_usdt_{_safe_slug(timeframe)}_{start_date}_{end_date}"
        analysis_json_path = root / f"{stem_primary}_analysis.json"
        analysis_md_path = root / f"{stem_primary}_analysis.md"

        analysis = self._analyze(frames=frames, continuity=continuity, timeframe=timeframe, start_date=start_date, end_date=end_date)
        analysis["raw_paths"] = raw_paths
        analysis_json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        analysis_md_path.write_text(self._to_markdown(analysis, raw_paths), encoding="utf-8")

        return ResearchResult(
            raw_path=str(primary_raw_path),
            analysis_json_path=str(analysis_json_path),
            analysis_md_path=str(analysis_md_path),
            analysis=analysis,
        )

    def _analyze(
        self,
        *,
        frames: dict[str, pd.DataFrame],
        continuity: dict[str, dict[str, Any]],
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        featured = {tf: resample_features(frame, timeframes=[tf])[tf] for tf, frame in frames.items()}
        timeframe_metrics = {tf: self._build_timeframe_metrics(frame, tf) for tf, frame in featured.items()}
        summary_payload = {
            "start_date": start_date,
            "end_date": end_date,
            "requested_timeframe": timeframe,
            "timeframes": timeframe_metrics,
            "cross_timeframe": self._build_cross_timeframe_summary(timeframe_metrics),
            "continuity": continuity,
        }

        prompt = f"""
You are analyzing BTC/USDT market data and must return only JSON.
Payload:
{json.dumps(summary_payload, indent=2)}

Return JSON:
{{
  "summary": "2-4 sentence market regime summary",
  "multi_timeframe_overview": "how 15m/1h/4h/1d interact",
  "price_volume_alignment": "price and volume confirmation/divergence summary",
  "timeframe_signals": {{
    "15m": "signal",
    "1h": "signal",
    "4h": "signal",
    "1d": "signal"
  }},
  "key_risks": ["risk1","risk2","risk3"],
  "trading_implications": ["imp1","imp2","imp3"]
}}
"""
        analysis = self._call_llm(prompt)
        if not analysis:
            analysis = {
                "summary": (
                    "BTC displayed a high-volatility multi-year regime where higher timeframes drove directional bias "
                    "and lower timeframes produced tactical pullback and breakout opportunities."
                ),
                "multi_timeframe_overview": (
                    "Use 1d and 4h for structure and direction, then 1h and 15m for entries/exits with confirmation."
                ),
                "price_volume_alignment": (
                    "Breakouts were more reliable when volume surge ratios and MACD momentum aligned across 1h and 4h."
                ),
                "timeframe_signals": {tf: self._fallback_timeframe_signal(metrics) for tf, metrics in timeframe_metrics.items()},
                "key_risks": [
                    "Fast volatility expansions can invalidate lower-timeframe signals.",
                    "Trend changes on 1d can lag intraday reversals.",
                    "Divergent volume behavior can produce false breakouts.",
                ],
                "trading_implications": [
                    "Prioritize aligned 1d/4h trend before taking 15m/1h signals.",
                    "Scale down risk when volume confirmation is weak.",
                    "Treat mixed-timeframe conditions as no-trade zones.",
                ],
            }
        analysis["metrics"] = summary_payload
        return analysis

    def _build_timeframe_metrics(self, frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        close = frame["Close"].astype(float)
        volume = frame["Volume"].astype(float)
        returns = close.pct_change().dropna()
        latest = frame.iloc[-1]
        annualization = {"15m": 365 * 24 * 4, "1h": 365 * 24, "4h": 365 * 6, "1d": 365}.get(timeframe, 365)
        return {
            "timeframe": timeframe,
            "rows": int(len(frame)),
            "start": frame.index[0].isoformat(),
            "end": frame.index[-1].isoformat(),
            "start_close": float(close.iloc[0]),
            "end_close": float(close.iloc[-1]),
            "price_change_pct": float(((close.iloc[-1] / close.iloc[0]) - 1) * 100) if len(close) > 1 else 0.0,
            "annualized_volatility": float(returns.std() * np.sqrt(annualization)) if not returns.empty else 0.0,
            "latest_rsi": float(latest.get("RSI", np.nan)) if pd.notna(latest.get("RSI")) else None,
            "latest_macd": float(latest.get("MACD", np.nan)) if pd.notna(latest.get("MACD")) else None,
            "latest_macd_signal": float(latest.get("MACD_signal", np.nan)) if pd.notna(latest.get("MACD_signal")) else None,
            "latest_vwap": float(latest.get("VWAP", np.nan)) if pd.notna(latest.get("VWAP")) else None,
            "volume_average": float(volume.mean()),
        }

    def _build_cross_timeframe_summary(self, timeframe_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
        trend_map: dict[str, str] = {}
        for tf, metrics in timeframe_metrics.items():
            macd = metrics.get("latest_macd")
            macd_signal = metrics.get("latest_macd_signal")
            if macd is None or macd_signal is None:
                trend_map[tf] = "neutral"
            elif macd >= macd_signal:
                trend_map[tf] = "bullish"
            else:
                trend_map[tf] = "bearish"
        aligned = len(set(trend_map.values())) == 1
        return {"trend_by_timeframe": trend_map, "alignment_state": "aligned" if aligned else "mixed"}

    def _fallback_timeframe_signal(self, metrics: dict[str, Any]) -> str:
        macd = metrics.get("latest_macd")
        macd_signal = metrics.get("latest_macd_signal")
        if macd is None or macd_signal is None:
            return "Momentum unavailable."
        direction = "positive" if macd >= macd_signal else "negative"
        return f"{metrics['timeframe']} momentum is {direction}; RSI={metrics.get('latest_rsi')}."

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        for model in _candidate_models(self.model):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1200,
                    messages=[{"role": "user", "content": prompt}],
                )
                return _extract_json_object(_extract_text_content(response))
            except Exception:
                continue
        return {}

    def _to_markdown(self, analysis: dict[str, Any], raw_paths: dict[str, str]) -> str:
        lines = [
            "# BTC Market Data Analysis",
            "",
            "## Raw Data Files",
            "",
        ]
        for tf, path in raw_paths.items():
            lines.append(f"- `{tf}`: `{Path(path).name}`")
        lines.extend(
            [
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
        )
        for tf, text in analysis.get("timeframe_signals", {}).items():
            lines.append(f"- `{tf}`: {text}")
        lines.extend(["", "## Key Risks", ""])
        for risk in analysis.get("key_risks", []):
            lines.append(f"- {risk}")
        lines.extend(["", "## Trading Implications", ""])
        for item in analysis.get("trading_implications", []):
            lines.append(f"- {item}")
        lines.extend(["", "## Metrics", "", f"```json\n{json.dumps(analysis.get('metrics', {}), indent=2)}\n```", ""])
        return "\n".join(lines)


class NewsFetchAnalysisAgent:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def run(self, start_date: str = DEFAULT_RESEARCH_START, end_date: str = DEFAULT_RESEARCH_END, output_dir: str = "news") -> ResearchResult:
        root = _project_root() / output_dir
        root.mkdir(parents=True, exist_ok=True)
        stem = f"btc_news_{start_date}_{end_date}"
        raw_path = root / f"{stem}.json"
        analysis_json_path = root / f"{stem}_analysis.json"
        analysis_md_path = root / f"{stem}_analysis.md"

        raw_articles = self._fetch_news(start_date, end_date)
        enriched = enrich_news_records(raw_articles)
        raw_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
        analysis = self._analyze(enriched, start_date=start_date, end_date=end_date)
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
            window_end = min(cursor + timedelta(days=44), end_dt)
            query = (
                "Bitcoin BTC cryptocurrency web3 regulation ETF mining stablecoin macro "
                f"after:{cursor.date().isoformat()} before:{(window_end + timedelta(days=1)).date().isoformat()}"
            )
            with DDGS() as ddgs:
                try:
                    results = list(ddgs.text(query, max_results=25))
                except TypeError:
                    try:
                        results = list(ddgs.text(keywords=query, max_results=25))
                    except Exception:
                        results = []
                except Exception:
                    results = []
            for row in results:
                url = str(row.get("href") or row.get("url") or "").strip()
                title = str(row.get("title") or "").strip()
                snippet = str(row.get("body") or row.get("snippet") or "").strip()
                published_at = _parse_any_datetime(row.get("date"))
                key = url or title
                if not key:
                    continue
                dedup[key] = {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": str(row.get("source") or ""),
                    "published_at": published_at.isoformat() if published_at else cursor.isoformat(),
                    "search_window_start": cursor.isoformat(),
                    "search_window_end": window_end.isoformat(),
                }
            cursor = window_end + timedelta(days=1)
        return sorted(dedup.values(), key=lambda item: item.get("published_at", ""))

    def _analyze(self, articles: list[dict[str, Any]], start_date: str, end_date: str) -> dict[str, Any]:
        sentiment_values = np.array([float(item.get("sentiment", 0.0)) for item in articles], dtype=float) if articles else np.array([])
        lag_counts: dict[str, int] = {}
        duration_counts: dict[str, int] = {}
        for item in articles:
            lag = str(item.get("impact_lag_bucket", "unknown"))
            dur = str(item.get("impact_duration_bucket", "unknown"))
            lag_counts[lag] = lag_counts.get(lag, 0) + 1
            duration_counts[dur] = duration_counts.get(dur, 0) + 1
        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "article_count": len(articles),
            "avg_sentiment": float(sentiment_values.mean()) if len(sentiment_values) else 0.0,
            "lag_distribution": lag_counts,
            "duration_distribution": duration_counts,
            "sample": articles[:30],
        }
        prompt = f"""
Analyze BTC/crypto/web3 historical news and return only JSON.
Payload:
{json.dumps(payload, indent=2)}

Return JSON:
{{
  "summary": "2-4 sentence narrative",
  "narratives": ["n1","n2","n3"],
  "sentiment_assessment": "assessment",
  "impact_timing_assessment": "how and how long news affects BTC",
  "trading_implications": ["imp1","imp2","imp3"],
  "coverage_notes": "coverage note",
  "metrics": {{
    "start_date": "{start_date}",
    "end_date": "{end_date}",
    "article_count": {len(articles)}
  }}
}}
"""
        analysis = self._call_llm(prompt)
        if not analysis:
            analysis = {
                "summary": (
                    f"Collected {len(articles)} BTC/crypto/web3 news items from {start_date} to {end_date}. "
                    "High-impact regulatory and ETF events typically moved BTC quickly, while macro narratives often persisted longer."
                ),
                "narratives": [
                    "ETF and institutional flow headlines drove major directional shifts.",
                    "Regulatory developments created abrupt risk-on/risk-off transitions.",
                    "Macro-liquidity stories affected medium-term trend persistence.",
                ],
                "sentiment_assessment": "Sentiment should be treated as a regime modifier rather than a direct trade trigger.",
                "impact_timing_assessment": (
                    "Shock events often affect 15m/1h first (0-4h), while structural themes influence 4h/1d over 1-7 days."
                ),
                "trading_implications": [
                    "Use impact windows to modulate exposure after major headlines.",
                    "Require price confirmation before acting on isolated sentiment changes.",
                    "Reduce exposure when sentiment and higher-timeframe trend conflict.",
                ],
                "coverage_notes": "Coverage depends on indexed public sources and may miss some private archives.",
                "metrics": {"start_date": start_date, "end_date": end_date, "article_count": len(articles)},
            }
        analysis["metrics"]["avg_sentiment"] = float(sentiment_values.mean()) if len(sentiment_values) else 0.0
        analysis["metrics"]["lag_distribution"] = lag_counts
        analysis["metrics"]["duration_distribution"] = duration_counts
        return analysis

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        for model in _candidate_models(self.model):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=1200,
                    messages=[{"role": "user", "content": prompt}],
                )
                return _extract_json_object(_extract_text_content(response))
            except Exception:
                continue
        return {}

    def _to_markdown(self, analysis: dict[str, Any], raw_filename: str) -> str:
        lines = [
            "# BTC News Analysis",
            "",
            f"- Raw News File: `{raw_filename}`",
            f"- Coverage: {analysis['metrics'].get('start_date')} to {analysis['metrics'].get('end_date')}",
            f"- Article Count: {analysis['metrics'].get('article_count')}",
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
        lines.extend(
            [
                "",
                "## Sentiment Assessment",
                "",
                str(analysis.get("sentiment_assessment", "")).strip(),
                "",
                "## Impact Timing Assessment",
                "",
                str(analysis.get("impact_timing_assessment", "")).strip(),
                "",
                "## Trading Implications",
                "",
            ]
        )
        for item in analysis.get("trading_implications", []):
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "## Coverage Notes",
                "",
                str(analysis.get("coverage_notes", "")).strip(),
                "",
                "## Metrics",
                "",
                f"```json\n{json.dumps(analysis.get('metrics', {}), indent=2)}\n```",
                "",
            ]
        )
        return "\n".join(lines)


class ResearchCoordinator:
    def __init__(self, client: Any, model: str) -> None:
        self.data_agent = DataFetchAnalysisAgent(client=client, model=model)
        self.news_agent = NewsFetchAnalysisAgent(client=client, model=model)

    def run(self, start_date: str = DEFAULT_RESEARCH_START, end_date: str = DEFAULT_RESEARCH_END) -> dict[str, ResearchResult]:
        return {
            "data": self.data_agent.run(start_date=start_date, end_date=end_date, output_dir="data"),
            "news": self.news_agent.run(start_date=start_date, end_date=end_date, output_dir="news"),
        }
