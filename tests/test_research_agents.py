import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from btc_self_improve_agent import research_agents
from btc_self_improve_agent.research_agents import DataFetchAnalysisAgent, NewsFetchAnalysisAgent


class _FakeResponseText:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeResponseText(text)]


class _FakeMessages:
    def __init__(self, text: str):
        self._text = text

    def create(self, **_: object) -> _FakeResponse:
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text: str):
        self.messages = _FakeMessages(text)


class ResearchAgentsTest(unittest.TestCase):
    def test_data_agent_writes_raw_and_analysis_files(self) -> None:
        idx = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [100.0, 102.0, 101.0, 104.0, 106.0],
                "Volume": [10.0, 11.0, 12.0, 13.0, 14.0],
            },
            index=idx,
        )
        frame.index.name = "timestamp"
        llm_json = json.dumps(
            {
                "summary": "Market summary.",
                "trend_assessment": "Trend assessment.",
                "key_risks": ["Risk A"],
                "trading_implications": ["Implication A"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(research_agents, "_project_root", return_value=Path(tmpdir)):
                with patch.object(research_agents, "fetch_btc_data", return_value=frame):
                    result = DataFetchAnalysisAgent(client=_FakeClient(llm_json), model="fake-model").run()

            self.assertTrue(Path(result.raw_path).exists())
            self.assertTrue(Path(result.analysis_json_path).exists())
            self.assertTrue(Path(result.analysis_md_path).exists())
            payload = json.loads(Path(result.analysis_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"], "Market summary.")

    def test_news_agent_writes_raw_and_analysis_files(self) -> None:
        llm_json = json.dumps(
            {
                "summary": "News summary.",
                "narratives": ["ETF flows"],
                "sentiment_assessment": "Mixed.",
                "trading_implications": ["Use as filter."],
                "coverage_notes": "Best effort.",
                "metrics": {
                    "start_date": "2023-01-01",
                    "end_date": "2025-12-31",
                    "article_count": 1,
                    "dated_article_count": 1,
                },
            }
        )
        articles = [
            {
                "title": "Bitcoin ETF demand rises",
                "url": "https://example.com/article",
                "snippet": "Institutional demand improves.",
                "published_at": pd.Timestamp("2024-01-10T00:00:00Z").to_pydatetime(),
                "search_window_start": "2024-01-01T00:00:00+00:00",
                "search_window_end": "2024-03-30T00:00:00+00:00",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(research_agents, "_project_root", return_value=Path(tmpdir)):
                with patch.object(NewsFetchAnalysisAgent, "_fetch_news", return_value=articles):
                    result = NewsFetchAnalysisAgent(client=_FakeClient(llm_json), model="fake-model").run()

            self.assertTrue(Path(result.raw_path).exists())
            self.assertTrue(Path(result.analysis_json_path).exists())
            self.assertTrue(Path(result.analysis_md_path).exists())
            payload = json.loads(Path(result.analysis_json_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"], "News summary.")
