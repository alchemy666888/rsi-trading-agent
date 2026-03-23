from __future__ import annotations

from btc_self_improve_agent.agent import BTCSelfImprovingAgent
from btc_self_improve_agent.research_agents import DEFAULT_RESEARCH_END, DEFAULT_RESEARCH_START, ResearchCoordinator


def main() -> None:
    agent = BTCSelfImprovingAgent()
    research = ResearchCoordinator(agent.client, model=agent.model)
    research_outputs = research.run(start_date=DEFAULT_RESEARCH_START, end_date=DEFAULT_RESEARCH_END)
    data_summary = research_outputs["data"].analysis.get("summary", "")
    news_summary = research_outputs["news"].analysis.get("summary", "")

    result = agent.run(
        (
            "Design a BTC multi-timeframe strategy using 15m or 1h for entries, with 4h and 1d used for trend and signal confirmation. "
            "Use RSI, MACD, EMA structure, and news sentiment as a regime filter with annual return > 50% and MaxDD < 25%. "
            f"Use the market-data research in {research_outputs['data'].analysis_md_path} and the news research in "
            f"{research_outputs['news'].analysis_md_path}. Market summary: {data_summary} News summary: {news_summary}"
        ),
        epochs=5,
    )
    print("Best result:", result)


if __name__ == "__main__":
    main()
