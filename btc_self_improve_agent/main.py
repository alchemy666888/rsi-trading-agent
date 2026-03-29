from __future__ import annotations

from btc_self_improve_agent.agent import BTCSelfImprovingAgent
from btc_self_improve_agent.config import DEFAULT_CONFIG
from btc_self_improve_agent.observability import write_run_manifest
from btc_self_improve_agent.research_agents import DEFAULT_RESEARCH_END, DEFAULT_RESEARCH_START, ResearchCoordinator


def main() -> None:
    agent = BTCSelfImprovingAgent()
    research = ResearchCoordinator(agent.client, model=agent.model)
    research_outputs = research.run(start_date=DEFAULT_RESEARCH_START, end_date=DEFAULT_RESEARCH_END)

    data_analysis = research_outputs["data"].analysis
    news_analysis = research_outputs["news"].analysis
    raw_paths = data_analysis.get("raw_paths", {})
    result = agent.run(
        (
            "Design a recurring self-improving BTC strategy using 4h and 1d for market trend and regime context, "
            "and 15m and 1h for entries/exits. Integrate MA/EMA/SMA, MACD, RSI, Bollinger Bands, Ichimoku, VWAP, "
            "Volume Profile, and enriched historical BTC/crypto/web3 news impact timing from 2023-01-01 to 2025-12-31. "
            "Generate monthly lessons, final strategy markdown, and a final frozen backtest report."
        ),
        market_csv_paths=raw_paths,
        news_json_path=research_outputs["news"].raw_path,
    )

    manifest_path = write_run_manifest(
        {
            "symbol": DEFAULT_CONFIG.symbol_compact,
            "start_date": DEFAULT_CONFIG.start_date,
            "end_date": DEFAULT_CONFIG.end_date,
            "timeframes": list(DEFAULT_CONFIG.timeframes),
            "data_analysis_path": research_outputs["data"].analysis_md_path,
            "news_analysis_path": research_outputs["news"].analysis_md_path,
            "final_strategy_markdown_path": result.get("final_strategy_markdown_path"),
            "final_backtest_report_path": result.get("final_backtest_report_path"),
            "model": agent.model,
            "market_summary": data_analysis.get("summary", ""),
            "news_summary": news_analysis.get("summary", ""),
        }
    )
    print("Run manifest:", manifest_path)
    print("Final result:", result)


if __name__ == "__main__":
    main()
