# RSI Trading Agent (BTC Self-Improving Agent)

This project runs a recurring self-improving BTC paper-trading pipeline aligned to:

- `docs/developing-rsi-trading-ai-agents.md`
- `docs/vibe-coding-development-plan-v3.md`

The runtime now follows a monthly walk-forward loop from `2023-01-01` to `2025-12-31`:

1. fetch Binance BTCUSDT historical OHLCV for `15m`, `1h`, `4h`, and `1d`
2. pre-calculate indicators and multi-timeframe features
3. ingest and enrich BTC/crypto/web3 historical news with sentiment and impact windows
4. paper trade each month using `4h/1d` trend context with `15m/1h` entries/exits
5. reflect, learn lessons, and update strategy for the next month
6. generate a final strategy markdown and a final frozen backtest report

## Prerequisites

- Python 3.12+
- An Anthropic API key (Claude)

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
ANTHROPIC_API_KEY=your_api_key_here
# Optional: override default model
CLAUDE_MODEL=claude-3-5-sonnet-20240620
EOF
```

## Run the project

Run the main entrypoint:

```bash
python -m btc_self_improve_agent.main
```

The script performs research and monthly self-improvement, then writes artifacts including:

- `data/btc_usdt_15m_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_1h_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_4h_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_1d_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_15m_2023-01-01_2025-12-31_analysis.md`
- `news/btc_news_2023-01-01_2025-12-31.json`
- `news/btc_news_2023-01-01_2025-12-31_analysis.md`
- `backtest/monthly_YYYY-MM.md` (for each processed month)
- `docs/final-btc-strategy.md`
- `backtest/final_strategy_backtest_report.md`
- `traces/run_manifest_*.json`

## Run tests

`pytest` is supported when dependencies are installed.
