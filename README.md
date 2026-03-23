# RSI Trading Agent (BTC Self-Improving Agent)

This project runs a self-improving BTC strategy simulation loop that combines technical indicators and news sentiment, then reflects on performance to improve across epochs.

It now also includes:

- A market data fetch + analysis agent that saves BTC/USDT historical data and LLM analysis into `data/`
- A news fetch + analysis agent that saves BTC-related news archives and LLM analysis into `news/`

Both research agents cover `2023-01-01` through `2025-12-31` by default.

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

The script will execute multiple strategy-improvement epochs and print the best result at the end.

Before the strategy loop starts, it will also create:

- `data/btc_usdt_1d_2023-01-01_2025-12-31.csv`
- `data/btc_usdt_1d_2023-01-01_2025-12-31_analysis.json`
- `data/btc_usdt_1d_2023-01-01_2025-12-31_analysis.md`
- `news/btc_news_2023-01-01_2025-12-31.json`
- `news/btc_news_2023-01-01_2025-12-31_analysis.json`
- `news/btc_news_2023-01-01_2025-12-31_analysis.md`

## Run tests

```bash
pytest
```
