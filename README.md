# RSI Trading Agent (BTC Self-Improving Agent)

This project runs a self-improving BTC strategy simulation loop that combines technical indicators and news sentiment, then reflects on performance to improve across epochs.

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

## Run tests

```bash
pytest
```
