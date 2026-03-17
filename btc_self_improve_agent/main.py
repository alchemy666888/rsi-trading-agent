from __future__ import annotations

from btc_self_improve_agent.agent import BTCSelfImprovingAgent


def main() -> None:
    agent = BTCSelfImprovingAgent()
    result = agent.run(
        "Design a BTC daily strategy combining RSI, MACD, and news sentiment with annual return > 50% and MaxDD < 25%.",
        epochs=5,
    )
    print("Best result:", result)


if __name__ == "__main__":
    main()
