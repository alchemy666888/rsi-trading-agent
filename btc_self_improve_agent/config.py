from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class TradingConfig:
    symbol_exchange: str = "BTC/USDT"
    symbol_compact: str = "BTCUSDT"
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    timeframes: tuple[str, str, str, str] = ("15m", "1h", "4h", "1d")
    trend_timeframes: tuple[str, str] = ("4h", "1d")
    entry_timeframes: tuple[str, str] = ("15m", "1h")
    timezone: str = "UTC"
    monthly_score_threshold: float = 60.0

    @property
    def start_ts(self) -> str:
        return f"{self.start_date}T00:00:00Z"

    @property
    def end_ts(self) -> str:
        return f"{self.end_date}T23:59:59Z"

    @property
    def month_ids(self) -> list[str]:
        start = datetime.fromisoformat(self.start_date).replace(tzinfo=UTC)
        end = datetime.fromisoformat(self.end_date).replace(tzinfo=UTC)
        months: list[str] = []
        cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
        while cursor <= end:
            months.append(cursor.strftime("%Y-%m"))
            if cursor.month == 12:
                cursor = datetime(cursor.year + 1, 1, 1, tzinfo=UTC)
            else:
                cursor = datetime(cursor.year, cursor.month + 1, 1, tzinfo=UTC)
        return months


DEFAULT_CONFIG = TradingConfig()
