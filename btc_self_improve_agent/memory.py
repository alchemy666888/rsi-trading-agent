from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MemoryManager:
    def __init__(self, session_id: str, db_path: str = "strategies.db"):
        self.session_id = session_id
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                strategy TEXT,
                metrics TEXT,
                score INT,
                lesson TEXT
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS monthly_strategies (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                month_id TEXT NOT NULL,
                strategy_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                score REAL NOT NULL,
                lesson TEXT NOT NULL,
                change_log TEXT,
                validation_flags TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_monthly_session_month
            ON monthly_strategies (session_id, month_id)
            """
        )
        self.conn.commit()

    def store_strategy(self, strategy: dict[str, Any], metrics: dict[str, Any], score: int, lesson: str) -> None:
        self.cursor.execute(
            "INSERT INTO strategies (session_id, strategy, metrics, score, lesson) VALUES (?, ?, ?, ?, ?)",
            (self.session_id, json.dumps(strategy), json.dumps(metrics), score, lesson),
        )
        self.conn.commit()

    def get_relevant_lessons(self, min_score: int = 70, limit: int = 3) -> str:
        self.cursor.execute(
            "SELECT lesson FROM strategies WHERE session_id = ? AND score >= ? ORDER BY score DESC LIMIT ?",
            (self.session_id, min_score, limit),
        )
        lessons = [row[0] for row in self.cursor.fetchall()]
        return "\n".join(lessons) if lessons else "No past lessons yet."

    def store_monthly_strategy(
        self,
        *,
        month_id: str,
        strategy: dict[str, Any],
        metrics: dict[str, Any],
        score: float,
        lesson: str,
        change_log: dict[str, Any] | None = None,
        validation_flags: dict[str, Any] | None = None,
    ) -> None:
        self.cursor.execute(
            """
            INSERT INTO monthly_strategies (
                session_id, month_id, strategy_json, metrics_json, score, lesson, change_log, validation_flags, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id,
                month_id,
                json.dumps(strategy),
                json.dumps(metrics),
                float(score),
                lesson,
                json.dumps(change_log or {}),
                json.dumps(validation_flags or {}),
                datetime.now(UTC).isoformat(),
            ),
        )
        self.conn.commit()

    def get_recent_monthly_lessons(self, limit: int = 8) -> list[dict[str, Any]]:
        self.cursor.execute(
            """
            SELECT month_id, lesson, score, strategy_json, metrics_json, change_log, validation_flags
            FROM monthly_strategies
            WHERE session_id = ?
            ORDER BY month_id DESC
            LIMIT ?
            """,
            (self.session_id, limit),
        )
        rows = self.cursor.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "month_id": row[0],
                    "lesson": row[1],
                    "score": float(row[2]),
                    "strategy": json.loads(row[3]),
                    "metrics": json.loads(row[4]),
                    "change_log": json.loads(row[5] or "{}"),
                    "validation_flags": json.loads(row[6] or "{}"),
                }
            )
        return result

    def get_monthly_context(self, top_n: int = 5) -> str:
        lessons = self.get_recent_monthly_lessons(limit=max(1, top_n))
        if not lessons:
            return "No monthly lessons yet."
        lines: list[str] = []
        for item in reversed(lessons):
            lines.append(
                f"{item['month_id']} score={item['score']:.2f}: {item['lesson']}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        self.conn.close()
