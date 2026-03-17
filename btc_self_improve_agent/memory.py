from __future__ import annotations

import json
import sqlite3
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

    def close(self) -> None:
        self.conn.close()
