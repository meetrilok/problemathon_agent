import json
import sqlite3
from typing import Iterable

from idea_bot.config import AppConfig
from idea_bot.domain.models import IdeaRecord


class IdeaRepository:
    """Repository pattern for idea persistence (SQLite)."""

    def __init__(self, config: AppConfig) -> None:
        self._db_path = config.db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        conn = self._connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ideas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def list_ideas(self) -> list[IdeaRecord]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM ideas ORDER BY created_at DESC").fetchall()
        conn.close()
        return [
            IdeaRecord(
                id=int(r["id"]),
                name=r["name"],
                email=r["email"],
                title=r["title"],
                description=r["description"],
                category=r["category"],
                created_at=r["created_at"],
                embedding=json.loads(r["embedding"]),
            )
            for r in rows
        ]

    def add_idea(
        self,
        *,
        name: str,
        email: str,
        title: str,
        description: str,
        category: str,
        created_at: str,
        embedding: list[float],
    ) -> int:
        conn = self._connect()
        cursor = conn.execute(
            """
            INSERT INTO ideas (name, email, title, description, category, created_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, email, title, description, category, created_at, json.dumps(embedding)),
        )
        conn.commit()
        new_id = int(cursor.lastrowid)
        conn.close()
        return new_id

    def iter_embeddings(self) -> Iterable[IdeaRecord]:
        return self.list_ideas()
