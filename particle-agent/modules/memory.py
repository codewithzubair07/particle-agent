"""Conversation memory module for Particle.

Persists conversation turns and arbitrary key-value facts to SQLite so the
assistant can recall past interactions across restarts.

Schema
------
``conversations`` — one row per message turn
    id         INTEGER  PRIMARY KEY
    session_id TEXT     — groups turns into logical conversations
    role       TEXT     — 'user' | 'assistant' | 'system'
    content    TEXT     — message body
    created_at TEXT     — ISO-8601 UTC timestamp

``facts`` — long-term user facts / preferences
    id         INTEGER  PRIMARY KEY
    key        TEXT     UNIQUE
    value      TEXT
    updated_at TEXT     — ISO-8601 UTC timestamp
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.memory")

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL DEFAULT 'default',
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    key        TEXT    NOT NULL UNIQUE,
    value      TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Memory:
    """Thread-safe SQLite-backed conversation memory store."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            db_path = cfg.paths.memory_db
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        logger.info("Memory store initialised at %s", self._path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_DB_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Conversation API
    # ------------------------------------------------------------------

    def add_turn(
        self, role: str, content: str, session_id: str = "default"
    ) -> int:
        """Insert a conversation turn and return its row ID."""
        sql = (
            "INSERT INTO conversations (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)"
        )
        with self._lock:
            cur = self._conn.execute(sql, (session_id, role, content, _now_iso()))
            self._conn.commit()
            row_id: int = cur.lastrowid  # type: ignore[assignment]
        logger.debug("Memory.add_turn session=%s role=%s id=%d", session_id, role, row_id)
        return row_id

    def get_recent(
        self, session_id: str = "default", limit: int = 20
    ) -> list[dict]:
        """Return the *limit* most-recent turns for a session, oldest first."""
        sql = (
            "SELECT role, content, created_at FROM conversations "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (session_id, limit)).fetchall()
        result = [dict(r) for r in reversed(rows)]
        logger.debug(
            "Memory.get_recent session=%s limit=%d returned=%d", session_id, limit, len(result)
        )
        return result

    def get_all_sessions(self) -> list[str]:
        """Return a list of all distinct session IDs."""
        sql = "SELECT DISTINCT session_id FROM conversations ORDER BY session_id"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [r["session_id"] for r in rows]

    def clear_session(self, session_id: str = "default") -> int:
        """Delete all turns in a session; returns number of deleted rows."""
        sql = "DELETE FROM conversations WHERE session_id = ?"
        with self._lock:
            cur = self._conn.execute(sql, (session_id,))
            self._conn.commit()
        logger.info("Memory.clear_session session=%s deleted=%d", session_id, cur.rowcount)
        return cur.rowcount

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Full-text search across all conversation content (LIKE-based)."""
        sql = (
            "SELECT session_id, role, content, created_at FROM conversations "
            "WHERE content LIKE ? ORDER BY id DESC LIMIT ?"
        )
        pattern = f"%{query}%"
        with self._lock:
            rows = self._conn.execute(sql, (pattern, limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Facts API
    # ------------------------------------------------------------------

    def set_fact(self, key: str, value: str) -> None:
        """Upsert a long-term fact."""
        sql = (
            "INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at"
        )
        with self._lock:
            self._conn.execute(sql, (key, value, _now_iso()))
            self._conn.commit()
        logger.debug("Memory.set_fact key=%s", key)

    def get_fact(self, key: str) -> Optional[str]:
        """Return a stored fact value, or ``None`` if absent."""
        sql = "SELECT value FROM facts WHERE key = ?"
        with self._lock:
            row = self._conn.execute(sql, (key,)).fetchone()
        return row["value"] if row else None

    def list_facts(self) -> dict[str, str]:
        """Return all stored facts as a plain dict."""
        sql = "SELECT key, value FROM facts ORDER BY key"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_fact(self, key: str) -> bool:
        """Remove a fact; returns ``True`` if a row was deleted."""
        sql = "DELETE FROM facts WHERE key = ?"
        with self._lock:
            cur = self._conn.execute(sql, (key,))
            self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()
        logger.info("Memory store closed")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Memory | None = None
_singleton_lock = threading.Lock()


def get_memory() -> Memory:
    """Return the module-level :class:`Memory` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = Memory()
    return _instance
