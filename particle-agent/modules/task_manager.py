"""Task management module for Particle.

Provides full CRUD for user tasks backed by SQLite.  Tasks are surfaced in the
daily briefing and users can manage them via Telegram commands.

Schema
------
``tasks``
    id          INTEGER  PRIMARY KEY
    title       TEXT     NOT NULL
    description TEXT     DEFAULT ''
    priority    TEXT     NOT NULL  -- 'high' | 'medium' | 'low'
    due_date    TEXT              -- ISO-8601 date (YYYY-MM-DD), nullable
    status      TEXT     NOT NULL  -- 'pending' | 'in_progress' | 'completed' | 'deleted'
    created_at  TEXT     NOT NULL
    updated_at  TEXT     NOT NULL
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.task_manager")

VALID_PRIORITIES = ("high", "medium", "low")
VALID_STATUSES = ("pending", "in_progress", "completed", "deleted")

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    priority    TEXT    NOT NULL DEFAULT 'medium',
    due_date    TEXT,
    status      TEXT    NOT NULL DEFAULT 'pending',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status   ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskManager:
    """Thread-safe SQLite-backed task manager."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            db_path = cfg.paths.task_db
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logger.info("TaskManager initialised at %s", self._path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_DB_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        due_date: Optional[str] = None,
    ) -> int:
        """Create a new task and return its ID."""
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'. Must be one of {VALID_PRIORITIES}")

        now = _now_iso()
        sql = (
            "INSERT INTO tasks (title, description, priority, due_date, status, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, 'pending', ?, ?)"
        )
        with self._lock:
            cur = self._conn.execute(
                sql, (title, description, priority, due_date, now, now)
            )
            self._conn.commit()
            task_id: int = cur.lastrowid  # type: ignore[assignment]
        logger.info("Task created id=%d title=%r priority=%s due=%s", task_id, title, priority, due_date)
        return task_id

    def get(self, task_id: int) -> Optional[dict]:
        """Return a single task by ID, or ``None`` if not found."""
        sql = "SELECT * FROM tasks WHERE id = ?"
        with self._lock:
            row = self._conn.execute(sql, (task_id,)).fetchone()
        return dict(row) if row else None

    def update(
        self,
        task_id: int,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        due_date: Optional[str] = None,
        status: Optional[str] = None,
    ) -> bool:
        """Partially update a task; returns ``True`` on success."""
        if priority is not None and priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'")
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'")

        fields: list[str] = []
        values: list = []
        for col, val in [
            ("title", title),
            ("description", description),
            ("priority", priority),
            ("due_date", due_date),
            ("status", status),
        ]:
            if val is not None:
                fields.append(f"{col} = ?")
                values.append(val)

        if not fields:
            return False

        fields.append("updated_at = ?")
        values.append(_now_iso())
        values.append(task_id)

        sql = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"
        with self._lock:
            cur = self._conn.execute(sql, values)
            self._conn.commit()
        logger.info("Task updated id=%d fields=%s", task_id, fields)
        return cur.rowcount > 0

    def complete(self, task_id: int) -> bool:
        """Mark a task as completed."""
        return self.update(task_id, status="completed")

    def delete(self, task_id: int) -> bool:
        """Soft-delete a task by setting status to 'deleted'."""
        return self.update(task_id, status="deleted")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        include_deleted: bool = False,
    ) -> list[dict]:
        """Return tasks, optionally filtered by status/priority."""
        conditions: list[str] = []
        params: list = []

        if not include_deleted:
            conditions.append("status != 'deleted'")

        if status:
            conditions.append("status = ?")
            params.append(status)
        if priority:
            conditions.append("priority = ?")
            params.append(priority)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM tasks {where} ORDER BY due_date ASC, priority ASC, id ASC"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def pending(self) -> list[dict]:
        """Return all pending tasks."""
        return self.list_tasks(status="pending")

    def due_soon(self, days_ahead: int = 1) -> list[dict]:
        """Return pending tasks due within *days_ahead* days from today."""
        from datetime import date, timedelta

        cutoff = (date.today() + timedelta(days=days_ahead)).isoformat()
        today = date.today().isoformat()
        sql = (
            "SELECT * FROM tasks WHERE status = 'pending' "
            "AND due_date IS NOT NULL AND due_date >= ? AND due_date <= ? "
            "ORDER BY due_date ASC"
        )
        with self._lock:
            rows = self._conn.execute(sql, (today, cutoff)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()
        logger.info("TaskManager closed")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: TaskManager | None = None
_singleton_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    """Return the module-level :class:`TaskManager` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = TaskManager()
    return _instance
