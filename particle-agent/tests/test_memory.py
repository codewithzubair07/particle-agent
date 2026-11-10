"""Regression tests for Memory query limit validation (issue #9).

``Memory.get_recent()`` and ``Memory.search()`` used to pass caller-provided
limits straight into SQLite.  SQLite treats ``LIMIT -1`` as "no limit" and
silently accepts zero, so an invalid limit returned the entire table instead
of failing fast.  These tests pin down the new contract: both methods require
a strictly positive integer and raise ``ValueError`` otherwise, while valid
limits behave exactly as before.

Run from the ``particle-agent/`` project root:

    python -m pytest tests/test_memory.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project package importable when pytest runs from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.memory import Memory  # noqa: E402


@pytest.fixture()
def memory(tmp_path: Path) -> Memory:
    """Return a Memory instance backed by a temporary SQLite database."""
    store = Memory(db_path=tmp_path / "memory_test.db")
    yield store
    store.close()


def _seed(store: Memory) -> None:
    store.add_turn("user", "first message", session_id="s1")
    store.add_turn("assistant", "second message", session_id="s1")
    store.add_turn("user", "third message", session_id="s1")


# ---------------------------------------------------------------------------
# Positive-limit behaviour is unchanged
# ---------------------------------------------------------------------------


def test_get_recent_positive_limit_unchanged(memory: Memory) -> None:
    _seed(memory)
    turns = memory.get_recent(session_id="s1", limit=2)
    assert [t["content"] for t in turns] == ["second message", "third message"]


def test_search_positive_limit_unchanged(memory: Memory) -> None:
    _seed(memory)
    hits = memory.search("message", limit=2)
    assert len(hits) == 2


def test_get_recent_limit_larger_than_row_count(memory: Memory) -> None:
    _seed(memory)
    assert len(memory.get_recent(session_id="s1", limit=100)) == 3


# ---------------------------------------------------------------------------
# Invalid limits raise ValueError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_limit", [0, -1, -100])
def test_get_recent_rejects_non_positive_limits(
    memory: Memory, bad_limit: int
) -> None:
    _seed(memory)
    with pytest.raises(ValueError, match="limit"):
        memory.get_recent(session_id="s1", limit=bad_limit)


@pytest.mark.parametrize("bad_limit", [0, -1, -100])
def test_search_rejects_non_positive_limits(memory: Memory, bad_limit: int) -> None:
    _seed(memory)
    with pytest.raises(ValueError, match="limit"):
        memory.search("message", limit=bad_limit)


@pytest.mark.parametrize("bad_limit", ["5", 2.5, None, [5]])
def test_get_recent_rejects_non_integer_limits(
    memory: Memory, bad_limit: object
) -> None:
    _seed(memory)
    with pytest.raises(ValueError, match="limit"):
        memory.get_recent(session_id="s1", limit=bad_limit)  # type: ignore[arg-type]


def test_limit_must_not_be_a_bool(memory: Memory) -> None:
    _seed(memory)
    # bool is a subclass of int in Python; True would silently act as limit=1.
    with pytest.raises(ValueError, match="limit"):
        memory.get_recent(session_id="s1", limit=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Error messages identify the invalid argument
# ---------------------------------------------------------------------------


def test_error_message_names_method_and_value(memory: Memory) -> None:
    _seed(memory)
    with pytest.raises(ValueError) as excinfo:
        memory.get_recent(session_id="s1", limit=-3)
    assert "get_recent" in str(excinfo.value)
    assert "-3" in str(excinfo.value)


# ---------------------------------------------------------------------------
# No partial work on rejection
# ---------------------------------------------------------------------------


def test_invalid_limit_returns_no_rows(memory: Memory) -> None:
    _seed(memory)
    # The rejection happens before the query runs; afterwards the store
    # is still intact and behaves normally.
    with pytest.raises(ValueError):
        memory.get_recent(session_id="s1", limit=0)
    assert len(memory.get_recent(session_id="s1", limit=20)) == 3
