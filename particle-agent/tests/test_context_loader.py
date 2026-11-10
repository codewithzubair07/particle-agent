"""Regression tests for context chunk ids.

Two context files that share a filename but live in different subdirectories
must not overwrite each other in the vector index.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.context_loader import ContextLoader


class FakeCollection:
    """Minimal stand-in for a ChromaDB collection keyed by chunk id."""

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}

    def upsert(self, ids, documents, metadatas) -> None:  # noqa: ANN001
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            self.store[chunk_id] = {"document": document, "metadata": metadata}

    @property
    def ids(self) -> list[str]:
        return list(self.store)


def _loader(tmp_path: Path) -> ContextLoader:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    return ContextLoader(context_dir=context_dir, chroma_dir=tmp_path / "chroma")


def test_duplicate_filenames_in_separate_directories_get_distinct_ids(tmp_path):
    loader = _loader(tmp_path)
    work = loader._context_dir / "work" / "notes.md"
    personal = loader._context_dir / "personal" / "notes.md"
    work.parent.mkdir()
    personal.parent.mkdir()
    work.write_text("work note", encoding="utf-8")
    personal.write_text("personal note", encoding="utf-8")

    collection = FakeCollection()
    loader._collection = collection

    loader.index_file(work)
    loader.index_file(personal)

    assert collection.ids == ["work/notes.md::0", "personal/notes.md::0"]
    assert collection.store["work/notes.md::0"]["metadata"]["source"] == str(work)
    assert collection.store["personal/notes.md::0"]["metadata"]["source"] == str(
        personal
    )


def test_reindexing_the_same_file_is_idempotent(tmp_path):
    loader = _loader(tmp_path)
    notes = loader._context_dir / "notes.md"
    notes.write_text("same file, indexed twice", encoding="utf-8")

    collection = FakeCollection()
    loader._collection = collection

    loader.index_file(notes)
    loader.index_file(notes)

    assert collection.ids == ["notes.md::0"]
    assert collection.store["notes.md::0"]["metadata"]["source"] == str(notes)
