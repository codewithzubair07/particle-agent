"""Context indexing module for Particle.

Loads user context files (PDF, TXT, DOCX) from the ``context/`` directory,
chunks and indexes them into a ChromaDB collection, and exposes a similarity
search API so any module can retrieve relevant user context for LLM prompts.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

from modules.config_loader import get_config

logger = logging.getLogger("particle.context_loader")

# ---------------------------------------------------------------------------
# Optional heavy imports — degrade gracefully if packages missing
# ---------------------------------------------------------------------------

try:
    import chromadb  # type: ignore
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction  # type: ignore
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed — context search will be unavailable")

try:
    import pypdf  # type: ignore
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument  # type: ignore
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False

_CHUNK_SIZE = 800       # characters per chunk
_CHUNK_OVERLAP = 100    # overlap between consecutive chunks


def _chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping chunks of ~_CHUNK_SIZE characters."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error("Failed reading %s: %s", path, exc)
        return ""


def _read_pdf(path: Path) -> str:
    if not _PDF_AVAILABLE:
        logger.warning("pypdf not installed; skipping %s", path)
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        logger.error("Failed reading PDF %s: %s", path, exc)
        return ""


def _read_docx(path: Path) -> str:
    if not _DOCX_AVAILABLE:
        logger.warning("python-docx not installed; skipping %s", path)
        return ""
    try:
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        logger.error("Failed reading DOCX %s: %s", path, exc)
        return ""


def _read_file(path: Path) -> str:
    """Read a file's text content regardless of format."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in (".docx", ".doc"):
        return _read_docx(path)
    return _read_txt(path)


class ContextLoader:
    """Manages user context files and a ChromaDB vector index."""

    def __init__(
        self,
        context_dir: str | Path | None = None,
        chroma_dir: str | Path | None = None,
    ) -> None:
        cfg = get_config()
        self._context_dir = Path(context_dir or cfg.paths.context_dir)
        self._chroma_dir = Path(chroma_dir or cfg.paths.chroma_dir)
        self._lock = threading.Lock()
        self._client: Optional[object] = None
        self._collection: Optional[object] = None

        self._context_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        if _CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            logger.warning("ChromaDB unavailable — falling back to plain text search")

    # ------------------------------------------------------------------
    # ChromaDB setup
    # ------------------------------------------------------------------

    def _init_chroma(self) -> None:
        try:
            self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
            self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
                name="particle_context",
                embedding_function=DefaultEmbeddingFunction(),
            )
            logger.info(
                "ChromaDB collection 'particle_context' ready at %s", self._chroma_dir
            )
        except Exception as exc:
            logger.error("ChromaDB init failed: %s", exc)
            self._collection = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_all(self) -> int:
        """Index all supported files in the context directory.

        Returns the total number of chunks indexed.
        """
        extensions = {".txt", ".md", ".pdf", ".docx", ".doc"}
        files = [
            p
            for p in self._context_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]
        total = 0
        for path in files:
            total += self.index_file(path)
        logger.info("Context index_all: %d files, %d chunks", len(files), total)
        return total

    def index_file(self, path: Path) -> int:
        """Index a single file; returns the number of chunks added/updated."""
        text = _read_file(path)
        if not text.strip():
            logger.warning("No text extracted from %s — skipping", path)
            return 0

        chunks = _chunk_text(text)
        if not chunks:
            return 0

        if self._collection is None:
            logger.debug("No ChromaDB collection — file %s not indexed", path)
            return 0

        ids = [f"{path.name}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]

        try:
            with self._lock:
                self._collection.upsert(  # type: ignore[union-attr]
                    ids=ids, documents=chunks, metadatas=metadatas
                )
            logger.info("Indexed %d chunks from %s", len(chunks), path.name)
        except Exception as exc:
            logger.error("Failed to index %s: %s", path, exc)
            return 0

        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Return the top *n_results* context chunks most similar to *query*.

        Each result dict contains keys: ``content``, ``source``, ``chunk``.
        """
        if self._collection is None:
            return self._fallback_search(query, n_results)

        try:
            with self._lock:
                results = self._collection.query(  # type: ignore[union-attr]
                    query_texts=[query], n_results=n_results
                )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            return [
                {"content": doc, "source": meta.get("source", ""), "chunk": meta.get("chunk", 0)}
                for doc, meta in zip(docs, metas)
            ]
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return self._fallback_search(query, n_results)

    def build_context_string(self, query: str, n_results: int = 5) -> str:
        """Return a formatted context block suitable for LLM prompts."""
        hits = self.search(query, n_results)
        if not hits:
            return ""
        parts = ["[Relevant context from user files:]"]
        for i, hit in enumerate(hits, 1):
            src = Path(hit["source"]).name if hit["source"] else "unknown"
            parts.append(f"\n--- Excerpt {i} (from {src}) ---\n{hit['content']}")
        return "\n".join(parts)

    def _fallback_search(self, query: str, n_results: int) -> list[dict]:
        """Plain substring search over all context files (ChromaDB fallback)."""
        query_lower = query.lower()
        results: list[dict] = []
        extensions = {".txt", ".md", ".pdf", ".docx", ".doc"}
        for path in self._context_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            text = _read_file(path)
            for chunk in _chunk_text(text):
                if query_lower in chunk.lower():
                    results.append({"content": chunk, "source": str(path), "chunk": 0})
                    if len(results) >= n_results:
                        return results
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_context_files(self) -> list[str]:
        """Return a list of all context file names."""
        extensions = {".txt", ".md", ".pdf", ".docx", ".doc"}
        return [
            p.name
            for p in sorted(self._context_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in extensions
        ]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: ContextLoader | None = None
_singleton_lock = threading.Lock()


def get_context_loader() -> ContextLoader:
    """Return the module-level :class:`ContextLoader` singleton."""
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = ContextLoader()
    return _instance
