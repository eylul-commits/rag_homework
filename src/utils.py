from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

_PATH_KEYS = frozenset({"persist_directory", "pdf_directory", "eval_dataset_path"})


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_eval_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation examples from JSON. Each item must include id, question, and ground_truth.

    Optional field: difficulty (e.g. easy | medium | hard).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Eval dataset not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Eval dataset must be a JSON array of objects")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Item {i} must be an object")
        missing = {"id", "question", "ground_truth"} - row.keys()
        if missing:
            raise ValueError(f"Item {i} missing keys: {sorted(missing)}")
        out.append(dict(row))
    return out


@contextmanager
def timer() -> Generator[dict[str, float | None], None, None]:
    """Measure wall time for a block. After the block, ``stats["elapsed_ms"]`` is set."""
    stats: dict[str, float | None] = {"elapsed_ms": None}
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


def get_config(**overrides: Any) -> dict[str, Any]:
    """Default RAG / ingest settings merged with ``overrides``. Path values are ``Path`` objects.

    Environment (if set): ``OLLAMA_MODEL``, ``OLLAMA_EMBED_MODEL``, ``OLLAMA_BASE_URL``,
    ``GEMINI_MODEL``, ``GOOGLE_API_KEY`` (Gemini when ``provider`` is not Ollama).
    """
    root = project_root()
    defaults: dict[str, Any] = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
        "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "base_url": os.getenv("OLLAMA_BASE_URL"),
        "collection_name": "rag_docs",
        "persist_directory": root / "chroma_db",
        "pdf_directory": root / "data" / "pdfs",
        "eval_dataset_path": root / "data" / "eval_dataset.json",
        "provider": "ollama",
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
    }
    merged = {**defaults, **overrides}
    for key in _PATH_KEYS:
        if key in merged and merged[key] is not None:
            merged[key] = Path(merged[key])
    return merged
