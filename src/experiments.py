"""Experiment runners and parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .evaluate_langsmith import collect_eval_score_means, evaluate_rag_pipeline
from .ingest import ingest
from .rag_pipeline import RAGPipeline
from .utils import get_config, project_root

# Chunk size, overlap, and retrieval depth — 11 configurations
EXPERIMENT_GRID: list[dict[str, int]] = [
    {"chunk_size": 500, "chunk_overlap": 50, "top_k": 3},
    {"chunk_size": 500, "chunk_overlap": 100, "top_k": 5},
    {"chunk_size": 800, "chunk_overlap": 100, "top_k": 4},
    {"chunk_size": 1000, "chunk_overlap": 100, "top_k": 3},
    {"chunk_size": 1000, "chunk_overlap": 200, "top_k": 5},
    {"chunk_size": 1000, "chunk_overlap": 200, "top_k": 7},
    {"chunk_size": 1500, "chunk_overlap": 200, "top_k": 3},
    {"chunk_size": 1500, "chunk_overlap": 300, "top_k": 5},
    {"chunk_size": 2000, "chunk_overlap": 200, "top_k": 3},
    {"chunk_size": 2000, "chunk_overlap": 400, "top_k": 5},
    {"chunk_size": 2000, "chunk_overlap": 400, "top_k": 7},
]


def experiment_slug(exp: dict[str, int]) -> str:
    return f"cs{exp['chunk_size']}_co{exp['chunk_overlap']}_k{exp['top_k']}"


def append_experiment_csv(
    csv_path: Path,
    batch_id: str,
    exp_slug: str,
    agg: dict[str, Any],
    *,
    langsmith_url: str | None,
    extra_notes: dict[str, Any],
) -> None:
    """Append one row per overall metric; notes hold config + URL + by-difficulty JSON."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file() or csv_path.stat().st_size == 0
    overall = agg.get("overall") or {}
    by_diff = agg.get("by_difficulty") or {}
    base_notes = {
        **extra_notes,
        "langsmith_url": langsmith_url,
        "mean_by_difficulty": by_diff,
    }
    notes_json = json.dumps(base_notes, ensure_ascii=False)

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["run_id", "experiment", "metric", "value", "notes"])
        for metric in sorted(overall.keys()):
            w.writerow([batch_id, exp_slug, f"mean_{metric}", f"{overall[metric]:.6f}", notes_json])


def run_single_experiment(
    cfg: dict[str, Any],
    exp: dict[str, int],
    dataset_name: str,
    *,
    skip_ingest: bool = False,
    provider: str = "ollama",
    max_concurrency: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    pdf_dir = cfg["pdf_directory"]
    persist = cfg["persist_directory"]
    collection = cfg["collection_name"]
    embed = cfg["embedding_model"]
    base_url = cfg["base_url"]

    if not skip_ingest:
        n_chunks = ingest(
            pdf_dir=pdf_dir,
            persist_directory=persist,
            collection_name=collection,
            chunk_size=exp["chunk_size"],
            chunk_overlap=exp["chunk_overlap"],
            embedding_model=embed,
            base_url=base_url,
            force=True,
        )
        print(f"Ingested {n_chunks} chunks (cs={exp['chunk_size']}, co={exp['chunk_overlap']}).")
    else:
        print("Skipping ingest (--skip-ingest); using existing Chroma (top_k still applies).")

    pipeline = RAGPipeline(
        persist_directory=persist,
        collection_name=collection,
        embedding_model=embed,
        base_url=base_url,
        top_k=exp["top_k"],
        provider=provider,
        ollama_model=cfg["ollama_model"],
        gemini_model=cfg["gemini_model"],
        google_api_key=cfg.get("google_api_key"),
    )

    slug = experiment_slug(exp)
    prefix = f"rag_exp_{slug}"
    if provider != "ollama":
        prefix = f"{prefix}_{provider}"

    extra_meta = {
        "chunk_size": exp["chunk_size"],
        "chunk_overlap": exp["chunk_overlap"],
        "experiment_slug": slug,
    }

    results = evaluate_rag_pipeline(
        pipeline,
        dataset_name,
        experiment_prefix=prefix,
        description=f"RAG sweep: chunk_size={exp['chunk_size']}, overlap={exp['chunk_overlap']}, top_k={exp['top_k']}, provider={provider}.",
        extra_metadata=extra_meta,
        max_concurrency=max_concurrency,
    )

    agg = collect_eval_score_means(results)
    return results, agg


def run_experiment_batch(
    dataset_name: str,
    *,
    csv_path: Path,
    indices: list[int] | None = None,
    skip_ingest: bool = False,
    provider: str = "ollama",
    with_gemini: bool = False,
    gemini_grid_index: int = 4,
    max_concurrency: int | None = None,
) -> None:
    cfg = get_config()
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    to_run = list(range(len(EXPERIMENT_GRID))) if indices is None else indices
    if skip_ingest and len(to_run) > 1:
        print(
            "Warning: --skip-ingest with multiple experiments uses one Chroma index for all runs; "
            "only top_k changes. Re-ingest per config unless you are debugging.",
            file=sys.stderr,
        )
    for i in to_run:
        if i < 0 or i >= len(EXPERIMENT_GRID):
            raise IndexError(f"Experiment index {i} out of range (0..{len(EXPERIMENT_GRID) - 1})")
        exp = EXPERIMENT_GRID[i]
        print(f"\n=== Experiment {i}: {experiment_slug(exp)} (provider={provider}) ===")
        results, agg = run_single_experiment(
            cfg,
            exp,
            dataset_name,
            skip_ingest=skip_ingest,
            provider=provider,
            max_concurrency=max_concurrency,
        )
        url = getattr(results, "url", None)
        append_experiment_csv(
            csv_path,
            batch_id,
            experiment_slug(exp),
            agg,
            langsmith_url=url,
            extra_notes={
                "grid_index": i,
                "chunk_size": exp["chunk_size"],
                "chunk_overlap": exp["chunk_overlap"],
                "top_k": exp["top_k"],
                "provider": provider,
            },
        )

    if with_gemini:
        gi = gemini_grid_index
        if gi < 0 or gi >= len(EXPERIMENT_GRID):
            raise IndexError(f"--gemini-config-index {gi} out of range")
        exp = EXPERIMENT_GRID[gi]
        print(f"\n=== Bonus: Gemini with grid config {gi}: {experiment_slug(exp)} ===")
        if not cfg.get("google_api_key"):
            print("Warning: GOOGLE_API_KEY not set; skipping Gemini run.", file=sys.stderr)
            return
        results, agg = run_single_experiment(
            cfg,
            exp,
            dataset_name,
            skip_ingest=True,
            provider="gemini",
            max_concurrency=max_concurrency,
        )
        url = getattr(results, "url", None)
        append_experiment_csv(
            csv_path,
            batch_id,
            f"{experiment_slug(exp)}_gemini",
            agg,
            langsmith_url=url,
            extra_notes={
                "grid_index": gi,
                "chunk_size": exp["chunk_size"],
                "chunk_overlap": exp["chunk_overlap"],
                "top_k": exp["top_k"],
                "provider": "gemini",
            },
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run chunk/overlap/top_k sweeps; each trial re-ingests, evaluates on LangSmith, logs CSV.",
    )
    p.add_argument(
        "--dataset-name",
        default=os.getenv("LANGSMITH_DATASET", "rag-course-eval"),
        help="LangSmith dataset name",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=project_root() / "results" / "experiment_results.csv",
        help="Append results here",
    )
    p.add_argument(
        "--experiment-index",
        type=int,
        default=None,
        help="Run only this 0-based grid index (default: run full grid)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit")
    p.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Do not re-ingest (use current Chroma; chunk params only affect naming/metadata)",
    )
    p.add_argument(
        "--provider",
        choices=["ollama", "gemini"],
        default="ollama",
        help="LLM for generation during the sweep (default: ollama)",
    )
    p.add_argument(
        "--with-gemini",
        action="store_true",
        help="After the sweep, run one Gemini eval using the grid row from --gemini-config-index",
    )
    p.add_argument(
        "--gemini-config-index",
        type=int,
        default=4,
        help="EXPERIMENT_GRID index for the optional Gemini run (default: 4)",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Override RAG_EVAL_MAX_CONCURRENCY for LangSmith evaluate()",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(project_root() / ".env")
    args = parse_args(argv)

    if args.dry_run:
        print(f"Grid ({len(EXPERIMENT_GRID)} configs):")
        for i, exp in enumerate(EXPERIMENT_GRID):
            print(f"  [{i}] {experiment_slug(exp)}")
        return 0

    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Error: LANGCHAIN_API_KEY is required for LangSmith evaluation.", file=sys.stderr)
        return 1
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    indices = None if args.experiment_index is None else [args.experiment_index]
    try:
        run_experiment_batch(
            args.dataset_name,
            csv_path=args.csv.resolve(),
            indices=indices,
            skip_ingest=args.skip_ingest,
            provider=args.provider,
            with_gemini=args.with_gemini,
            gemini_grid_index=args.gemini_config_index,
            max_concurrency=args.max_concurrency,
        )
    except Exception as e:
        print(f"Experiment batch failed: {e}", file=sys.stderr)
        return 1

    print(f"\nAppended metrics to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
