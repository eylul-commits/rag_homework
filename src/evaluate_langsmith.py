"""LangSmith evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langsmith import Client
from langsmith.evaluation import evaluate, run_evaluator
from langsmith.schemas import Example, Run

from .prompts import (
    CRITERION_CONCISENESS,
    CRITERION_CORRECTNESS,
    CRITERION_HALLUCINATION,
    CRITERION_RELEVANCE,
    EVALUATOR_PROMPT,
)
from .rag_pipeline import RAGPipeline
from .utils import get_config, load_eval_dataset, project_root


DEFAULT_DATASET_NAME = "rag-course-eval"


def _format_context_from_outputs(outputs: dict[str, Any]) -> str:
    ctx = outputs.get("retrieved_context")
    if isinstance(ctx, str):
        return ctx
    return ""


def _parse_judge_json(text: str) -> tuple[float, str]:
    text = (text or "").strip()
    if not text:
        return 0.0, "empty judge response"
    # remove the markdown code block (modeller json'ı kod blok ile sarmalıyor genelde)
    if text.startswith("```"):
        m = re.match(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
        return float(data["score"]), str(data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            return float(data["score"]), str(data.get("reasoning", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return 0.0, "failed to parse judge JSON"


def build_judge_llm() -> ChatOllama | ChatGoogleGenerativeAI:
    provider = os.getenv("JUDGE_PROVIDER", "ollama").lower().strip()
    base_url = os.getenv("OLLAMA_BASE_URL")
    if provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("JUDGE_PROVIDER=gemini requires GOOGLE_API_KEY")
        model = os.getenv("JUDGE_GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
    model = os.getenv("JUDGE_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "llama3"))
    return ChatOllama(model=model, base_url=base_url, temperature=0)


def _make_llm_metric_evaluator(
    metric_key: str,
    criterion: str,
    judge_chain: Any,
) -> Any:
    @run_evaluator
    def _eval(run: Run, example: Optional[Example] = None) -> dict[str, Any]:
        outputs = run.outputs or {}
        answer = outputs.get("output", "")
        if not isinstance(answer, str):
            answer = str(answer)
        context = _format_context_from_outputs(outputs)
        question = ""
        ground_truth = ""
        if example:
            question = str((example.inputs or {}).get("question", ""))
            ground_truth = str((example.outputs or {}).get("ground_truth", ""))
        raw = judge_chain.invoke(
            {
                "criterion": criterion,
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "context": context,
            }
        )
        score, reasoning = _parse_judge_json(raw)
        score = max(0.0, min(1.0, score))
        return {"key": metric_key, "score": score, "comment": reasoning}

    _eval.__name__ = f"eval_{metric_key}"
    return _eval


def build_llm_evaluators(judge_llm: ChatOllama | ChatGoogleGenerativeAI) -> list[Any]:
    #fill the prompt → LLM → extract string
    chain = EVALUATOR_PROMPT | judge_llm | StrOutputParser()
    return [
        _make_llm_metric_evaluator("correctness", CRITERION_CORRECTNESS, chain),
        _make_llm_metric_evaluator("relevance", CRITERION_RELEVANCE, chain),
        _make_llm_metric_evaluator("hallucination", CRITERION_HALLUCINATION, chain),
        _make_llm_metric_evaluator("conciseness", CRITERION_CONCISENESS, chain),
    ]


def upload_eval_dataset(
    client: Client,
    dataset_name: str,
    eval_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    rows = load_eval_dataset(eval_path)
    if client.has_dataset(dataset_name=dataset_name):
        if not overwrite:
            print(
                f"Dataset {dataset_name!r} already exists; skip upload. "
                "Use --overwrite-dataset to delete and re-upload.",
                file=sys.stderr,
            )
            return
        client.delete_dataset(dataset_name=dataset_name)
    client.create_dataset(
        dataset_name=dataset_name,
        description="RAG homework: questions and ground-truth answers from course PDFs.",
    )
    examples = [
        {
            "inputs": {"question": r["question"]},
            "outputs": {"ground_truth": r["ground_truth"]},
            "metadata": {
                "source_id": r["id"],
                **({"difficulty": r["difficulty"]} if r.get("difficulty") else {}),
            },
        }
        for r in rows
    ]
    resp = client.create_examples(dataset_name=dataset_name, examples=examples)
    count = resp.get("count", len(examples)) if isinstance(resp, dict) else len(examples)
    print(f"Uploaded {count} examples to LangSmith dataset {dataset_name!r}.")


def _rag_target_fn(pipeline: RAGPipeline, inputs: dict[str, Any]) -> dict[str, Any]:
    question = str(inputs.get("question", ""))
    result = pipeline.query(question)
    return {
        "output": result["answer"],
        "retrieved_context": "\n\n".join(d.page_content for d in result["source_documents"]),
    }


def evaluate_rag_pipeline(
    pipeline: RAGPipeline,
    dataset_name: str,
    *,
    experiment_prefix: str,
    description: str = "RAG evaluation with LLM-as-judge metrics.",
    extra_metadata: dict[str, Any] | None = None,
    max_concurrency: int | None = None,
) -> Any:
    """Run LangSmith ``evaluate`` on a built ``RAGPipeline`` with the standard judge suite."""
    judge_llm = build_judge_llm()
    evaluators = build_llm_evaluators(judge_llm)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return _rag_target_fn(pipeline, inputs)

    mc = max_concurrency
    if mc is None:
        mc = int(os.getenv("RAG_EVAL_MAX_CONCURRENCY", "0"))

    meta: dict[str, str] = {
        "top_k": str(pipeline._top_k),
        "rag_provider": pipeline._provider,
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            meta[str(k)] = str(v)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        description=description,
        metadata=meta,
        max_concurrency=mc,
    )

    print(f"Experiment: {results.experiment_name}")
    if results.url:
        print(f"Compare / results: {results.url}")

    _print_run_aggregates(results)
    return results


def run_baseline_evaluation(
    dataset_name: str,
    *,
    pipeline_overrides: dict[str, Any] | None = None,
    experiment_prefix: str = "rag_baseline",
    max_concurrency: int | None = None,
) -> Any:
    pipeline = RAGPipeline.from_config(**(pipeline_overrides or {}))
    return evaluate_rag_pipeline(
        pipeline,
        dataset_name,
        experiment_prefix=experiment_prefix,
        description="Baseline RAG (Ollama + Chroma) with LLM-as-judge metrics.",
        max_concurrency=max_concurrency,
    )


def collect_eval_score_means(results: Any) -> dict[str, Any]:
    """Aggregate mean judge scores over an ``ExperimentResults`` iterator."""
    from collections import defaultdict

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    sums_by_diff: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts_by_diff: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in results:
        example = row.get("example")
        diff = "unknown"
        if example and example.metadata:
            diff = str(example.metadata.get("difficulty", "unknown"))
        ev = row.get("evaluation_results") or {}
        for res in ev.get("results", []):
            key = getattr(res, "key", None)
            score = getattr(res, "score", None)
            if isinstance(res, dict):
                key = key or res.get("key")
                score = score if score is not None else res.get("score")
            if key is None or score is None:
                continue
            sums[key] += float(score)
            counts[key] += 1
            sums_by_diff[diff][key] += float(score)
            counts_by_diff[diff][key] += 1

    overall = {k: sums[k] / counts[k] for k in counts}
    by_diff: dict[str, dict[str, float]] = {}
    for diff in sums_by_diff:
        by_diff[diff] = {
            k: sums_by_diff[diff][k] / counts_by_diff[diff][k] for k in counts_by_diff[diff]
        }
    return {"overall": overall, "by_difficulty": by_diff, "_counts": dict(counts)}


def _print_run_aggregates(results: Any) -> None:
    agg = collect_eval_score_means(results)
    overall = agg["overall"]
    if not overall:
        print("No evaluation scores collected (check LangSmith UI).")
        return

    print("\nMean scores (all examples):")
    for k in sorted(overall.keys()):
        print(f"  {k}: {overall[k]:.3f}")

    by_diff = agg["by_difficulty"]
    if len(by_diff) > 1 or (len(by_diff) == 1 and "unknown" not in by_diff):
        print("\nMean scores by difficulty (metadata.difficulty):")
        for diff in sorted(by_diff.keys()):
            print(f"  [{diff}]")
            for k in sorted(by_diff[diff].keys()):
                print(f"    {k}: {by_diff[diff][k]:.3f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload eval dataset and/or run LangSmith baseline evaluation.")
    p.add_argument("--upload-dataset", action="store_true", help="Upload local eval JSON to LangSmith")
    p.add_argument(
        "--overwrite-dataset",
        action="store_true",
        help="If dataset exists, delete it before upload",
    )
    p.add_argument("--run-baseline", action="store_true", help="Run RAG + evaluators on the LangSmith dataset")
    p.add_argument(
        "--dataset-name",
        default=os.getenv("LANGSMITH_DATASET", DEFAULT_DATASET_NAME),
        help="LangSmith dataset name",
    )
    p.add_argument(
        "--eval-path",
        type=Path,
        default=None,
        help="Path to eval_dataset.json (default: from get_config)",
    )
    p.add_argument("--experiment-prefix", default="rag_baseline", help="Prefix for LangSmith experiment name")
    p.add_argument("--provider", choices=["ollama", "gemini"], default=None, help="RAG LLM provider")
    p.add_argument("--top-k", type=int, default=None, help="RAG retrieval k")
    p.add_argument("--collection", default=None, help="Chroma collection name")
    p.add_argument("--persist-dir", type=Path, default=None, help="Chroma persist directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(project_root() / ".env")
    args = parse_args(argv)

    if args.upload_dataset or args.run_baseline:
        if not os.getenv("LANGCHAIN_API_KEY"):
            print("Error: LANGCHAIN_API_KEY is not set (needed for LangSmith).", file=sys.stderr)
            return 1
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    cfg = get_config()
    eval_path = args.eval_path or cfg["eval_dataset_path"]

    client = Client()

    if args.upload_dataset:
        try:
            upload_eval_dataset(
                client,
                args.dataset_name,
                Path(eval_path),
                overwrite=args.overwrite_dataset,
            )
        except Exception as e:
            print(f"Upload failed: {e}", file=sys.stderr)
            return 1

    if args.run_baseline:
        overrides: dict[str, Any] = {}
        if args.provider is not None:
            overrides["provider"] = args.provider
        if args.top_k is not None:
            overrides["top_k"] = args.top_k
        if args.collection is not None:
            overrides["collection_name"] = args.collection
        if args.persist_dir is not None:
            overrides["persist_directory"] = args.persist_dir.resolve()
        try:
            run_baseline_evaluation(
                args.dataset_name,
                pipeline_overrides=overrides or None,
                experiment_prefix=args.experiment_prefix,
            )
        except Exception as e:
            print(f"Baseline evaluation failed: {e}", file=sys.stderr)
            return 1

    if not args.upload_dataset and not args.run_baseline:
        print("Nothing to do. Use --upload-dataset and/or --run-baseline.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
