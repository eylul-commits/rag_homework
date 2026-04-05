# RAG homework

Retrieval-augmented QA over course PDFs using **LangChain**, **Ollama** (embeddings + local LLM), **Chroma**, and optional **LangSmith** evaluation. Optional **Google Gemini** can replace Ollama for generation.

## Prerequisites

- **Python 3.11+** recommended.
- **Ollama** running locally, with:
  - A chat model (e.g. `llama3`, `mistral`) — set via `OLLAMA_MODEL`.
  - An embedding model (e.g. `nomic-embed-text`) — set via `OLLAMA_EMBED_MODEL`; pull it with `ollama pull nomic-embed-text`.
- For LangSmith: an account and **API key** at [https://smith.langchain.com](https://smith.langchain.com).
- For Gemini (bonus / `--provider gemini`): **Google API key** with Generative Language API access.

## Repository layout

| Path | Purpose |
|------|---------|
| `data/pdfs/` | Source PDFs (place all course documents here). |
| `data/eval_dataset.json` | Eval questions, `ground_truth`, optional `difficulty`. |
| `src/ingest.py` | Chunk PDFs, embed with Ollama, persist to Chroma. |
| `src/rag_pipeline.py` | Load Chroma, retrieve, answer with Ollama or Gemini. |
| `src/evaluate_langsmith.py` | Upload dataset to LangSmith, baseline evaluation, LLM judges. |
| `src/experiments.py` | Parameter sweep (chunk / overlap / `top_k`) with LangSmith + CSV log. |
| `chroma_db/` | Local Chroma persistence (created by ingest; gitignored). |
| `results/` | `experiment_results.csv`, `manual_test_results.md`, `final_report.md`. |

## Setup

1. Create a virtual environment (recommended), then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy environment template and fill in secrets (create `.env` in the project root):

   ```bash
   # Minimum for local RAG (no LangSmith)
   OLLAMA_MODEL=llama3
   OLLAMA_EMBED_MODEL=nomic-embed-text
   # Optional
   # OLLAMA_BASE_URL=http://127.0.0.1:11434

   # LangSmith (tracing + evaluation)
   LANGCHAIN_API_KEY=your_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=rag-homework

   # Optional: dataset name for upload / experiments (default: rag-course-eval)
   # LANGSMITH_DATASET=rag-course-eval

   # Optional: judge LLM (defaults to Ollama, same host as RAG)
   # JUDGE_PROVIDER=ollama
   # JUDGE_OLLAMA_MODEL=llama3

   # Optional: Gemini for RAG or judge
   # GOOGLE_API_KEY=...
   # GEMINI_MODEL=gemini-2.0-flash
   ```

3. Add PDFs under `data/pdfs/` and edit `data/eval_dataset.json` (20 QA pairs with real content for the assignment).


## Suggested workflow for the assignment

**Step 1 — Prepare data.**
Place all 5 course PDFs in `data/pdfs/` and write 20 real QA pairs (8 easy, 8 medium, 4 hard) in `data/eval_dataset.json`. The eval dataset is your ground truth for every later step, so finish it before running anything.

**Step 2 — Ingest with defaults.**
```bash
python -m src.ingest --force
```
Embeds the PDFs into Chroma with the default chunk settings (1000 / 200). `--force` ensures a clean collection.

**Step 3 — Smoke-test the RAG pipeline.**
```bash
python -m src.rag_pipeline -q "What is regression testing?"
```
Try a few easy questions interactively. This catches obvious problems (wrong model, empty collection, bad embeddings) before you spend time on automated evaluation.

**Step 4 — Manual testing (Part 1 deliverable).**
Pick at least 5 questions from the dataset, run them one by one, and record the question, expected answer, RAG answer, and pass/fail in `results/manual_test_results.md`. Do this before LangSmith so you have a qualitative understanding of where the system succeeds and fails.

**Step 5 — Upload dataset to LangSmith.**
```bash
python -m src.evaluate_langsmith --upload-dataset
```
Creates the `rag-course-eval` dataset in LangSmith. Run once; use `--overwrite-dataset` if you later edit `eval_dataset.json`.

**Step 6 — Baseline evaluation (Part 2 deliverable).**
```bash
python -m src.evaluate_langsmith --run-baseline
```
Runs the full 20-question eval with all four LLM-as-judge metrics and the default config. This is the baseline all experiments compare against. Note overall and per-difficulty scores, and pick 3+ failure cases for your report.

**Step 7 — Parameter sweep (Part 3 deliverable).**
```bash
python -m src.experiments --dry-run          # preview the 11 configs
python -m src.experiments                    # full sweep (re-ingests per config)
```
Each config re-embeds with different chunk/overlap, queries with different top_k, and evaluates through LangSmith. Results append to `results/experiment_results.csv`. This is the longest step (many embedding + judge calls).

**Step 8 — Gemini comparison (bonus).**
```bash
python -m src.experiments --with-gemini --gemini-config-index 4
```
Runs the best Ollama config (or index 4 as default) but swaps Ollama for Gemini as the generation model. Compares correctness, latency, and cost. Requires `GOOGLE_API_KEY` in `.env`.

**Step 9 — Write the final report.**
Open `results/final_report.md` and summarize: methodology, baseline metrics (overall + by difficulty), 3+ failure cases, experiment comparison table, optimal config with justification, and (if applicable) Gemini vs Ollama findings.

## Detailed Usage

Run commands from the **project root** so `python -m src.<module>` resolves correctly.

### Ingest documents into Chroma

```bash
python -m src.ingest
python -m src.ingest --chunk-size 1000 --chunk-overlap 200 --force
```

`--force` deletes the existing Chroma collection (`rag_docs` by default) before rebuilding. Defaults: `data/pdfs`, `chroma_db/`, embedding model from `OLLAMA_EMBED_MODEL` or `nomic-embed-text`.

### Ask a single question (RAG)

```bash
python -m src.rag_pipeline -q "What is a unit test?"
python -m src.rag_pipeline -q "..." --top-k 5 --provider ollama
python -m src.rag_pipeline -q "..." --provider gemini
```

Requires a populated `chroma_db/` for the chosen `--collection` (default `rag_docs`).

### LangSmith: upload evaluation dataset

```bash
python -m src.evaluate_langsmith --upload-dataset
python -m src.evaluate_langsmith --upload-dataset --overwrite-dataset
```

Uses `data/eval_dataset.json` unless you pass `--eval-path`. Dataset name defaults to `LANGSMITH_DATASET` or `rag-course-eval`.

### LangSmith: baseline evaluation (full judge suite)

Runs the RAG pipeline on every dataset example with four LLM-as-judge metrics (correctness, relevance, hallucination, conciseness). The judge model is configured separately via `JUDGE_PROVIDER` / `JUDGE_OLLAMA_MODEL` (or Gemini + `GOOGLE_API_KEY`).

```bash
python -m src.evaluate_langsmith --run-baseline
python -m src.evaluate_langsmith --run-baseline --experiment-prefix my_baseline --top-k 4
```

Sequential evaluation by default (`RAG_EVAL_MAX_CONCURRENCY=0`) to avoid overloading Ollama; increase if your stack can handle parallel calls.

### Experiments: chunk size, overlap, retrieval depth

Runs **11** predefined configurations. Each step **re-ingests** with the new chunk settings (unless `--skip-ingest`), then runs the same LangSmith evaluator suite and **appends** rows to `results/experiment_results.csv`.

```bash
python -m src.experiments --dry-run
python -m src.experiments
python -m src.experiments --experiment-index 4
python -m src.experiments --with-gemini --gemini-config-index 4
```

- `--skip-ingest`: only use with a **single** `--experiment-index`; otherwise all trials share one stale index (a warning is printed).
- `--with-gemini`: after the sweep, one extra run using Gemini on the grid row selected by `--gemini-config-index` (requires `GOOGLE_API_KEY`; uses current Chroma from the last ingest).

## Environment variables (reference)

| Variable | Role |
|----------|------|
| `OLLAMA_MODEL` | Chat model for RAG (default `llama3`). |
| `OLLAMA_EMBED_MODEL` | Embedding model for Chroma (default `nomic-embed-text`). |
| `OLLAMA_BASE_URL` | Ollama server URL if not default. |
| `LANGCHAIN_API_KEY` | LangSmith API key. |
| `LANGCHAIN_TRACING_V2` | Set `true` to trace LangChain runs. |
| `LANGCHAIN_PROJECT` | LangSmith project name. |
| `LANGSMITH_DATASET` | Dataset name for upload / experiments. |
| `JUDGE_PROVIDER` | `ollama` or `gemini` for evaluators. |
| `JUDGE_OLLAMA_MODEL` | Judge chat model (defaults toward `OLLAMA_MODEL`). |
| `GOOGLE_API_KEY` / `GEMINI_MODEL` | Gemini for RAG or judge. |
| `RAG_EVAL_MAX_CONCURRENCY` | LangSmith `evaluate` concurrency (default `0` = sequential). |

## Troubleshooting

- **Empty or missing Chroma**: Run `python -m src.ingest --force` after adding PDFs.
- **Embedding errors**: Ensure the embed model is pulled in Ollama (`ollama pull nomic-embed-text`).
- **LangSmith auth errors**: Check `LANGCHAIN_API_KEY` and that the project exists or can be created.
- **Gemini errors**: Confirm `GOOGLE_API_KEY` and model name (e.g. `gemini-2.0-flash`) match your account.
