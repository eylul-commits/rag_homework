# RAG homework

## Layout

- `data/pdfs/` — source documents
- `data/eval_dataset.json` — evaluation questions and ground truth
- `src/` — ingestion, RAG pipeline, experiments, LangSmith eval
- `chroma_db/` — persisted Chroma store (generated at runtime)
- `results/` — manual tests, experiment CSV, final report

## Setup

```bash
pip install -r requirements.txt
```

## Usage

_Document ingest, run, and eval commands as you implement them._
