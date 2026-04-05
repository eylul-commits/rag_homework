from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_pdf_documents(pdf_dir: Path) -> list:
    paths = sorted(pdf_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")
    docs: list = []
    for path in paths:
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())
    return docs


def delete_collection_if_exists(persist_directory: Path, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=str(persist_directory))
    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)


def ingest(
    pdf_dir: Path,
    persist_directory: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    base_url: str | None,
    force: bool,
) -> int:
    # Guard invalid splitter settings
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Chroma needs a directory on disk
    persist_directory.mkdir(parents=True, exist_ok=True)

    # Start fresh for this collection when requested
    if force:
        delete_collection_if_exists(persist_directory, collection_name)

    # Load all PDFs, then split into overlapping text chunks
    raw_docs = load_pdf_documents(pdf_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    splits = splitter.split_documents(raw_docs)

    # Vectorize chunks with Ollama and write to Chroma
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

    return len(splits)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma with Ollama embeddings.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=root / "data" / "pdfs",
        help="Directory containing PDF files (default: data/pdfs)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=root / "chroma_db",
        help="Chroma persistence directory (default: chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default="rag_docs",
        help="Chroma collection name (default: rag_docs)",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Ollama embedding model (default: env OLLAMA_EMBED_MODEL or nomic-embed-text)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama server URL (default: env OLLAMA_BASE_URL or Ollama default)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing collection and rebuild from PDFs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(project_root() / ".env")
    args = parse_args(argv)

    import os

    embedding_model = args.embedding_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    base_url = args.base_url or os.getenv("OLLAMA_BASE_URL")

    args.pdf_dir = args.pdf_dir.resolve()
    args.persist_dir = args.persist_dir.resolve()

    if not args.pdf_dir.is_dir():
        print(f"Error: PDF directory does not exist: {args.pdf_dir}", file=sys.stderr)
        return 1

    try:
        n = ingest(
            pdf_dir=args.pdf_dir,
            persist_directory=args.persist_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=embedding_model,
            base_url=base_url,
            force=args.force,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Ingested {n} chunks into {args.persist_dir} (collection={args.collection}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
