"""RAG retrieval and generation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings

from .prompts import RAG_PROMPT
from .utils import get_config, project_root


# Join retrieved chunk texts into a single string for the prompt
def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


class RAGPipeline:
    """Retrieve chunks from Chroma (Ollama embeddings) and answer with Ollama or Gemini."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "rag_docs",
        embedding_model: str = "nomic-embed-text",
        base_url: str | None = None,
        top_k: int = 4,
        provider: str = "ollama",
        ollama_model: str = "llama3",
        gemini_model: str = "gemini-2.5-flash",
        google_api_key: str | None = None,
    ) -> None:
        self._persist_directory = Path(persist_directory)
        self._collection_name = collection_name
        self._top_k = top_k
        self._provider = provider.lower().strip()

        # Connect to the existing Chroma collection using Ollama embeddings
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        self._vectorstore = Chroma(
            persist_directory=str(self._persist_directory),
            collection_name=collection_name,
            embedding_function=embeddings,
            create_collection_if_not_exists=False,
        )
        # Retriever returns the top_k most similar chunks for a query
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": top_k})

        # Select the generation LLM based on provider
        if self._provider == "ollama":
            self._llm = ChatOllama(model=ollama_model, base_url=base_url)
        elif self._provider == "gemini":
            key = google_api_key
            if not key:
                raise ValueError("Gemini requires google_api_key or GOOGLE_API_KEY in the environment.")
            self._llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=key)
        else:
            raise ValueError(f"Unknown provider: {provider!r}. Use 'ollama' or 'gemini'.")

        # LCEL chain: fill the prompt template -> send to LLM -> extract string
        self._generation_chain = RAG_PROMPT | self._llm | StrOutputParser()

    @classmethod
    def from_config(cls, **overrides: Any) -> RAGPipeline:
        """Build a pipeline using env vars + defaults from get_config(), with optional overrides."""
        c = get_config(**overrides)
        return cls(
            persist_directory=c["persist_directory"],
            collection_name=c["collection_name"],
            embedding_model=c["embedding_model"],
            base_url=c["base_url"],
            top_k=c["top_k"],
            provider=c["provider"],
            ollama_model=c["ollama_model"],
            gemini_model=c["gemini_model"],
            google_api_key=c.get("google_api_key"),
        )

    def query(self, question: str) -> dict[str, Any]:
        """Retrieve relevant chunks, generate an answer, return both."""
        docs = self._retriever.invoke(question)
        context = _format_context(docs)
        answer = self._generation_chain.invoke({"context": context, "question": question})
        return {"answer": answer, "source_documents": docs}

    def batch_query(self, questions: list[str]) -> list[dict[str, Any]]:
        """Run query() for each question sequentially."""
        return [self.query(q) for q in questions]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline on one question.")
    parser.add_argument("--query", "-q", type=str, help="Question to ask")
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini"],
        default=None,
        help="LLM backend (default: from env / get_config)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name (default: rag_docs)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        help="Chroma persist directory (default: ./chroma_db)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(project_root() / ".env")
    args = parse_args(argv)

    # Only pass CLI flags that were explicitly set (None = use default from config)
    overrides: dict[str, Any] = {}
    if args.provider is not None:
        overrides["provider"] = args.provider
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.collection is not None:
        overrides["collection_name"] = args.collection
    if args.persist_dir is not None:
        overrides["persist_directory"] = args.persist_dir.resolve()

    if not args.query:
        print("Error: pass --query \"...\"", file=sys.stderr)
        return 1

    # Build pipeline from env + overrides, then run the question
    try:
        pipeline = RAGPipeline.from_config(**overrides)
        result = pipeline.query(args.query)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
