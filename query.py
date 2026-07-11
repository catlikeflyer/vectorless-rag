"""
query.py — Query the Vectorless RAG database interactively or in one-shot mode.

Usage:
    python query.py                                # interactive REPL
    python query.py -q "What is latency?"          # single-shot query
    python query.py -q "chunking" --top-k 5        # return top-5 results
    python query.py -q "SQLite" --generate         # pipe results into Ollama
"""

import argparse
import sys
import numpy as np
from rag import VectorlessRAG, generate_answer

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"


def mock_query_embedding(dim: int = 1536) -> list[float]:
    """
    Placeholder query embedding.  For pure FTS5 keyword results this doesn't
    affect ranking — swap in a real embedder here for semantic reranking.
    """
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def run_query(rag: VectorlessRAG, query: str, top_k: int, generate: bool, model: str):
    """Execute a query and print results.

    Strategy:
    1. Try the raw query as-is (works for single keywords and exact FTS5 phrases).
    2. If no results and the query has multiple words, retry with OR-joined terms
       so each word is searched independently — broader but still ranked by cosine.
    """
    q_emb = mock_query_embedding()
    results = rag.query(query, q_emb, fts_limit=50, top_k=top_k)

    # Fallback: OR search on individual terms
    if not results and " " in query:
        or_query = " OR ".join(query.split())
        results = rag.query(or_query, q_emb, fts_limit=50, top_k=top_k)
        if results:
            query = f"{query}  {DIM}(fuzzy: {or_query}){RESET}"

    if not results:
        print(f"{YELLOW}No results found for: '{query}'{RESET}\n")
        return

    print(f"\n{BOLD}{CYAN}Results for: \"{query}\"{RESET}  ({len(results)} chunk(s))\n")
    for rank, (chunk, score) in enumerate(results, 1):
        src = chunk.get("filepath", "unknown")
        snippet = chunk["content"].strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        print(f"  {BOLD}[{rank}]{RESET} {DIM}{src}{RESET}")
        print(f"      {snippet}")
        print(f"      {DIM}cosine score: {score:.4f}{RESET}\n")

    if generate:
        chunks_only = [r[0] for r in results]
        print(f"{BOLD}Generating answer via Ollama ({model})...{RESET}")
        try:
            answer = generate_answer(chunks_only, query, model=model)
            print(f"\n{BOLD}Answer:{RESET}\n{answer}\n")
        except RuntimeError as e:
            print(f"{YELLOW}[ollama] {e}{RESET}\n")


def repl(rag: VectorlessRAG, top_k: int, generate: bool, model: str):
    print(f"{BOLD}Vectorless RAG — Interactive Query{RESET}")
    print(f"{DIM}Type a keyword or phrase. Press Ctrl-C or Ctrl-D to exit.{RESET}\n")
    while True:
        try:
            query = input(f"{CYAN}query>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not query:
            continue
        run_query(rag, query, top_k, generate, model)


def main():
    parser = argparse.ArgumentParser(description="Query the Vectorless RAG database")
    parser.add_argument("--db",       default="rag.db",              help="SQLite DB path (default: rag.db)")
    parser.add_argument("-q", "--query", metavar="TEXT",             help="One-shot query (omit for interactive REPL)")
    parser.add_argument("--top-k",    type=int, default=3,           help="Number of chunks to return (default: 3)")
    parser.add_argument("--generate", action="store_true",           help="Pipe results into Ollama for a full answer")
    parser.add_argument("--model",    default="qwen3.5:4b-mlx",      help="Ollama model to use with --generate")
    args = parser.parse_args()

    rag = VectorlessRAG(args.db)

    if args.query:
        run_query(rag, args.query, args.top_k, args.generate, args.model)
    else:
        repl(rag, args.top_k, args.generate, args.model)


if __name__ == "__main__":
    main()
