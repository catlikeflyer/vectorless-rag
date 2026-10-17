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
from rag import VectorlessRAG, generate_answer, embed_texts, DEFAULT_GEN_MODEL

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"


def run_query(rag: VectorlessRAG, query: str, top_k: int, generate: bool, model: str, embed_model: str = None):
    """Execute a query and print results.

    Pure FTS5/BM25 by default; with embed_model, BM25 candidates are fused with
    cosine similarity (RRF). Input is sanitized in rag.py, so any text is safe.
    """
    q_emb = None
    if embed_model:
        try:
            q_emb = embed_texts([query], embed_model)[0]
        except Exception as e:
            print(f"{YELLOW}Embedding failed ({e}); falling back to pure BM25.{RESET}")
    results = rag.query(query, q_emb, fts_limit=50, top_k=top_k)

    if not results:
        print(f"{YELLOW}No results found for: '{query}'{RESET}\n")
        return

    print(f"\n{BOLD}{CYAN}Results for: \"{query}\"{RESET}  ({len(results)} chunk(s))\n")
    for rank, (chunk, score) in enumerate(results, 1):
        src = chunk.get("filepath", "unknown")
        if chunk.get("chunk_index") is not None:
            src += f"  (chunk {chunk['chunk_index']})"
        snippet = chunk["content"].strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        print(f"  {BOLD}[{rank}]{RESET} {DIM}{src}{RESET}")
        print(f"      {snippet}")
        label = "rrf score" if q_emb is not None else "bm25 score"
        print(f"      {DIM}{label}: {score:.4f}{RESET}\n")

    if generate:
        chunks_only = [r[0] for r in results]
        print(f"{BOLD}Generating answer via Ollama ({model})...{RESET}")
        try:
            answer = generate_answer(chunks_only, query, model=model)
            print(f"\n{BOLD}Answer:{RESET}\n{answer}\n")
        except Exception as e:
            print(f"{YELLOW}[ollama] {e}{RESET}\n")


def repl(rag: VectorlessRAG, top_k: int, generate: bool, model: str, embed_model: str = None):
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
        run_query(rag, query, top_k, generate, model, embed_model)


def main():
    parser = argparse.ArgumentParser(description="Query the Vectorless RAG database")
    parser.add_argument("--db",       default="rag.db",              help="SQLite DB path (default: rag.db)")
    parser.add_argument("-q", "--query", metavar="TEXT",             help="One-shot query (omit for interactive REPL)")
    parser.add_argument("--top-k",    type=int, default=3,           help="Number of chunks to return (default: 3)")
    parser.add_argument("--generate", action="store_true",           help="Pipe results into Ollama for a full answer")
    parser.add_argument("--model",    default=DEFAULT_GEN_MODEL,     help="Ollama model for --generate (default: $OLLAMA_MODEL or llama3.2)")
    parser.add_argument("--embed-model", default=None,               help="Ollama embedding model for hybrid search (must match ingest)")
    args = parser.parse_args()

    rag = VectorlessRAG(args.db)

    if args.query:
        run_query(rag, args.query, args.top_k, args.generate, args.model, args.embed_model)
    else:
        repl(rag, args.top_k, args.generate, args.model, args.embed_model)


if __name__ == "__main__":
    main()
