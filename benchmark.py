"""
benchmark.py — Vectorless RAG vs ChromaDB Benchmark Runner

Measures: import/init time, indexing latency, disk size, avg query latency, and QPS.
Usage:
    python benchmark.py --num-docs 1000 --num-queries 100
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Import timing: measure wall-clock cost of loading each library
# ──────────────────────────────────────────────────────────────────────────────
_t_rag_import = time.perf_counter()
from rag import VectorlessRAG
_rag_import_s = time.perf_counter() - _t_rag_import

_t_chroma_import = time.perf_counter()
try:
    import chromadb
    _chroma_import_s = time.perf_counter() - _t_chroma_import
except ImportError:
    chromadb = None
    _chroma_import_s = None

EMBEDDING_DIM = 1536
VOCAB = ["scalable", "minimalist", "over-engineered", "database",
         "latency", "vector", "systems", "SQLite", "index"]


# ──────────────────────────────────────────────────────────────────────────────
# Data generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_mock_data(num_docs: int, embedding_dim: int = EMBEDDING_DIM):
    """Generate synthetic documents with randomized content and unit-norm mock embeddings."""
    rng = np.random.default_rng(42)
    docs = []
    for i in range(num_docs):
        words = rng.choice(VOCAB, size=15)
        keyword = VOCAB[i % len(VOCAB)]
        content = (
            f"Document {i}: " + " ".join(words) +
            f". This details RAG information for keyword {keyword}."
        )
        vec = rng.standard_normal(embedding_dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        docs.append({"filepath": f"doc_{i}.md", "content": content, "embedding": vec.tolist()})
    return docs


def _random_query_vectors(num_queries: int, embedding_dim: int = EMBEDDING_DIM):
    rng = np.random.default_rng(99)
    vecs = rng.standard_normal((num_queries, embedding_dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ──────────────────────────────────────────────────────────────────────────────
# Vectorless RAG benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_vectorless(docs, num_queries: int, db_path: str):
    """Profile SQLite FTS5 + NumPy Vectorless RAG implementation."""
    # Client / collection init
    t0 = time.perf_counter()
    rag = VectorlessRAG(db_path)
    init_s = time.perf_counter() - t0

    # Indexing
    t0 = time.perf_counter()
    rag.add_documents(docs)
    index_s = time.perf_counter() - t0

    # Disk footprint
    size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # Query latency
    q_vecs = _random_query_vectors(num_queries)
    t0 = time.perf_counter()
    for i in range(num_queries):
        keyword = VOCAB[i % len(VOCAB)]
        rag.query(keyword, q_vecs[i].tolist(), fts_limit=50, top_k=3)
    elapsed = time.perf_counter() - t0

    query_ms = elapsed / num_queries * 1000
    qps = num_queries / elapsed

    return {
        "import_s": _rag_import_s,
        "init_s":   init_s,
        "index_s":  index_s,
        "size_mb":  size_mb,
        "query_ms": query_ms,
        "qps":      qps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_chroma(docs, num_queries: int, persist_dir: str):
    """Profile persistent ChromaDB implementation."""
    if chromadb is None:
        return None

    # Client / collection init
    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("benchmark_collection")
    init_s = time.perf_counter() - t0

    # Indexing (batch insert)
    t0 = time.perf_counter()
    collection.add(
        ids=[str(i) for i in range(len(docs))],
        documents=[d["content"] for d in docs],
        embeddings=[d["embedding"] for d in docs],
        metadatas=[{"filepath": d["filepath"]} for d in docs],
    )
    index_s = time.perf_counter() - t0

    # Disk footprint (entire directory)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(persist_dir)
        for f in fns
    ) / (1024 * 1024)

    # Query latency — pure vector search (no metadata filter to keep it fair)
    q_vecs = _random_query_vectors(num_queries)
    t0 = time.perf_counter()
    for i in range(num_queries):
        collection.query(query_embeddings=[q_vecs[i].tolist()], n_results=3)
    elapsed = time.perf_counter() - t0

    query_ms = elapsed / num_queries * 1000
    qps = num_queries / elapsed

    return {
        "import_s": _chroma_import_s,
        "init_s":   init_s,
        "index_s":  index_s,
        "size_mb":  size_mb,
        "query_ms": query_ms,
        "qps":      qps,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────────────────────

def _cell(val, fmt):
    return fmt.format(val) if val is not None else "N/A (skipped)"


def print_results(num_docs, num_queries, vl, ch):
    W = 80
    COL = [28, 26, 22]

    def row(label, vl_val, ch_val):
        print(f"  {label:<{COL[0]}} {vl_val:<{COL[1]}} {ch_val:<{COL[2]}}")

    print("\n" + "=" * W)
    print(f"  BENCHMARK RESULTS  —  {num_docs:,} docs  |  {num_queries:,} queries")
    print("=" * W)
    row("Metric", "Vectorless RAG (SQLite+NumPy)", "ChromaDB")
    print("  " + "-" * (W - 2))

    metrics = [
        ("Import latency",    "import_s", "{:.4f} s"),
        ("Client init time",  "init_s",   "{:.4f} s"),
        ("Indexing time",     "index_s",  "{:.4f} s"),
        ("Disk size",         "size_mb",  "{:.3f} MB"),
        ("Avg query latency", "query_ms", "{:.2f} ms"),
        ("Throughput (QPS)",  "qps",      "{:.1f} q/s"),
    ]

    for name, key, fmt in metrics:
        vl_val = _cell(vl.get(key), fmt)
        ch_val = _cell(ch.get(key) if ch else None, fmt)
        row(name, vl_val, ch_val)

    print("=" * W + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vectorless RAG vs ChromaDB Benchmark Runner")
    parser.add_argument("--num-docs",    type=int, default=1000, help="Number of documents to index (default: 1000)")
    parser.add_argument("--num-queries", type=int, default=100,  help="Number of query rounds to run (default: 100)")
    args = parser.parse_args()

    print(f"Generating {args.num_docs:,} synthetic documents "
          f"({EMBEDDING_DIM}-d mock embeddings)...")
    docs = generate_mock_data(args.num_docs)

    temp_dir = tempfile.mkdtemp(prefix="vless_bench_")
    sqlite_db_path = os.path.join(temp_dir, "rag.db")
    chroma_db_dir  = os.path.join(temp_dir, "chroma_db")

    try:
        print("\n[1/2] Benchmarking Vectorless RAG (SQLite FTS5 + NumPy)...")
        vl_metrics = benchmark_vectorless(docs, args.num_queries, sqlite_db_path)
        print("      Done.")

        if chromadb is None:
            print("[2/2] ChromaDB not installed — skipping ChromaDB benchmark.")
            ch_metrics = None
        else:
            print("[2/2] Benchmarking ChromaDB (PersistentClient)...")
            ch_metrics = benchmark_chroma(docs, args.num_queries, chroma_db_dir)
            print("      Done.")

        print_results(args.num_docs, args.num_queries, vl_metrics, ch_metrics)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
