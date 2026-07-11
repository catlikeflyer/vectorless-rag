"""
ingest.py — Load real markdown/text files into the Vectorless RAG database.

Usage:
    python ingest.py --dir ./docs                      # ingest a whole folder
    python ingest.py --files README.md notes.md        # ingest specific files
    python ingest.py --dir ./docs --chunk-size 600     # custom chunk size

Embeddings: Uses random unit-norm vectors as placeholders. The FTS5 keyword
search layer works fully without real embeddings. Swap in a real embedding
model (e.g. nomic-embed-text via Ollama) when you want semantic reranking.
"""

import os
import sys
import glob
import argparse
import numpy as np
from rag import VectorlessRAG, read_and_chunk_file

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".mdx"}


def collect_files(paths: list[str]) -> list[str]:
    """Expand directories and globs into a flat list of supported text files."""
    found = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                        found.append(os.path.join(root, f))
        elif os.path.isfile(p):
            found.append(p)
        else:
            # Try glob pattern
            matches = glob.glob(p, recursive=True)
            found.extend(m for m in matches if os.path.isfile(m))
    return sorted(set(found))


def mock_embedding(dim: int = 1536) -> list[float]:
    """Random unit-norm vector — placeholder until a real embedder is wired in."""
    v = np.random.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def main():
    parser = argparse.ArgumentParser(description="Ingest real files into the Vectorless RAG DB")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir",   metavar="PATH",     help="Directory to recursively scan for .md/.txt files")
    src.add_argument("--files", metavar="FILE", nargs="+", help="One or more specific files to ingest")
    parser.add_argument("--db",         default="rag.db", help="SQLite DB path (default: rag.db)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk (default: 500)")
    parser.add_argument("--overlap",    type=int, default=50,  help="Overlap between chunks (default: 50)")
    parser.add_argument("--dim",        type=int, default=1536, help="Mock embedding dimension (default: 1536)")
    args = parser.parse_args()

    paths = [args.dir] if args.dir else args.files
    files = collect_files(paths)

    if not files:
        print(f"No supported files found in: {paths}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s). Chunking and inserting into '{args.db}'...\n")

    rag = VectorlessRAG(args.db)
    total_chunks = 0

    for filepath in files:
        try:
            chunks = read_and_chunk_file(filepath, args.chunk_size, args.overlap)
        except Exception as e:
            print(f"  [SKIP] {filepath}: {e}")
            continue

        docs = [
            {
                "filepath": filepath,
                "content":  chunk,
                "embedding": mock_embedding(args.dim),
            }
            for chunk in chunks
        ]
        rag.add_documents(docs)
        total_chunks += len(chunks)
        print(f"  ✓ {filepath}  ({len(chunks)} chunks)")

    print(f"\nDone. Indexed {total_chunks} chunks from {len(files)} file(s) → '{args.db}'")


if __name__ == "__main__":
    main()
