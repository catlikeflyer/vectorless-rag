"""
ingest.py — Load real markdown/text files into the Vectorless RAG database.

Usage:
    python ingest.py --dir ./docs                      # ingest a whole folder
    python ingest.py --files README.md notes.md        # ingest specific files
    python ingest.py --dir ./docs --chunk-size 600     # custom chunk size

Embeddings: By default none are stored and retrieval is pure FTS5/BM25.
Pass --embed-model (an Ollama embedding model, e.g. nomic-embed-text) to also
store real embeddings for hybrid BM25 + cosine reranking.
"""

import os
import sys
import glob
import argparse
from rag import VectorlessRAG, read_and_chunk_file, embed_texts

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


def main():
    parser = argparse.ArgumentParser(description="Ingest real files into the Vectorless RAG DB")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir",   metavar="PATH",     help="Directory to recursively scan for .md/.txt files")
    src.add_argument("--files", metavar="FILE", nargs="+", help="One or more specific files to ingest")
    parser.add_argument("--db",         default="rag.db", help="SQLite DB path (default: rag.db)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk (default: 500)")
    parser.add_argument("--overlap",    type=int, default=50,  help="Overlap between chunks (default: 50)")
    parser.add_argument("--embed-model", default=None, help="Ollama embedding model for hybrid search (default: none, pure FTS5)")
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

        try:
            embeddings = embed_texts(chunks, args.embed_model) if args.embed_model else [None] * len(chunks)
        except Exception as e:
            print(f"Embedding failed ({e}). Is Ollama running and '{args.embed_model}' pulled?", file=sys.stderr)
            sys.exit(1)
        docs = [
            {"filepath": filepath, "content": chunk, "embedding": emb, "chunk_index": i}
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        replaced = rag.delete_by_filepath(filepath)
        rag.add_documents(docs)
        total_chunks += len(chunks)
        note = f", replaced {replaced} old" if replaced else ""
        print(f"  ✓ {filepath}  ({len(chunks)} chunks{note})")

    print(f"\nDone. Indexed {total_chunks} chunks from {len(files)} file(s) → '{args.db}'")


if __name__ == "__main__":
    main()
