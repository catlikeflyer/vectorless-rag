import os
import re
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_GEN_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

RRF_K = 60  # standard Reciprocal Rank Fusion constant

def sanitize_fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: quoted alphanumeric tokens joined by OR.
    Strips FTS5 syntax (quotes, operators, punctuation). Returns '' if nothing usable."""
    tokens = re.findall(r"\w+", text or "", flags=re.UNICODE)
    return " OR ".join(f'"{t}"' for t in tokens)

def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    """Embed texts with a local Ollama embedding model (e.g. nomic-embed-text)."""
    import ollama
    return ollama.embed(model=model, input=texts)["embeddings"]

def _cut_point(text: str, start: int, end: int) -> int:
    """Best split position in (start, end]: paragraph > sentence > whitespace, searched
    only in the back half of the window; falls back to a hard cut at `end`."""
    lo = start + (end - start) // 2
    window = text[lo:end]
    for pat in ("\n\n", ". ", "? ", "! ", ".\n", "\n", " "):
        i = window.rfind(pat)
        if i != -1:
            return lo + i + len(pat)
    return end

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into ~chunk_size-char chunks at paragraph/sentence/word boundaries,
    with ~overlap chars carried into the next chunk (snapped to a word start)."""
    if chunk_size - overlap <= 0 or len(text) <= chunk_size: return [text]
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _cut_point(text, start, end)
        chunks.append(text[start:end])
        if end >= n: break
        nxt = max(end - overlap, start + 1)
        if nxt > 0 and not text[nxt - 1].isspace():  # don't begin mid-word
            m = re.search(r"\s", text[nxt:end])
            if m: nxt += m.end()
        start = nxt
    return chunks

def read_and_chunk_file(filepath: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return chunk_text(f.read(), chunk_size, overlap)

class VectorlessRAG:
    def __init__(self, db_path: str = "rag.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, filepath TEXT, content TEXT, embedding BLOB, chunk_index INTEGER)")
                cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
                if "chunk_index" not in cols:  # migrate pre-existing databases
                    conn.execute("ALTER TABLE chunks ADD COLUMN chunk_index INTEGER")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_filepath ON chunks(filepath)")
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, chunk_id UNINDEXED, tokenize='unicode61')")
        finally:
            conn.close()

    def delete_by_filepath(self, filepath: str) -> int:
        """Remove all chunks (and FTS rows) for a file. Returns number removed."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            with conn:
                ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE filepath = ?", (filepath,))]
                conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(i,) for i in ids])
                conn.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
            return len(ids)
        finally:
            conn.close()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Insert chunks. 'embedding' is optional; omit it for pure FTS5/BM25 retrieval."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                cursor = conn.cursor()
                for doc in documents:
                    emb = doc.get('embedding')
                    emb_bytes = np.array(emb, dtype=np.float32).tobytes() if emb is not None else None
                    cursor.execute("INSERT INTO chunks (filepath, content, embedding, chunk_index) VALUES (?, ?, ?, ?)", (doc['filepath'], doc['content'], emb_bytes, doc.get('chunk_index')))
                    cursor.execute("INSERT INTO chunks_fts (content, chunk_id) VALUES (?, ?)", (doc['content'], cursor.lastrowid))
        finally:
            conn.close()

    def query(self, query_text: str, query_embedding: Optional[List[float]] = None, fts_limit: int = 50, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve top_k chunks.

        1. FTS5 candidates, ordered by BM25 (sanitized query, so punctuation never errors).
        2. No query_embedding -> pure BM25; score = -bm25 (higher is better).
        3. With query_embedding -> cosine rerank of candidates fused with BM25 via RRF;
           score = RRF score (higher is better).
        """
        fts_query = sanitize_fts_query(query_text)
        if not fts_query: return []

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            rows = conn.execute(
                "SELECT c.id, c.filepath, c.content, c.embedding, bm25(chunks_fts) AS score, c.chunk_index "
                "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.chunk_id "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?", (fts_query, fts_limit)).fetchall()
        finally:
            conn.close()

        if not rows: return []
        chunk = lambda r: {"id": r[0], "filepath": r[1], "content": r[2], "chunk_index": r[5]}

        if query_embedding is None:
            return [(chunk(r), -float(r[4])) for r in rows[:top_k]]

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0: q_vec /= q_norm

        def vec(r):
            v = np.frombuffer(r[3], dtype=np.float32) if r[3] else None
            return v if v is not None and v.shape == q_vec.shape else np.zeros_like(q_vec)
        stacked = np.vstack([vec(r) for r in rows])
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        similarities = np.dot(stacked / norms, q_vec)

        # RRF: rows are in BM25 order; tied BM25 scores share a rank. Cosine ranked separately.
        bm_rank, prev = [], None
        for i, r in enumerate(rows):
            if prev is None or r[4] != prev[0]: prev = (r[4], i)
            bm_rank.append(prev[1])
        cos_order = np.argsort(-similarities, kind="stable")
        cos_rank = {int(i): rank for rank, i in enumerate(cos_order)}
        fused = [(chunk(r), 1.0 / (RRF_K + bm_rank[i] + 1) + 1.0 / (RRF_K + cos_rank[i] + 1)) for i, r in enumerate(rows)]
        return sorted(fused, key=lambda x: x[1], reverse=True)[:top_k]

def generate_answer(chunks: List[Dict[str, Any]], query: str, model: str = DEFAULT_GEN_MODEL) -> str:
    """Answer from retrieved chunks via a local Ollama model (override with OLLAMA_MODEL)."""
    import ollama
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    try:
        return ollama.chat(model=model, messages=[
            {"role": "system", "content": "Answer using only the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ])["message"]["content"]
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed ({type(e).__name__}: {e}). Make sure Ollama is running "
            f"('ollama serve') and the model is pulled ('ollama pull {model}'); "
            f"use --model or OLLAMA_MODEL to choose another.") from e

