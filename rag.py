import os
import sqlite3
import numpy as np
from typing import List, Dict, Any, Tuple

try:
    import ollama as _ollama
except ImportError:
    _ollama = None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if chunk_size <= overlap:
            break
        start += (chunk_size - overlap)
    return chunks or [text]

def read_and_chunk_file(filepath: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Read a text/markdown file and return overlapping chunks."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return chunk_text(f.read(), chunk_size, overlap)

class VectorlessRAG:
    """SQLite FTS5 + NumPy In-Memory Cosine Similarity Hybrid Search Engine."""
    def __init__(self, db_path: str = "rag.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, filepath TEXT, content TEXT, embedding BLOB
            )""")
            conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, chunk_id UNINDEXED, tokenize='unicode61'
            )""")
            conn.commit()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Insert chunks with serialized embeddings into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for doc in documents:
                emb_bytes = np.array(doc['embedding'], dtype=np.float32).tobytes()
                cursor.execute(
                    "INSERT INTO chunks (filepath, content, embedding) VALUES (?, ?, ?)",
                    (doc['filepath'], doc['content'], emb_bytes)
                )
                cursor.execute(
                    "INSERT INTO chunks_fts (content, chunk_id) VALUES (?, ?)",
                    (doc['content'], cursor.lastrowid)
                )
            conn.commit()

    def query(self, query_text: str, query_embedding: List[float], fts_limit: int = 50, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Hybrid search matching terms via FTS5, then reranking with NumPy cosine similarity."""
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                SELECT c.id, c.filepath, c.content, c.embedding FROM chunks c
                JOIN chunks_fts f ON c.id = f.chunk_id
                WHERE chunks_fts MATCH ? LIMIT ?
                """, (query_text, fts_limit))
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                cursor.execute("SELECT id, filepath, content, embedding FROM chunks LIMIT ?", (fts_limit,))
                rows = cursor.fetchall()

        if not rows:
            return []

        embeddings = [np.frombuffer(r[3], dtype=np.float32) for r in rows]
        norms = [np.linalg.norm(e) for e in embeddings]
        stacked = np.vstack([e / n if n > 0 else e for e, n in zip(embeddings, norms)])
        similarities = np.dot(stacked, q_vec)

        results = []
        for i, row in enumerate(rows):
            results.append(({"id": row[0], "filepath": row[1], "content": row[2]}, float(similarities[i])))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def generate_answer(chunks: List[Dict[str, Any]], query: str, model: str = "qwen3.5:4b-mlx") -> str:
    """Call a local Ollama model with retrieved context chunks as grounding."""
    if _ollama is None:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    response = _ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Answer using only the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response["message"]["content"]
