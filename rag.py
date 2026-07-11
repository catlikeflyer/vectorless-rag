import sqlite3
import numpy as np
from typing import List, Dict, Any, Tuple

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    step = chunk_size - overlap
    if step <= 0: return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), step)] or [text]

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
                conn.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, filepath TEXT, content TEXT, embedding BLOB)")
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, chunk_id UNINDEXED, tokenize='unicode61')")
        finally:
            conn.close()

    def add_documents(self, documents: List[Dict[str, Any]]):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            with conn:
                cursor = conn.cursor()
                for doc in documents:
                    emb_bytes = np.array(doc['embedding'], dtype=np.float32).tobytes()
                    cursor.execute("INSERT INTO chunks (filepath, content, embedding) VALUES (?, ?, ?)", (doc['filepath'], doc['content'], emb_bytes))
                    cursor.execute("INSERT INTO chunks_fts (content, chunk_id) VALUES (?, ?)", (doc['content'], cursor.lastrowid))
        finally:
            conn.close()

    def query(self, query_text: str, query_embedding: List[float], fts_limit: int = 50, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0: q_vec /= q_norm

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT c.id, c.filepath, c.content, c.embedding FROM chunks c JOIN chunks_fts f ON c.id = f.chunk_id WHERE chunks_fts MATCH ? LIMIT ?", (query_text, fts_limit))
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                cursor.execute("SELECT id, filepath, content, embedding FROM chunks LIMIT ?", (fts_limit,))
                rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows: return []

        embeddings = [np.frombuffer(r[3], dtype=np.float32) if r[3] else np.zeros(len(q_vec), dtype=np.float32) for r in rows]
        stacked = np.vstack(embeddings)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        similarities = np.dot(stacked / norms, q_vec)

        results = [({"id": r[0], "filepath": r[1], "content": r[2]}, float(s)) for r, s in zip(rows, similarities)]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

def generate_answer(chunks: List[Dict[str, Any]], query: str, model: str = "qwen3.5:4b-mlx") -> str:
    import ollama
    context = "\n\n".join(f"[{i+1}] {c['content']}" for i, c in enumerate(chunks))
    return ollama.chat(model=model, messages=[
        {"role": "system", "content": "Answer using only the provided context. Be concise."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ])["message"]["content"]
