# Vectorless RAG Engine & ChromaDB Benchmark

This repository implements a lightweight **Vectorless RAG Engine** and compares its performance directly against **ChromaDB**. It serves as the technical validation for the **"AI Over Hype"** thesis, demonstrating that native SQLite features (FTS5) combined with simple vector math (NumPy) can comfortably replace heavy infrastructure for small-to-mid scale knowledge bases.

---

## Directory Structure

*   [rag.py](rag.py): The core RAG engine codebase (implemented in under 100 lines of Python code using native SQLite and NumPy).
*   [ingest.py](ingest.py): CLI tool to recursively scan, chunk, and index local Markdown/text documents into the database.
*   [query.py](query.py): Interactive REPL and command-line search query runner with optional local LLM grounding via Ollama.
*   [benchmark.py](benchmark.py): The test and measurement runner to compare indexing time, disk footprint, and query execution speed against ChromaDB.
*   [report.py](report.py): Runs a thorough comparative benchmark and generates a rich, self-contained HTML report with Chart.js visualizations.
*   [requirements.txt](requirements.txt): Python dependencies needed for local testing and execution.

---

## Technical Architecture & Database Layout

To keep execution fast and low-friction, the database layout consists of two tables inside a single SQLite file (`rag.db`):

1.  **`chunks_fts` (Virtual FTS5 Table):** Used for instant keyword-based candidate matching.
2.  **`chunks` (Standard Table):** Stores raw file paths, document contents, and binary serialized NumPy embeddings.

### Database Schema
```sql
-- Virtual table for keyword candidate indexing
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, chunk_id UNINDEXED, tokenize='unicode61');

-- Standard table for metadata, content, and binary blobs
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    filepath TEXT,
    content TEXT,
    embedding BLOB
);
```

### In-Memory Hybrid Reranking Logic

1.  **Candidate Retrieval (FTS5):** Match keywords from the user query against `chunks_fts` to retrieve a top candidate pool (e.g., 50 chunks). This bounds the matrix multiplication size.
2.  **Binary Embedding Deserialization:** Convert the stored binary blobs back to Float32 NumPy arrays:
    ```python
    embeddings = [np.frombuffer(row[3], dtype=np.float32) for row in rows]
    ```
3.  **Vectorized Cosine Similarity:** Normalise the candidate embeddings and compute similarity scores with the query vector in a single numpy dot product operation:
    ```python
    stacked = np.vstack(embeddings)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    similarities = np.dot(stacked / norms, normalized_query_vector)
    ```

---

## Benchmarking Framework

The benchmark compares Vectorless RAG (SQLite + NumPy) against ChromaDB across the following metrics:

| Metric | Vectorless RAG (SQLite + NumPy) | ChromaDB (Local SQLite/HNSW backend) |
| :--- | :--- | :--- |
| **Setup Friction** | Zero dependencies, built-in SQLite3 | Needs `chromadb` pip installation and native compiler |
| **Import / Init Time** | `import sqlite3` ($\approx 0.004\text{s}$) | `import chromadb` ($\approx 0.2\text{s}$) |
| **Indexing Latency** | Sequential SQLite transactions | ChromaDB sqlite metadata index insertion |
| **Disk Size** | Singular `.db` file (Highly optimized SQLite paging) | Directory folder hierarchy (typically 2-4x database size) |
| **Query Latency** | SQL FTS MATCH + NumPy matrix multiplication | Full vector index traverse (HNSW) |

---

## Getting Started

### 1. Create a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Core Console Benchmark
```bash
python3 benchmark.py --num-docs 1000 --num-queries 100
```

### 4. Run the HTML Report Generator
Generate a fully responsive, styled HTML report with visual charts comparing the two systems:
```bash
python3 report.py --num-docs 1000 --num-queries 100
# Generates benchmark_report.html
```

### 5. Ingest and Query Local Documents
To index a folder of actual Markdown/text essays (e.g. from `datasets/paul-graham/essays`):
```bash
python3 ingest.py --dir ./datasets/paul-graham/essays --db pg.db
```
Then query the index interactively or single-shot:
```bash
# Interactive REPL mode
python3 query.py --db pg.db

# Single-shot query
python3 query.py --db pg.db -q "Why should you build things that do not scale?"
```

---

## Branding & Alignment Hook

This project serves as the technical validation for the **"AI Over Hype"** thesis (Essay #4). The objective is to highlight cost efficiency and show that standard relational database features (SQLite FTS5) + basic vector math (NumPy) can comfortably replace heavy infrastructure for 95% of corporate knowledge bases.
