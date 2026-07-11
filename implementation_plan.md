# Refined Implementation Plan: Vectorless RAG Engine & ChromaDB Benchmark

This document details the exact technical design and a step-by-step time estimation for implementing the **Vectorless RAG Engine** and its benchmark against **ChromaDB**.

---

## 1. Estimated Time Commitment (Total: 7.5 - 12 Hours)

Given your target side-project bandwidth of **10–15 hours/week**, this entire project can be completed, benchmarked, and documented within **a single week's cycle**.

```
[Phase 1: DB & Parser] ██░░░░░░░░ 2.0h
[Phase 2: RAG Engine ] █░░░░░░░░░ 1.5h
[Phase 3: Benchmark  ] ███░░░░░░░ 3.0h
[Phase 4: Optimization] █░░░░░░░░░ 1.5h
[Phase 5: Content    ] ██░░░░░░░  2.5h
Total: ~10.5 Hours (1-Week Sprint)
```

### Phase-by-Phase Breakdown

*   **Phase 1: Document Parser & SQLite Setup (1.5 – 2.5 Hours)**
    *   Set up a simple Markdown/text file reader.
    *   Design the SQLite database schema with `fts5` enabled.
    *   Write the insertion logic (chunks, metadata, and binary embeddings).
*   **Phase 2: RAG Hybrid Search Engine (1.0 – 1.5 Hours)**
    *   Implement keyword lookup using FTS5 MATCH queries.
    *   Implement query embedding NumPy cosine similarity computation.
    *   Integrate simple LLM call wrapper (direct OpenAI/Anthropic API).
*   **Phase 3: ChromaDB Benchmarking Setup (2.0 – 3.0 Hours)**
    *   Set up ChromaDB persistent client wrapper.
    *   Write random document set generator (supporting mock vectors to test offline).
    *   Write metrics logger (QPS, indexing speed, disk storage, import latency).
*   **Phase 4: Debugging, Refactoring & 100-Line Limit Verification (1.0 – 2.0 Hours)**
    *   Trim down any excess code to keep the core RAG script under the 100-line limit (for branding appeal).
    *   Resolve database lock conditions or numpy byte conversions.
*   **Phase 5: Writing README & Substack Draft #4 (2.0 – 3.0 Hours)**
    *   Run benchmarks and paste the comparative table into the README.
    *   Draft the Substack essay comparing setup friction, dependencies, and performance.

---

## 2. Technical Architecture & Data Schemas

To keep execution fast and low-friction, here is the exact database layout and computation flow.

### A. Database Layout (`rag.db`)

We will use two tables in SQLite:
1.  **`chunks_fts` (Virtual FTS5 Table):** Used for instant keyword search.
2.  **`chunks` (Standard Table):** Used to store raw content and binary serialized embeddings.

```sql
-- Virtual table for keyword candidate indexing
CREATE VIRTUAL TABLE chunks_fts USING fts5(content, chunk_id UNINDEXED);

-- Standard table for metadata, content, and numpy blobs
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    filepath TEXT,
    content TEXT,
    embedding BLOB
);
```

### B. Core NumPy Cosine Similarity Code Pattern

To store high-dimensional embeddings (e.g., 1536 float values) efficiently in SQLite, we serialize the NumPy array to bytes and read it back directly.

```python
import sqlite3
import numpy as np

# Serialize:
embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

# Deserialize & Calculate Cosine Similarity:
# 1. Retrieve candidates from SQLite
cursor.execute("SELECT content, embedding FROM chunks WHERE id IN (SELECT chunk_id FROM chunks_fts MATCH ? LIMIT 50)", (query_keyword,))
rows = cursor.fetchall()

# 2. Extract embeddings and form a 2D matrix
candidate_embeddings = np.frombuffer(b"".join([r[1] for r in rows]), dtype=np.float32).reshape(len(rows), 1536)

# 3. Compute cosine similarity in one NumPy operation
# (assuming query_vector is normalized and candidate_embeddings are normalized)
similarities = np.dot(candidate_embeddings, query_vector)
```

---

## 3. Benchmarking Framework Details

The benchmark script (`benchmark.py`) will automatically measure the following profiles:

| Metric | Vectorless RAG (SQLite + NumPy) | ChromaDB (Local SQLite/DuckDB/Parquet backend) |
| :--- | :--- | :--- |
| **Setup Friction** | Zero dependencies, built-in SQLite3 | Needs `chromadb` pip installation and native compiler |
| **Import / Init Time** | `import sqlite3` ($\approx 0.05\text{s}$) | `import chromadb` ($\approx 0.8\text{ - }1.5\text{s}$) |
| **Indexing Latency** | Sequential SQLite transactions | ChromaDB sqlite metadata index insertion |
| **Disk Size** | Singular `.db` file (Highly optimized SQLite paging) | Directory folder hierarchy (typically 2-4x database size) |
| **Query Latency (10k docs)**| SQL FTS MATCH (e.g., limit 50) + local matrix multiplication | Full vector index traverse (HNSW) |

---

## 4. Verification Plan

### Automated Run Command
```bash
python projects/brand-builders/vectorless-rag/benchmark.py --num-docs 1000 --num-queries 100
```
This command runs the offline generator (1,000 synthetic documents, 1,536-dimensional mock embeddings), executes the indexer on both systems, runs 100 queries, and prints the performance matrix.
