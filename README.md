# Vectorless RAG Engine & ChromaDB Benchmark Base Setup

This directory contains the base setup for the **Vectorless RAG Engine** and its benchmark comparisons against **ChromaDB**. It is designed to prove that you do not need heavy vector databases or complex orchestration layers (like LangChain) for small-to-mid scale RAG workloads.

## Directory Structure

*   [implementation_plan.md](file:///Users/dhnam/Desktop/about-me-okf/projects/brand-builders/vectorless-rag/implementation_plan.md): The detailed design architecture, database schema, NumPy mathematical implementation details, and step-by-step 1-week time estimates.
*   [rag.py](file:///Users/dhnam/Desktop/about-me-okf/projects/brand-builders/vectorless-rag/rag.py): The core RAG engine codebase template (aimed to be implemented in under 100 lines of Python code using native SQLite and NumPy).
*   [benchmark.py](file:///Users/dhnam/Desktop/about-me-okf/projects/brand-builders/vectorless-rag/benchmark.py): The test and measurement runner to compare indexing time, disk footprint, and query execution speed against ChromaDB.
*   [requirements.txt](file:///Users/dhnam/Desktop/about-me-okf/projects/brand-builders/vectorless-rag/requirements.txt): Python dependencies needed for local testing and execution.

## Getting Started

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Benchmark (once fully implemented):**
    ```bash
    python benchmark.py --num-docs 1000 --num-queries 100
    ```

## Branding & Alignment Hook

This project serves as the technical validation for the **"AI Over Hype"** thesis (Essay #4). The objective is to highlight cost efficiency and show that standard relational database features (SQLite FTS5) + basic vector math (NumPy) can comfortably replace heavy infrastructure for 95% of corporate knowledge bases.
