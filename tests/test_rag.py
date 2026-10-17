import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import VectorlessRAG, chunk_text, read_and_chunk_file, sanitize_fts_query


@pytest.fixture
def rag(tmp_path):
    return VectorlessRAG(str(tmp_path / "t.db"))


def docs(*texts, embs=None):
    return [{"filepath": f"d{i}.md", "content": t, **({"embedding": embs[i]} if embs else {})}
            for i, t in enumerate(texts)]


# ── chunking ──────────────────────────────────────────────────────────────
def test_chunk_overlap():
    chunks = chunk_text("a" * 100, chunk_size=40, overlap=10)
    assert chunks[0] == "a" * 40 and len(chunks) == 3  # no boundaries -> hard cuts at 0-40, 30-70, 60-100
    assert "".join(c[10:] if i else c for i, c in enumerate(chunks)) == "a" * 100


def test_chunk_respects_word_boundaries():
    text = " ".join(f"word{i}" for i in range(200))
    words = set(text.split())
    for c in chunk_text(text, chunk_size=100, overlap=20):
        assert len(c) <= 100
        assert set(c.split()) <= words  # no word sliced in half


def test_chunk_prefers_sentence_boundary():
    text = ("Alpha beta gamma delta. " * 3) + "Epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
    first = chunk_text(text, chunk_size=80, overlap=10)[0]
    assert first.endswith(". ")


def test_chunk_covers_all_text():
    text = " ".join(f"w{i}" for i in range(500))
    joined = " ".join(chunk_text(text, 120, 30))
    assert all(f"w{i}" in joined.split() for i in range(500))


def test_chunk_short_and_empty():
    assert chunk_text("hi", 500, 50) == ["hi"]
    assert chunk_text("", 500, 50) == [""]


def test_chunk_bad_overlap_returns_whole():
    assert chunk_text("abcdef", 3, 3) == ["abcdef"]


def test_read_and_chunk_file(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("x" * 120)
    assert len(read_and_chunk_file(str(p), 100, 20)) == 2


# ── schema ────────────────────────────────────────────────────────────────
def test_schema_created(rag):
    conn = sqlite3.connect(rag.db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {"chunks", "chunks_fts"} <= names


def test_init_idempotent(rag):
    VectorlessRAG(rag.db_path)  # must not raise


# ── sanitization ──────────────────────────────────────────────────────────
def test_sanitize():
    assert sanitize_fts_query('C++ "don\'t" -scale AND (x)') == \
        '"C" OR "don" OR "t" OR "scale" OR "AND" OR "x"'
    assert sanitize_fts_query("!!! ???") == ""
    assert sanitize_fts_query("") == ""


@pytest.mark.parametrize("q", ['C++', "don't scale", "error: 404", '"unbalanced', "a AND", "(", "*", "NOT"])
def test_fts_syntax_never_crashes(rag, q):
    rag.add_documents(docs("we use C++ and error 404 and don't scale"))
    rag.query(q)  # no exception


def test_punctuation_only_returns_empty_not_garbage(rag):
    rag.add_documents(docs("alpha", "beta"))
    assert rag.query("???") == []


# ── pure BM25 mode ────────────────────────────────────────────────────────
def test_pure_fts_no_embeddings_needed(rag):
    rag.add_documents(docs("sqlite is great", "chroma is a vector db"))
    res = rag.query("sqlite")
    assert len(res) == 1 and res[0][0]["filepath"] == "d0.md"


def test_no_match_returns_empty_not_arbitrary_rows(rag):
    rag.add_documents(docs("alpha", "beta"))
    assert rag.query("zzzz") == []


def test_bm25_orders_by_relevance(rag):
    rag.add_documents(docs("cat " + "filler " * 30, "cat cat cat dog", "nothing here"))
    res = rag.query("cat", top_k=2)
    assert res[0][0]["filepath"] == "d1.md"
    assert res[0][1] >= res[1][1]


def test_bm25_ranked_before_limit(rag):
    # best match inserted last; ORDER BY rank must still surface it under a small fts_limit
    texts = ["cat " + "filler " * 20 for _ in range(30)] + ["cat cat cat cat"]
    rag.add_documents(docs(*texts))
    assert rag.query("cat", fts_limit=3, top_k=1)[0][0]["filepath"] == "d30.md"


def test_deterministic(rag):
    rag.add_documents(docs("apple pie", "apple tart", "apple juice"))
    assert rag.query("apple") == rag.query("apple")


# ── hybrid mode ───────────────────────────────────────────────────────────
def test_hybrid_cosine_breaks_bm25_tie(rag):
    e = np.eye(3).tolist()
    rag.add_documents(docs("apple pie", "apple tart", "apple juice", embs=e))
    res = rag.query("apple", query_embedding=[0, 0, 1], top_k=3)
    assert res[0][0]["filepath"] == "d2.md"


def test_hybrid_null_and_mismatched_embeddings_ok(rag):
    rag.add_documents([
        {"filepath": "a", "content": "apple one", "embedding": [1, 0, 0]},
        {"filepath": "b", "content": "apple two"},                       # NULL
        {"filepath": "c", "content": "apple three", "embedding": [1, 0]},  # wrong dim
    ])
    assert len(rag.query("apple", query_embedding=[1, 0, 0], top_k=3)) == 3


def test_cosine_known_vectors(rag):
    rag.add_documents(docs("kw x", "kw y", embs=[[1, 0], [0, 1]]))
    res = rag.query("kw", query_embedding=[1, 0], top_k=2)
    assert res[0][0]["filepath"] == "d0.md" and res[0][1] > res[1][1]


# ── idempotency, metadata, migration ──────────────────────────────────────
def test_delete_by_filepath_replaces(rag):
    rag.add_documents([{"filepath": "f.md", "content": "zebra one"}, {"filepath": "g.md", "content": "zebra two"}])
    assert rag.delete_by_filepath("f.md") == 1
    rag.add_documents([{"filepath": "f.md", "content": "zebra new"}])
    res = rag.query("zebra", top_k=10)
    assert sorted(c["content"] for c, _ in res) == ["zebra new", "zebra two"]


def test_chunk_index_roundtrip(rag):
    rag.add_documents([{"filepath": "f.md", "content": "zebra", "chunk_index": 7}])
    assert rag.query("zebra")[0][0]["chunk_index"] == 7


def test_migrates_old_schema(tmp_path):
    p = str(tmp_path / "old.db")
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, filepath TEXT, content TEXT, embedding BLOB)")
    conn.execute("INSERT INTO chunks (filepath, content) VALUES ('o.md', 'legacy zebra')")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, chunk_id UNINDEXED, tokenize='unicode61')")
    conn.execute("INSERT INTO chunks_fts (content, chunk_id) VALUES ('legacy zebra', 1)")
    conn.commit(); conn.close()
    res = VectorlessRAG(p).query("zebra")
    assert res[0][0]["chunk_index"] is None and res[0][0]["filepath"] == "o.md"


# ── end to end ────────────────────────────────────────────────────────────
def test_end_to_end_ingest_query(rag, tmp_path):
    f = tmp_path / "essay.md"
    f.write_text("Do things that don't scale. " * 40 + "Unique zebra sentence.")
    rag.add_documents([{"filepath": str(f), "content": c} for c in read_and_chunk_file(str(f))])
    top = rag.query("zebra")[0][0]
    assert "zebra" in top["content"] and top["filepath"] == str(f)
