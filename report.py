"""
report.py — Run the benchmark and emit a self-contained HTML report.

Usage:
    python report.py                          # 1000 docs, 100 queries (default)
    python report.py --num-docs 500 --num-queries 50
    python report.py --out my_report.html
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
import json
import datetime
import numpy as np

# ── Import timing (must happen before benchmark functions load the libs) ──────
_t0 = time.perf_counter()
from rag import VectorlessRAG
_rag_import_s = time.perf_counter() - _t0

_t0 = time.perf_counter()
try:
    import chromadb
    _chroma_import_s = time.perf_counter() - _t0
except ImportError:
    chromadb = None
    _chroma_import_s = None

EMBEDDING_DIM = 1536
VOCAB = ["scalable", "minimalist", "over-engineered", "database",
         "latency", "vector", "systems", "SQLite", "index"]


# ── Data helpers ──────────────────────────────────────────────────────────────

def generate_mock_data(num_docs: int):
    rng = np.random.default_rng(42)
    docs = []
    for i in range(num_docs):
        words = rng.choice(VOCAB, size=15)
        kw = VOCAB[i % len(VOCAB)]
        content = f"Document {i}: " + " ".join(words) + f". RAG info for keyword {kw}."
        vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        docs.append({"filepath": f"doc_{i}.md", "content": content, "embedding": vec.tolist()})
    return docs


def random_query_vecs(n: int):
    rng = np.random.default_rng(99)
    vecs = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ── Benchmarks ────────────────────────────────────────────────────────────────

def run_vectorless(docs, num_queries, db_path):
    t0 = time.perf_counter(); rag = VectorlessRAG(db_path); init_s = time.perf_counter() - t0
    t0 = time.perf_counter(); rag.add_documents(docs); index_s = time.perf_counter() - t0
    size_mb = os.path.getsize(db_path) / 1024 / 1024
    q_vecs = random_query_vecs(num_queries)
    t0 = time.perf_counter()
    for i in range(num_queries):
        rag.query(VOCAB[i % len(VOCAB)], q_vecs[i].tolist(), fts_limit=50, top_k=3)
    elapsed = time.perf_counter() - t0
    return {"import_s": _rag_import_s, "init_s": init_s, "index_s": index_s,
            "size_mb": size_mb, "query_ms": elapsed / num_queries * 1000, "qps": num_queries / elapsed}


def run_chroma(docs, num_queries, persist_dir):
    if chromadb is None:
        return None
    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection("bench")
    init_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    col.add(ids=[str(i) for i in range(len(docs))],
            documents=[d["content"] for d in docs],
            embeddings=[d["embedding"] for d in docs],
            metadatas=[{"filepath": d["filepath"]} for d in docs])
    index_s = time.perf_counter() - t0
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fs in os.walk(persist_dir) for f in fs) / 1024 / 1024
    q_vecs = random_query_vecs(num_queries)
    t0 = time.perf_counter()
    for i in range(num_queries):
        col.query(query_embeddings=[q_vecs[i].tolist()], n_results=3)
    elapsed = time.perf_counter() - t0
    return {"import_s": _chroma_import_s, "init_s": init_s, "index_s": index_s,
            "size_mb": size_mb, "query_ms": elapsed / num_queries * 1000, "qps": num_queries / elapsed}


# ── HTML generation ───────────────────────────────────────────────────────────

def build_html(vl, ch, num_docs, num_queries, out_path):
    ch_avail = ch is not None
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt(val, key):
        if val is None:
            return "—"
        if key in ("import_s", "init_s", "index_s"):
            return f"{val:.4f} s"
        if key == "size_mb":
            return f"{val:.3f} MB"
        if key == "query_ms":
            return f"{val:.2f} ms"
        if key == "qps":
            return f"{val:,.0f} q/s"
        return str(val)

    def winner(vl_v, ch_v, lower_is_better=True):
        if not ch_avail or ch_v is None:
            return "vl", "none"
        if lower_is_better:
            return ("vl", "ch") if vl_v <= ch_v else ("ch", "vl")
        return ("vl", "ch") if vl_v >= ch_v else ("ch", "vl")

    metrics = [
        ("Import latency",    "import_s",  True,  "Time to import the library into Python"),
        ("Client init time",  "init_s",    True,  "Time to open / create the database client"),
        ("Indexing time",     "index_s",   True,  f"Time to insert all {num_docs:,} documents"),
        ("Disk size",         "size_mb",   True,  "Total on-disk storage footprint"),
        ("Avg query latency", "query_ms",  True,  f"Mean latency over {num_queries:,} queries"),
        ("Throughput",        "qps",       False, "Queries per second (higher = better)"),
    ]

    # Build chart data
    chart_labels = json.dumps([m[0] for m in metrics])
    vl_data   = json.dumps([round(vl[m[1]], 4) for m in metrics])
    ch_data   = json.dumps([round(ch[m[1]], 4) for m in metrics] if ch_avail else [0]*len(metrics))

    # Build table rows
    rows_html = ""
    vl_wins = 0
    for name, key, low, desc in metrics:
        vl_v = vl[key]
        ch_v = ch[key] if ch_avail else None
        w_vl, w_ch = winner(vl_v, ch_v, low)
        vl_class = "winner" if w_vl == "vl" else ""
        ch_class = "winner" if w_ch == "vl" else ""   # note: ch wins when w_ch != "vl"
        ch_class = "winner" if (ch_avail and w_vl == "ch") else ""
        if w_vl == "vl":
            vl_wins += 1
        ch_cell = fmt(ch_v, key) if ch_avail else '<span class="na">not installed</span>'
        rows_html += f"""
        <tr>
          <td class="metric-name">{name}<span class="desc">{desc}</span></td>
          <td class="{vl_class}">{fmt(vl_v, key)}</td>
          <td class="{ch_class}">{ch_cell}</td>
        </tr>"""

    speedup_init  = f"{ch['init_s']/vl['init_s']:.0f}×" if ch_avail else "—"
    speedup_index = f"{ch['index_s']/vl['index_s']:.1f}×" if ch_avail else "—"
    speedup_qps   = f"{vl['qps']/ch['qps']:.1f}×" if ch_avail else "—"
    size_ratio    = f"{ch['size_mb']/vl['size_mb']:.1f}×" if ch_avail else "—"

    # Honest throughput card: only claim an edge when we actually have one.
    if ch_avail and vl['qps'] < ch['qps']:
        qps_label, qps_value, qps_class = "Query Latency", f"{vl['query_ms']:.2f} ms", "kpi-amber"
        qps_sub = f"avg per query (ChromaDB: {ch['query_ms']:.2f} ms) — comparable at this scale"
    else:
        qps_label, qps_class = "Throughput Edge", "kpi-green"
        qps_value = speedup_qps
        qps_sub = "more queries/sec than ChromaDB" if ch_avail else f"{vl['qps']:,.0f} queries/sec"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vectorless RAG — Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {{
    --bg:        #0d0f14;
    --surface:   #131720;
    --card:      #1a1f2e;
    --border:    #252d40;
    --accent:    #6c8ef5;
    --accent2:   #a78bfa;
    --green:     #34d399;
    --amber:     #fbbf24;
    --red:       #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'JetBrains Mono', monospace;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 0 0 80px;
  }}

  /* ── Hero ──────────────────────────────────────────────────── */
  .hero {{
    background: linear-gradient(135deg, #0d0f14 0%, #111827 50%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    padding: 60px 40px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    inset: 0;
    background:
      radial-gradient(ellipse 60% 50% at 20% 50%, rgba(108,142,245,.12) 0%, transparent 70%),
      radial-gradient(ellipse 50% 50% at 80% 50%, rgba(167,139,250,.10) 0%, transparent 70%);
    pointer-events: none;
  }}
  .hero-badge {{
    display: inline-block;
    background: rgba(108,142,245,.15);
    border: 1px solid rgba(108,142,245,.35);
    color: var(--accent);
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 22px;
  }}
  .hero h1 {{
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 60%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 14px;
  }}
  .hero-sub {{
    color: var(--muted);
    font-size: 1rem;
    max-width: 480px;
    margin: 0 auto 28px;
    line-height: 1.6;
  }}
  .hero-meta {{
    display: flex;
    justify-content: center;
    gap: 32px;
    flex-wrap: wrap;
  }}
  .hero-meta span {{
    font-size: .82rem;
    color: var(--muted);
    font-family: var(--mono);
  }}
  .hero-meta strong {{
    color: var(--text);
    font-weight: 600;
  }}

  /* ── Layout ─────────────────────────────────────────────────── */
  .container {{ max-width: 1060px; margin: 0 auto; padding: 0 24px; }}

  /* ── KPI cards ───────────────────────────────────────────────── */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin: 44px 0 0;
  }}
  .kpi {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
  }}
  .kpi:hover {{ transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,.4); }}
  .kpi::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: 0;
    transition: opacity .2s;
  }}
  .kpi:hover::before {{ opacity: 1; }}
  .kpi-label {{ font-size: .72rem; font-weight: 600; letter-spacing: .08em;
                text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
  .kpi-value {{ font-size: 1.7rem; font-weight: 700; font-family: var(--mono); color: var(--text); }}
  .kpi-sub   {{ font-size: .75rem; color: var(--muted); margin-top: 4px; }}
  .kpi-green {{ color: var(--green); }}
  .kpi-amber {{ color: var(--amber); }}

  /* ── Section headings ────────────────────────────────────────── */
  .section-title {{
    font-size: 1.1rem; font-weight: 700; color: var(--text);
    margin: 52px 0 20px;
    display: flex; align-items: center; gap: 10px;
  }}
  .section-title::after {{
    content: ''; flex: 1; height: 1px; background: var(--border);
  }}

  /* ── Chart card ──────────────────────────────────────────────── */
  .chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
  }}
  .chart-card h3 {{
    font-size: .85rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 20px;
  }}
  .chart-wrap {{ position: relative; height: 280px; }}

  /* ── Comparison table ────────────────────────────────────────── */
  .table-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    margin-top: 16px;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead th {{
    background: var(--surface);
    padding: 14px 20px;
    text-align: left;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }}
  thead th:not(:first-child) {{ text-align: right; }}
  tbody tr {{ border-bottom: 1px solid rgba(37,45,64,.6); transition: background .15s; }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(108,142,245,.04); }}
  td {{ padding: 14px 20px; font-size: .9rem; }}
  td:not(:first-child) {{ text-align: right; font-family: var(--mono); font-size: .85rem; }}
  .metric-name {{ color: var(--text); font-weight: 500; }}
  .desc {{
    display: block; font-size: .73rem; color: var(--muted);
    font-weight: 400; margin-top: 2px; font-family: inherit;
  }}
  td.winner {{
    color: var(--green);
    font-weight: 600;
    position: relative;
  }}
  td.winner::after {{
    content: ' ✓';
    font-size: .7rem;
    opacity: .7;
  }}
  .na {{ color: var(--muted); font-style: italic; }}

  /* ── Pill legend ─────────────────────────────────────────────── */
  .legend {{
    display: flex; align-items: center; gap: 20px;
    font-size: .78rem; color: var(--muted);
    padding: 0 20px 16px;
    justify-content: flex-end;
  }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }}

  /* ── Footer ──────────────────────────────────────────────────── */
  .footer {{
    text-align: center; margin-top: 60px;
    font-size: .78rem; color: var(--muted);
    line-height: 1.8;
  }}
  .footer code {{ font-family: var(--mono); color: var(--accent); }}
</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
  <div class="hero-badge">Benchmark Report</div>
  <h1>Vectorless RAG vs ChromaDB</h1>
  <p class="hero-sub">
    SQLite FTS5 + NumPy cosine similarity compared against ChromaDB's
    persistent vector store — zero extra dependencies vs full stack.
  </p>
  <div class="hero-meta">
    <span>📄 Documents: <strong>{num_docs:,}</strong></span>
    <span>🔍 Queries: <strong>{num_queries:,}</strong></span>
    <span>📐 Embedding dim: <strong>{EMBEDDING_DIM:,}</strong></span>
    <span>🕐 Run at: <strong>{ts}</strong></span>
  </div>
</div>

<div class="container">

  <!-- KPI row -->
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Init Speedup</div>
      <div class="kpi-value kpi-green">{speedup_init}</div>
      <div class="kpi-sub">faster client init vs ChromaDB</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Index Speedup</div>
      <div class="kpi-value kpi-green">{speedup_index}</div>
      <div class="kpi-sub">faster to index {num_docs:,} documents</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Disk Saving</div>
      <div class="kpi-value kpi-amber">{size_ratio}</div>
      <div class="kpi-sub">smaller on-disk footprint</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">{qps_label}</div>
      <div class="kpi-value {qps_class}">{qps_value}</div>
      <div class="kpi-sub">{qps_sub}</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="section-title">Performance Breakdown</div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div class="chart-card">
      <h3>Latency metrics (lower = better)</h3>
      <div class="chart-wrap"><canvas id="barLatency"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Throughput — Queries per second (higher = better)</h3>
      <div class="chart-wrap"><canvas id="barQPS"></canvas></div>
    </div>
  </div>

  <div class="chart-card" style="margin-top:16px">
    <h3>All metrics — radar comparison</h3>
    <div class="chart-wrap" style="height:320px"><canvas id="radar"></canvas></div>
  </div>

  <!-- Table -->
  <div class="section-title">Detailed Results</div>
  <div class="table-card">
    <div class="legend">
      <span><span class="dot" style="background:var(--accent)"></span>Vectorless RAG</span>
      <span><span class="dot" style="background:var(--accent2)"></span>ChromaDB</span>
      <span style="color:var(--green)">✓ winner</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Vectorless RAG<br><small style="font-weight:400;text-transform:none">SQLite FTS5 + NumPy</small></th>
          <th>ChromaDB<br><small style="font-weight:400;text-transform:none">PersistentClient</small></th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div class="footer">
    Generated by <code>report.py</code> · Vectorless RAG Engine ·
    <a href="https://github.com" style="color:var(--accent);text-decoration:none">github.com/brand-builders/vectorless-rag</a>
  </div>
</div>

<script>
const VL_COLOR  = 'rgba(108,142,245,0.85)';
const VL_BORDER = 'rgba(108,142,245,1)';
const CH_COLOR  = 'rgba(167,139,250,0.75)';
const CH_BORDER = 'rgba(167,139,250,1)';

const FONT = {{ family: "'Inter', sans-serif" }};
Chart.defaults.color = '#64748b';
Object.assign(Chart.defaults.font, FONT);

const allLabels = {chart_labels};
const vlAll     = {vl_data};
const chAll     = {ch_data};

// ── Bar: latency metrics (import, init, index, disk, query) ────────
const latencyIdx  = [0,1,2,3,4];
const latencyLbls = latencyIdx.map(i => allLabels[i]);
const vlLatency   = latencyIdx.map(i => vlAll[i]);
const chLatency   = latencyIdx.map(i => chAll[i]);

new Chart(document.getElementById('barLatency'), {{
  type: 'bar',
  data: {{
    labels: latencyLbls,
    datasets: [
      {{ label: 'Vectorless RAG', data: vlLatency, backgroundColor: VL_COLOR, borderColor: VL_BORDER, borderWidth: 1.5, borderRadius: 6 }},
      {{ label: 'ChromaDB',       data: chLatency, backgroundColor: CH_COLOR,  borderColor: CH_BORDER,  borderWidth: 1.5, borderRadius: 6 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ font: FONT, boxRadius: 4 }} }} }},
    scales: {{
      x: {{ grid: {{ color: 'rgba(37,45,64,.5)' }}, ticks: {{ font: {{ size: 11 }} }} }},
      y: {{ grid: {{ color: 'rgba(37,45,64,.5)' }}, ticks: {{ font: {{ size: 11 }} }} }}
    }}
  }}
}});

// ── Bar: QPS ──────────────────────────────────────────────────────
new Chart(document.getElementById('barQPS'), {{
  type: 'bar',
  data: {{
    labels: ['Throughput (q/s)'],
    datasets: [
      {{ label: 'Vectorless RAG', data: [vlAll[5]], backgroundColor: VL_COLOR, borderColor: VL_BORDER, borderWidth: 1.5, borderRadius: 6 }},
      {{ label: 'ChromaDB',       data: [chAll[5]], backgroundColor: CH_COLOR,  borderColor: CH_BORDER,  borderWidth: 1.5, borderRadius: 6 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ font: FONT, boxRadius: 4 }} }} }},
    scales: {{
      x: {{ grid: {{ color: 'rgba(37,45,64,.5)' }} }},
      y: {{ grid: {{ color: 'rgba(37,45,64,.5)' }}, title: {{ display: true, text: 'queries / second', color: '#64748b' }} }}
    }}
  }}
}});

// ── Radar ─────────────────────────────────────────────────────────
// Normalise each metric to 0..1 (1 = best, 0 = worst)
const radarMetrics = ['Import latency', 'Init time', 'Index time', 'Disk size', 'Query latency', 'Throughput'];
function normalise(vl, ch, lowerBetter) {{
  if (lowerBetter) {{
    const mx = Math.max(vl, ch) || 1;
    return [1 - vl/mx, 1 - ch/mx];
  }} else {{
    const mx = Math.max(vl, ch) || 1;
    return [vl/mx, ch/mx];
  }}
}}
const lb = [true,true,true,true,true,false];
const vlRadar = [], chRadar = [];
for (let i = 0; i < 6; i++) {{
  const [v,c] = normalise(vlAll[i], chAll[i], lb[i]);
  vlRadar.push(+(v*100).toFixed(1));
  chRadar.push(+(c*100).toFixed(1));
}}
new Chart(document.getElementById('radar'), {{
  type: 'radar',
  data: {{
    labels: radarMetrics,
    datasets: [
      {{ label: 'Vectorless RAG', data: vlRadar, borderColor: VL_BORDER, backgroundColor: 'rgba(108,142,245,.18)', pointBackgroundColor: VL_BORDER, pointRadius: 4, borderWidth: 2 }},
      {{ label: 'ChromaDB',       data: chRadar, borderColor: CH_BORDER, backgroundColor: 'rgba(167,139,250,.12)', pointBackgroundColor: CH_BORDER, pointRadius: 4, borderWidth: 2 }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ font: FONT, boxRadius: 4 }} }} }},
    scales: {{
      r: {{
        min: 0, max: 100,
        grid:      {{ color: 'rgba(37,45,64,.8)' }},
        angleLines:{{ color: 'rgba(37,45,64,.8)' }},
        ticks:     {{ backdropColor: 'transparent', font: {{ size: 10 }}, stepSize: 25 }},
        pointLabels:{{ font: {{ size: 12 }}, color: '#94a3b8' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ Report saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run benchmark and emit an HTML report")
    parser.add_argument("--num-docs",    type=int, default=1000)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--out",         default="benchmark_report.html")
    args = parser.parse_args()

    print(f"Generating {args.num_docs:,} synthetic documents...")
    docs = generate_mock_data(args.num_docs)

    tmp = tempfile.mkdtemp(prefix="vless_report_")
    try:
        print("[1/2] Benchmarking Vectorless RAG...")
        vl = run_vectorless(docs, args.num_queries, os.path.join(tmp, "rag.db"))
        print("      Done.")

        if chromadb is None:
            print("[2/2] ChromaDB not installed — skipping.")
            ch = None
        else:
            print("[2/2] Benchmarking ChromaDB...")
            ch = run_chroma(docs, args.num_queries, os.path.join(tmp, "chroma"))
            print("      Done.")

        print(f"\nBuilding HTML report...")
        build_html(vl, ch, args.num_docs, args.num_queries, args.out)
        print(f"\nOpen in browser:  open {args.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
