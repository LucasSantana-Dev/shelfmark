#!/usr/bin/env python3
"""Measure-before-hardcode: corpus-fit the guessed fusion constants (RRF_K, BM25_WEIGHT)
against the 140 golden cases. A/B one variable at a time vs the honest floor.
Monkeypatches retrieval module globals — no edit to retrieval.py. Read-only on the index."""
import json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["RAG_QLOG"] = "off"          # don't pollute queries.sqlite with eval queries
import retrieval

cases = [json.loads(l) for l in (ROOT/"eval"/"golden.jsonl").read_text().splitlines() if l.strip()]
cases = [c for c in cases if "expect_path_contains" in c]

def scope_key(scope):
    if isinstance(scope, str): return scope or "none"
    if isinstance(scope, list) and scope: return "+".join(scope)
    return "none"

def evaluate(top=5):
    hit = ctot = chit = 0
    for c in cases:
        q = c["query"]; expect = c["expect_path_contains"]
        expect = expect if isinstance(expect, list) else [expect]
        scope = c.get("expect_scope")
        res = retrieval.search(q, top=top, scope_types=scope, scope_repos=["all"], cwd=None, rerank=False)
        hp = any(any(e in r["path"] for e in expect) for r in res[:top])
        hit += hp
        if scope_key(scope) == "code":
            ctot += 1; chit += hp
    return hit/len(cases), (chit/ctot if ctot else 0.0)

def cfg(rrf_k, bm25, auto):
    retrieval.RRF_K = rrf_k
    retrieval.BM25_WEIGHT = bm25
    retrieval.RERANK_AUTO = auto
    o, c = evaluate()
    return o, c

# --- floors ---
base_o, base_c = cfg(60, 1.5, False)   # pure fusion (auto-rerank off)
dep_o,  dep_c  = cfg(60, 1.5, True)    # deployed (auto-rerank on) — should ≈ 0.693
print(f"FLOOR pure-fusion (RRF_K=60 BM25=1.5 auto=off):  overall={base_o:.3f} code={base_c:.3f}")
print(f"FLOOR deployed    (RRF_K=60 BM25=1.5 auto=ON):   overall={dep_o:.3f} code={dep_c:.3f}   <- the 0.693 gate\n")

def line(tag, o, c):
    do = (o-base_o)*100; dc = (c-base_c)*100
    star = "  <-- beats floor" if (o > base_o + 1e-9 or c > base_c + 1e-9) else ""
    print(f"  {tag:28s} overall={o:.3f} ({do:+.1f}pp) code={c:.3f} ({dc:+.1f}pp){star}")

print("RRF_K sweep (BM25=1.5, auto=off, A/B one var):")
for k in (20, 40, 60, 80, 100, 150):
    o, c = cfg(k, 1.5, False); line(f"RRF_K={k}", o, c)

print("\nBM25_WEIGHT sweep (RRF_K=60, auto=off):")
for w in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
    o, c = cfg(60, w, False); line(f"BM25_WEIGHT={w}", o, c)
