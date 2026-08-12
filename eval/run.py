#!/usr/bin/env python3
"""Evaluate RAG retrieval quality against a curated Q/A dataset.

Inputs:  eval/dataset.jsonl  (one JSON per line: query, expect_path_contains, expect_scope)
Outputs: MRR, Hit@1, Hit@3, Hit@5 — prints table + writes eval/<label>.json

Usage:
  eval/run.py                         # runs baseline, writes eval/baseline.json
  eval/run.py --label post-reranker   # save as named run
  eval/run.py --top 10 --label wider
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval/ dir, for sibling imports
from retrieval import search
from case_quality import case_scope  # single source of truth for expect_scope normalization

DATASET = ROOT / "eval" / "dataset.jsonl"

# Iter 4 levers (memory-first plan, code de-artifact) — EVAL-ONLY, behind flags; they do
# NOT touch live retrieval. They test BENCHMARK.md's two never-cleanly-measured claims:
#  - RAG_EVAL_CWD_SCOPE: scope code queries to the expected repo (mirrors the live `recall`
#    skill's cwd scoping) instead of ["all"] — tests whether cross-repo confusion is an
#    eval artifact of all-repos scoping (BENCHMARK.md:56).
#  - RAG_EVAL_EXCLUDE_SYMBOL_LOOKUP: drop "where is X defined / find Class Y" structural
#    lookups (code scope) from scoring — they belong to Serena, not chunk-RAG (BENCHMARK.md:55);
#    the live `recall` skill already routes them. Delegation, not gaming.
CWD_SCOPE = os.environ.get("RAG_EVAL_CWD_SCOPE", "0").lower() in ("1", "on", "true")
EXCLUDE_SYMBOL_LOOKUP = os.environ.get("RAG_EVAL_EXCLUDE_SYMBOL_LOOKUP", "0").lower() in ("1", "on", "true")

# Longest name first so a more specific repo name (my-repo-plugin) matches
# before its prefix (my-repo).
from config import CURATED_REPOS as _REPOS  # noqa: E402

_CURATED_REPO_NAMES = sorted((r.name for r in _REPOS), key=len, reverse=True)
_SYMBOL_LOOKUP_RE = re.compile(
    r"\b(where(\s+is|'s|\s+are)|locate|definition of|defined|declared|implementation of|"
    r"which file|what file|find (the )?(class|function|method|interface|type|symbol|def))\b",
    re.I,
)


def _infer_repo(paths: list[str]) -> str | None:
    for p in paths:
        if not isinstance(p, str):
            continue
        for r in _CURATED_REPO_NAMES:
            if f"/{r}/" in p or p.startswith(f"{r}/"):
                return r
    return None


def is_symbol_lookup(query: str) -> bool:
    return bool(_SYMBOL_LOOKUP_RE.search(query))


def load(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(cases: list[dict], top: int, rerank: bool = False) -> dict:
    per_case = []
    excluded_symbol_lookup = 0
    for case in cases:
        # Tolerate the legacy diff-scoped schema (id/type/expected_files) by skipping it.
        if "expect_path_contains" not in case:
            continue
        q = case["query"]
        expect = case["expect_path_contains"]
        scope = case.get("expect_scope")
        scope_key = case_scope(case)  # centralized (was inline; see case_quality.case_scope)
        expected = expect if isinstance(expect, list) else [expect]
        # Delegate code symbol-lookups to Serena — exclude from chunk-RAG scoring.
        if EXCLUDE_SYMBOL_LOOKUP and "code" in scope_key and is_symbol_lookup(q):
            excluded_symbol_lookup += 1
            continue
        # cwd-scope code queries to their expected repo instead of all-repos.
        scope_repos = ["all"]
        if CWD_SCOPE and "code" in scope_key:
            repo = _infer_repo(expected)
            if repo:
                scope_repos = [repo]
        results = search(q, top=top, scope_types=scope, scope_repos=scope_repos, cwd=None, rerank=rerank)
        hit_rank = None
        for r in results:
            if any(e in r["path"] for e in expected):
                hit_rank = r["rank"]
                break
        per_case.append(
            {
                "query": q,
                "expect": expect,
                "scope": scope_key,
                "hit_rank": hit_rank,
                "top_hit": f"{results[0]['path']}:{results[0]['start_line']}" if results else None,
            }
        )

    def metrics(cases: list[dict]) -> dict:
        n = len(cases) or 1
        hits_at = lambda k: sum(1 for c in cases if c["hit_rank"] and c["hit_rank"] <= k) / n
        mrr = sum((1.0 / c["hit_rank"]) if c["hit_rank"] else 0.0 for c in cases) / n
        return {
            "n": len(cases),
            "mrr": round(mrr, 3),
            "hit@1": round(hits_at(1), 3),
            "hit@3": round(hits_at(3), 3),
            "hit@5": round(hits_at(5), 3),
        }

    by_scope: dict[str, list] = {}
    for c in per_case:
        by_scope.setdefault(c["scope"], []).append(c)

    out = metrics(per_case)
    out["by_scope"] = {sc: metrics(cs) for sc, cs in sorted(by_scope.items())}
    out["per_case"] = per_case
    out["excluded_symbol_lookup"] = excluded_symbol_lookup
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--dataset", default=str(DATASET), help="path to eval dataset jsonl")
    ap.add_argument("--rerank", action="store_true", help="enable cross-encoder reranking")
    ap.add_argument("--auto", action="store_true", help="rerank=None: per-scope auto policy (reranks code+standards, never memory)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = load(Path(args.dataset))
    started = time.time()
    rerank_arg = None if args.auto else args.rerank
    result = run(cases, top=args.top, rerank=rerank_arg)
    elapsed = time.time() - started

    rerank_tag = " [AUTO]" if args.auto else (" [RERANK]" if args.rerank else " [FAST]")
    print(
        f"[{args.label}]{rerank_tag}  n={result['n']}  MRR={result['mrr']}  "
        f"hit@1={result['hit@1']}  hit@3={result['hit@3']}  hit@5={result['hit@5']}  "
        f"({elapsed:.1f}s)"
    )
    for sc, m in result.get("by_scope", {}).items():
        print(f"    scope={sc:10} n={m['n']:>3}  hit@1={m['hit@1']}  hit@5={m['hit@5']}  mrr={m['mrr']}")
    if args.verbose:
        for c in result["per_case"]:
            status = "✓" if c["hit_rank"] else "✗"
            rank = f"#{c['hit_rank']}" if c["hit_rank"] else "MISS"
            print(f"  {status} {rank:>4}  {c['query'][:60]}  → {c['top_hit']}")
    out_path = ROOT / "eval" / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
