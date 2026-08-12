#!/usr/bin/env python3
"""eval/sweep.py — logged A/B harness wrapping eval/run.py.

Runs the matrix {dataset-current, holdout} for a labeled experiment, extracts
overall + per-scope Hit@5, compares to the stored baseline, applies the
KEEP/REVERT decision rule, and appends one row to eval/RESULTS.md (the ledger).

Every lever lives behind an env flag (retrieval.py) or a build flag (build.py),
so each experiment is exactly one command and fully reversible — nothing ships
until a row reads KEEP. Adding a future lever = add the flag + one sweep row.

Usage (the very first run MUST be the baseline — it stores eval/sweep-baseline.json):
  eval/sweep.py --label baseline
  RAG_PURPOSE_CHUNK=1 eval/sweep.py --label purpose-chunk --target memory
  RAG_RECENCY_WEIGHT=0.1 eval/sweep.py --label recency-w0.1 --target memory
  eval/sweep.py --label cwd-scope --target code

Decision rule (per experiment, vs baseline):
  KEEP if  memory Δ ≥ −0.5pp  AND  target-scope Δ ≥ +1pp     (else REVERT)
Hard guardrail: holdout-memory must also not regress > 0.5pp (memory is never
sacrificed — see retrieval.py:252-258, the −10.5pp bge / salvage lessons).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import run`
import run  # noqa: E402  (eval/run.py — reuse load()/run())
import case_quality  # noqa: E402  (eval/case_quality.py — reuse linting functions)

EVAL = ROOT / "eval"
DATASETS = {
    "current": EVAL / "dataset-current.jsonl",  # grown TRAIN set — the gate
    "holdout": EVAL / "holdout.jsonl",           # frozen — the honest number
}
RESULTS_MD = EVAL / "RESULTS.md"
BASELINE_JSON = EVAL / "sweep-baseline.json"

# Decision-rule thresholds.
MEM_GUARD = -0.005   # memory must not drop more than 0.5pp (current OR holdout)
TARGET_MIN = 0.01    # target scope must gain at least +1pp to KEEP

PRIMARY_GATE = 0.75  # memory-scope Hit@5 target on the grown set


def _h5(result: dict, scope: str | None = None) -> float | None:
    """Overall Hit@5 (scope=None) or a per-scope Hit@5 from a run() result."""
    if scope is None:
        return result.get("hit@5")
    sc = result.get("by_scope", {}).get(scope)
    return sc.get("hit@5") if sc else None


def _is_mem_target(case: dict) -> bool:
    """A case is 'memory-target' when expect_scope is memory OR the expected path is a
    memory note. (After the Iter-0 quarantine of mined session-fragment artifacts, these
    coincide: the curated memory cases carry expect_scope==memory.) This is the honest
    'quality memory' gate — protected by the guardrail and lifted by the purpose chunk."""
    if "memory" in (case.get("scope") or ""):
        return True
    es = case.get("expect")
    es = es if isinstance(es, list) else [es]
    return any(isinstance(e, str) and (("/memory/" in e) or e.startswith("memory/")
               or e.endswith("MEMORY.md")) for e in es)


def _mem_target(result: dict) -> tuple[float | None, int]:
    rows = [c for c in result.get("per_case", []) if _is_mem_target(c)]
    n = len(rows)
    if not n:
        return None, 0
    return sum(1 for c in rows if c["hit_rank"] and c["hit_rank"] <= 5) / n, n


def measure(top: int, rerank: bool) -> dict:
    """Run every dataset once; return a flat metrics snapshot for the ledger."""
    snap: dict = {"datasets": {}}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"WARN: dataset missing: {path}", file=sys.stderr)
            continue
        cases = run.load(path)
        t0 = time.time()
        res = run.run(cases, top=top, rerank=rerank)
        mt_h5, mt_n = _mem_target(res)
        snap["datasets"][name] = {
            "n": res["n"],
            "overall": _h5(res),
            "mrr": res.get("mrr"),
            "by_scope": {sc: m["hit@5"] for sc, m in res.get("by_scope", {}).items()},
            "memory_target": mt_h5,        # expected-path-is-a-memory-note (the gate)
            "memory_target_n": mt_n,
            "elapsed_s": round(time.time() - t0, 1),
        }
    cur = snap["datasets"].get("current", {})
    hol = snap["datasets"].get("holdout", {})
    snap["memory"] = cur.get("memory_target")                    # PRIMARY GATE (honest)
    snap["memory_n"] = cur.get("memory_target_n")
    snap["memory_scope"] = cur.get("by_scope", {}).get("memory")  # secondary (43 clean cases)
    snap["skills"] = cur.get("by_scope", {}).get("skills")
    snap["overall_train"] = cur.get("overall")
    snap["holdout"] = hol.get("overall")                          # STRETCH
    snap["holdout_memory"] = hol.get("memory_target")            # guardrail (memory never regress)
    snap["code"] = hol.get("by_scope", {}).get("code")           # holdout code (the 0.33 spot)
    snap["code_train"] = cur.get("by_scope", {}).get("code")
    return snap


def target_value(snap: dict, target: str) -> float | None:
    return {
        "memory": snap.get("memory"),
        "skills": snap.get("skills"),
        "code": snap.get("code"),
        "overall": snap.get("holdout"),  # "overall" stretch is the holdout number
    }.get(target)


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def delta(cur: float | None, base: float | None) -> str:
    if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
        return "—"
    d = (cur - base) * 100
    return f"{d:+.1f}pp"


def ensure_ledger() -> None:
    if RESULTS_MD.exists():
        return
    RESULTS_MD.write_text(
        "# RAG sweep ledger\n\n"
        "A/B results from `eval/sweep.py`. memory Hit@5 on `dataset-current` is the "
        f"PRIMARY GATE (target ≥ {PRIMARY_GATE:.2f}); holdout is the frozen honest number. "
        "KEEP rows ship; REVERT rows do not.\n\n"
        "Decision rule: KEEP if memory Δ ≥ −0.5pp AND target-scope Δ ≥ +1pp (and holdout "
        "memory not regressed > 0.5pp). See plan + retrieval.py:252-258.\n\n"
        "| label | target | mem h@5 | overall (train) | holdout h@5 | code h@5 | mem Δ | target Δ | verdict | when |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )


def decide(snap: dict, base: dict | None, target: str, is_baseline: bool) -> tuple[str, dict]:
    info = {"mem_delta": None, "target_delta": None, "holdout_mem_delta": None}
    if is_baseline or base is None:
        return ("BASELINE" if is_baseline else "NO-BASE"), info
    mem_d = (snap.get("memory") or 0) - (base.get("memory") or 0)
    hol_mem_d = (snap.get("holdout_memory") or 0) - (base.get("holdout_memory") or 0)
    tv, bv = target_value(snap, target), target_value(base, target)
    tgt_d = (tv - bv) if isinstance(tv, (int, float)) and isinstance(bv, (int, float)) else None
    info.update(mem_delta=mem_d, target_delta=tgt_d, holdout_mem_delta=hol_mem_d)
    guard_ok = mem_d >= MEM_GUARD and hol_mem_d >= MEM_GUARD
    gain_ok = isinstance(tgt_d, (int, float)) and tgt_d >= TARGET_MIN
    return ("KEEP" if (guard_ok and gain_ok) else "REVERT"), info


def lint_datasets(strict: bool = False) -> int:
    """Run case-quality linter on both datasets.

    Reports counts of stale labels and imperative-command cases.
    Exits with code 1 if any stale labels found (gate for CI/pre-commit), else 0.
    """
    # Load indexed paths + the set of repos that actually have indexed code (for B4:
    # distinguishing a genuinely-stale code label from a by-design graph-exclusion).
    conn = sqlite3.connect(str(ROOT / "index.sqlite"))
    try:
        indexed_paths = {row[0] for row in conn.execute("SELECT DISTINCT path FROM chunks")}
        code_repos = {r for (r,) in conn.execute(
            "SELECT DISTINCT repo FROM chunks WHERE source_type='code' AND repo IS NOT NULL")}
    except Exception as e:
        print(f"ERROR: Failed to read index.sqlite: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if not indexed_paths:
        print("ERROR: No indexed paths found in index.sqlite", file=sys.stderr)
        return 1

    # Run lint on both datasets
    all_stale = []
    imperative_counts = {}

    for name, path in DATASETS.items():
        if not path.exists():
            print(f"WARN: dataset missing: {path}", file=sys.stderr)
            continue

        cases = run.load(path)
        stale = case_quality.find_stale_labels(cases, indexed_paths)
        all_stale.extend([(name, s) for s in stale])

        imperative_count = sum(1 for c in cases if case_quality.is_imperative_command(c.get("query", "")))
        imperative_counts[name] = imperative_count

    # Bucket stale into fixable (real label rot → gate) vs coverage-gap (by-design unindexed → report).
    flat = [s for _, s in all_stale]
    buckets = case_quality.bucket_stale(flat, code_indexed_repos=code_repos)
    fixable, coverage = buckets["fixable"], buckets["coverage_gap"]
    fix_by_scope = collections.Counter(s.get("scope", "?") for s in fixable)

    # SIGNAL-FIRST output
    print("\n" + "=" * 72)
    print("EVAL CASE-QUALITY LINT")
    print("=" * 72)
    print("\nImperative-command cases (mined session-fragment noise):")
    for name in ["current", "holdout"]:
        if name in imperative_counts:
            print(f"  {name:20} {imperative_counts[name]:>3} cases")
    print(f"\nStale labels: {len(fixable)} FIXABLE (label rot — gates)  +  "
          f"{len(coverage)} coverage-gap (code, by-design unindexed — report only)")
    if fix_by_scope:
        print("  fixable by scope: " + ", ".join(f"{sc}={n}" for sc, n in fix_by_scope.most_common()))
    if fixable:
        print("\nFIXABLE stale (remap the label to the note's current path, or quarantine):")
        for s in fixable[:12]:
            print(f"  [{s.get('scope', ''):9}] {s['query'][:48]:48} → {s['expect']}")
        if len(fixable) > 12:
            print(f"  ... and {len(fixable) - 12} more")
    if coverage:
        # P2-4: surface (don't bury) code coverage-gaps. Most are by-design graph-excluded code,
        # but a genuinely-stale code label hides here too — we can't cheaply tell excluded-by-design
        # from real rot without graph state, so report for manual scan; do NOT gate.
        print("\ncoverage-gap stale (code; NOT gated — mostly by-design exclusions, scan for real rot):")
        for s in coverage[:8]:
            print(f"  [{s.get('scope', ''):9}] {s['query'][:48]:48} → {s['expect']}")
        if len(coverage) > 8:
            print(f"  ... and {len(coverage) - 8} more")

    print("=" * 72)
    gating = " (--strict → gate FAILS)" if (fixable and strict) else (" (advisory; add --strict to gate)" if fixable else "")
    print(f"VERDICT: {('FAIL — ' + str(len(fixable)) + ' fixable stale label(s)') if fixable else 'PASS — no fixable label rot'}{gating}")
    print("=" * 72)

    # Gate (exit 1) only under --strict, so pre-existing debt doesn't block adoption;
    # code coverage-gaps never gate (by-design unindexed).
    return 1 if (fixable and strict) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="experiment name (use 'baseline' first)")
    ap.add_argument("--lint", action="store_true", help="run case-quality linter instead of A/B sweep")
    ap.add_argument("--strict", action="store_true", help="with --lint: exit non-zero if fixable stale labels exist (CI/pre-commit gate)")
    ap.add_argument("--target", default="memory",
                    choices=["memory", "skills", "code", "overall"],
                    help="scope the decision rule judges (default: memory)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--rerank", action="store_true",
                    help="enable cross-encoder rerank (default off = the live --fast path)")
    ap.add_argument("--note", default="", help="free-text note appended to the row")
    args = ap.parse_args()

    if args.lint:
        return lint_datasets(strict=args.strict)

    if not args.label:
        ap.error("--label is required when not using --lint")

    is_baseline = args.label and args.label == "baseline"
    base = json.loads(BASELINE_JSON.read_text()) if BASELINE_JSON.exists() else None
    if base is None and not is_baseline:
        print("WARN: no eval/sweep-baseline.json — run `--label baseline` first. "
              "Recording row with no deltas.", file=sys.stderr)

    snap = measure(top=args.top, rerank=args.rerank)
    snap["label"] = args.label
    snap["target"] = args.target
    snap["when"] = time.strftime("%Y-%m-%d %H:%M")
    snap["flags"] = {k: v for k, v in os.environ.items() if k.startswith("RAG_")}

    verdict, info = decide(snap, base, args.target, is_baseline)
    snap["verdict"] = verdict
    snap["deltas"] = info

    # Persist the full snapshot; baseline run also writes the canonical baseline file.
    (EVAL / f"sweep-{args.label}.json").write_text(json.dumps(snap, indent=2))
    if is_baseline:
        BASELINE_JSON.write_text(json.dumps(snap, indent=2))

    ensure_ledger()
    note = f" {args.note}" if args.note else ""
    flagstr = " ".join(f"{k}={v}" for k, v in snap["flags"].items())
    row = (
        f"| {args.label} | {args.target} | {fmt(snap['memory'])} | {fmt(snap['overall_train'])} | "
        f"{fmt(snap['holdout'])} | {fmt(snap['code'])} | "
        f"{delta(snap['memory'], base.get('memory') if base else None)} | "
        f"{delta(target_value(snap, args.target), target_value(base, args.target) if base else None)} | "
        f"{verdict} | {snap['when']}{(' · ' + flagstr) if flagstr else ''}{note} |\n"
    )
    with RESULTS_MD.open("a") as fh:
        fh.write(row)

    # Verdict block to stdout.
    print("\n" + "=" * 72)
    print(f"SWEEP [{args.label}]  target={args.target}  flags=[{flagstr}]")
    print("-" * 72)
    print(f"  memory-target h@5 (GATE)   : {fmt(snap['memory'])}  n={snap.get('memory_n')}   "
          f"(Δ {delta(snap['memory'], base.get('memory') if base else None)})  "
          f"gate≥{PRIMARY_GATE:.2f} {'✓' if (snap['memory'] or 0) >= PRIMARY_GATE else '✗'}")
    print(f"  memory by_scope (2ndary)   : {fmt(snap['memory_scope'])}  (43 clean expect_scope==memory)")
    print(f"  skills h@5 (current)       : {fmt(snap['skills'])}")
    print(f"  overall h@5 (train)        : {fmt(snap['overall_train'])}")
    print(f"  holdout h@5 (stretch)      : {fmt(snap['holdout'])}   "
          f"(Δ {delta(snap['holdout'], base.get('holdout') if base else None)})")
    print(f"  holdout memory (guardrail) : {fmt(snap['holdout_memory'])}   "
          f"(Δ {delta(snap['holdout_memory'], base.get('holdout_memory') if base else None)})")
    print(f"  code h@5 (holdout)         : {fmt(snap['code'])}")
    if info["target_delta"] is not None:
        print("-" * 72)
        print(f"  RULE: mem Δ {info['mem_delta']*100:+.1f}pp (≥−0.5)  "
              f"holdout-mem Δ {info['holdout_mem_delta']*100:+.1f}pp (≥−0.5)  "
              f"target Δ {info['target_delta']*100:+.1f}pp (≥+1.0)")
    print("=" * 72)
    print(f"VERDICT: {verdict}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
