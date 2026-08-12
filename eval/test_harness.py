#!/usr/bin/env python3
"""Self-check tests for the load-bearing eval/index logic (B1).

Run:  venv/bin/python eval/test_harness.py   (exit 0 = pass; asserts on failure)

Covers the standing gate + levers that had no tests:
  - build.build_purpose_chunk  (synthetic frontmatter chunk: marker, tag/trigger filtering, None-on-empty)
  - case_quality gate logic    (imperative detection, stale + graph-aware bucketing/exit)
  - symbols.extract            (TS/JS + shell symbol extraction — B6)

The retrieval recency lever is default-OFF and its decay is trivial inline math — not covered here.
case_quality.py also ships its own __main__ self-checks; this file is the single CI entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build  # noqa: E402  (module-level import is light — no model load until main())
import case_quality  # noqa: E402
import symbols  # noqa: E402


def test_purpose_chunk():
    mem = "---\nname: Foo\ndescription: a thing\ntags:\n  - project/x\n  - topic/y\n  - status/active\n---\nbody"
    pc = build.build_purpose_chunk("memory", Path("x/foo.md"), mem)
    assert pc is not None and pc[0] == 0 and pc[1] == 0, pc          # (0,0) synthetic marker
    body = pc[2]
    assert "Foo: a thing" in body, body                              # name + description
    assert "project/x" in body and "topic/y" in body, body           # project/topic kept
    assert "status/" not in body, body                               # status/* dropped (no signal)

    sk = "---\nname: bar\ndescription: does X\ntriggers:\n  - go\n  - run\n---\n"
    pc2 = build.build_purpose_chunk("skills", Path("bar/SKILL.md"), sk)
    assert pc2 and "bar: does X" in pc2[2] and "triggers: go, run" in pc2[2], pc2

    assert build.build_purpose_chunk("memory", Path("n.md"), "no frontmatter") is None   # no fm -> None
    assert build.build_purpose_chunk("memory", Path("n.md"), "---\ntags:\n  - type/x\n---") is not None  # tags-only ok


def test_gate_logic():
    # imperative: conservative — real command yes, legit query no
    assert case_quality.is_imperative_command("merge queue is set, run the watched test")
    assert not case_quality.is_imperative_command("add rate limiting to the auth endpoint")

    stale = case_quality.find_stale_labels(
        [{"query": "q", "expect_path_contains": "memory/GONE.md", "expect_scope": "memory"},
         {"query": "c", "expect_path_contains": "repoA/src/x.ts", "expect_scope": "code"}],
        {"memory/here.md", "repoB/a.ts"},
    )
    assert len(stale) == 2, stale                                    # both labels absent from index

    b = case_quality.bucket_stale(stale)                            # no code-repos passed -> code is coverage-gap
    assert len(b["fixable"]) == 1 and len(b["coverage_gap"]) == 1, b
    b2 = case_quality.bucket_stale(stale, code_indexed_repos={"repoA"})  # repoA code indexed -> stale code gates
    assert len(b2["fixable"]) == 2 and not b2["coverage_gap"], b2


def test_symbols():
    # Test TS/JS: defs, imports, and calls (excluding keywords)
    d, c, imp = symbols.extract(
        "import { foo } from './m';\nexport function f(){ g(); helper(); if(x){} }\nclass C {}\nexport const h = () => {};\ninterface I {}\n",
        "typescript",
    )
    assert {n for (n, l, s, e) in d} == {"f", "C", "h", "I"}, d
    assert ("./m", "foo") in imp, imp
    # Calls: f calls g and helper (not if, which is a keyword)
    callee_names = {ce for (ca, ce, ln) in c}
    assert "g" in callee_names and "helper" in callee_names, c
    assert "if" not in callee_names, f"if should not be a callee (keyword), got: {c}"

    # Test shell: defs and conservative calls (only intra-file function calls)
    d2, c2, _ = symbols.extract("foo() {\n  bar()\n}\nfunction bar() {\n  :\n}\n", "shell")
    assert {n for (n, l, s, e) in d2} == {"foo", "bar"}, d2
    # bar() call from foo should appear (bar is defined in file)
    assert any(ca == "foo" and ce == "bar" for (ca, ce, ln) in c2), c2

    # Test Python: unchanged
    d3, _, _ = symbols.extract("def py_fn():\n    pass\n", "python")
    assert any(n == "py_fn" for (n, l, s, e) in d3), d3
    assert symbols.extract("whatever", "markdown") == ([], [], [])  # unknown language -> empty


if __name__ == "__main__":
    test_purpose_chunk()
    test_gate_logic()
    test_symbols()
    # also run case_quality's own self-checks (imperative/stale/bucket/scope) so one command covers all
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "case_quality.py")], check=True)
    print("✓ All harness tests passed")
