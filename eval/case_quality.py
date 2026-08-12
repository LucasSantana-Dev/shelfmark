#!/usr/bin/env python3
"""Pure functions for eval case quality linting.

Detects two recurring classes of noise in eval datasets:
  1. Stale expected paths — occur when notes are migrated/renamed after mining
  2. Imperative-command queries — session fragments mined as "queries" that aren't information-seeking

These functions are reusable by both the linter and any future eval-case miner.
"""
import re

# Operational-command verbs. run/merge/commit/push/deploy are EXCLUDED from a bare start
# because they double as nouns/modifiers in legit queries ("merge conflict", "commit
# message format", "deploy script") — but they DO count inside a step clause (", run X").
# Kept deliberately small: conservative, favors false-negatives.
_OP_VERBS = r"run|rerun|re-run|merge|commit|push|pull|deploy|redeploy|revert|rebase|dispatch\w*|redo|ship|rollback|start|stop"
# Verbs unambiguous enough to flag at the very start (rarely query-opening nouns).
_UNAMBIG_START = re.compile(r"^(please\s+|let'?s\s+)?(rerun|re-run|redo|redeploy|rollback|revert|rebase|cherry-pick|dispatch\w*)\b", re.I)
_CMD_STEP = re.compile(rf",\s*(then\s+)?({_OP_VERBS})\b", re.I)        # "X is set, run Y"
_CMD_PHRASE = re.compile(r"\b(do (these|this|that) for me|for me using|for me,)\b", re.I)
_QUESTION_START = ("who ", "what ", "where ", "when ", "why ", "how ", "which ")


def is_imperative_command(query: str) -> bool:
    """Detect mined session-command fragments masquerading as eval queries.

    CONSERVATIVE by construction — favors false-negatives. Flags only UNAMBIGUOUS commands:
      - a session-continuation start ("then ...")
      - an unambiguous-verb start (rerun/redo/revert/rebase/dispatch/...)
      - an agent-directed phrase ("do these for me", "for me using")
      - a step-list clause (", run X" / ", dispatch Y")
    Common query-opening verbs (add/update/set/check/create/fix/use) and noun-ambiguous
    op-verbs at the start (merge/run/commit/push/deploy — cf. "merge conflict", "commit
    message") are deliberately NOT treated as commands, so legit information-seeking queries
    pass. Never flags questions (leading who/what/where/when/why/how/which, a "?", or
    "find the"/"definition of").

    This is an advisory lint signal AND the pre-add checklist for hand-curated golden cases,
    so a false positive (rejecting a legit query) is worse than a false negative (missing
    some chatter) — hence the conservative bias.
    """
    q = (query or "").strip()
    if not q:
        return False
    ql = q.lower()
    if ql.startswith(_QUESTION_START):
        return False
    if "?" in q or ql.startswith(("find the ", "definition of ")):
        return False
    if ql.startswith("then "):
        return True
    return bool(_UNAMBIG_START.search(q) or _CMD_PHRASE.search(q) or _CMD_STEP.search(q))


# Scopes whose corpus is NOT fully indexed by design (graphed repos' raw code is excluded —
# ADR-0009/0011). A "stale" label in these scopes is usually a coverage gap (intended
# exclusion), not fixable label rot — the linter REPORTS but does NOT gate on them. Every
# other scope IS fully indexed, so a stale label there is real, fixable rot → gate.
# Trade-off (P2-4): a genuinely-stale code label (e.g. a renamed file in a NON-graphed curated
# repo) also lands here and won't gate — distinguishing it from by-design exclusions needs
# graph state we don't have at lint time. Mitigation: sweep.py --lint LISTS these for manual
# scan rather than only counting them, so real code rot is visible even though it doesn't gate.
COVERAGE_GAP_SCOPES = {"code"}


def case_scope(case: dict) -> str:
    """Normalize a case's expect_scope to a string key (list scopes joined by '+')."""
    s = case.get("expect_scope")
    if isinstance(s, str) and s:
        return s
    if isinstance(s, list) and s:
        return "+".join(s)
    return "none"


def find_stale_labels(cases: list[dict], index_paths: set[str]) -> list[dict]:
    """Find eval cases whose expected paths are not in the index.

    A case is STALE if NONE of its expected path substrings appears in ANY
    indexed path. This indicates the note was likely migrated or deleted after
    the case was mined.

    Args:
        cases: List of eval cases (each with "query" and "expect_path_contains")
        index_paths: Set of all indexed paths from the chunks table

    Returns:
        List of stale cases: [{"query": ..., "expect": ..., "reason": ...}, ...]
    """
    stale = []
    for case in cases:
        if "expect_path_contains" not in case:
            continue

        expect = case["expect_path_contains"]
        # Normalize to list
        expected = expect if isinstance(expect, list) else [expect]

        # A case is stale if NONE of its expected substrings is in ANY indexed path
        is_stale = True
        for expected_substr in expected:
            for indexed_path in index_paths:
                if expected_substr in indexed_path:
                    is_stale = False
                    break
            if not is_stale:
                break

        if is_stale:
            stale.append({
                "query": case.get("query", ""),
                "expect": expect,
                "scope": case_scope(case),
                "reason": "no indexed path contains expected substring",
            })

    return stale


def bucket_stale(stale: list[dict], code_indexed_repos: set | None = None) -> dict:
    """Split stale labels into 'fixable' (real label rot → gate) vs 'coverage_gap'
    (scope not fully indexed by design → report only).

    B4 (graph-aware): a code-scope stale label is normally a coverage gap (graphed repos'
    raw code is excluded). BUT if the label names a repo that DOES have indexed code, the
    file genuinely exists in the corpus, so a missing label is real rot → gate it. Symbol-
    like labels with no repo name stay coverage_gap (can't tell). `code_indexed_repos` =
    repos that have indexed code; None → all code is coverage_gap (back-compatible)."""
    fixable, coverage = [], []
    for s in stale:
        if s.get("scope") not in COVERAGE_GAP_SCOPES:
            fixable.append(s)
            continue
        exp = s.get("expect")
        exps = exp if isinstance(exp, list) else [exp]
        names_indexed_repo = bool(code_indexed_repos) and any(
            isinstance(e, str) and r and r in e
            for e in exps for r in code_indexed_repos
        )
        (fixable if names_indexed_repo else coverage).append(s)
    return {"fixable": fixable, "coverage_gap": coverage}


def relevance_ok(expected: list[str], top_paths: list[str]) -> bool:
    """Check if expected substring(s) appear in top retrieved paths.

    Utility for future eval-case miners: only keep a mined case if its label
    is actually retrievable in the top results.

    Args:
        expected: List of expected path substrings
        top_paths: List of top retrieved paths

    Returns:
        True if any expected substring appears in any top path
    """
    for exp_substr in expected:
        for top_path in top_paths:
            if exp_substr in top_path:
                return True
    return False


if __name__ == "__main__":
    # Self-check: imperative-command detection — CONSERVATIVE (favors false-negatives).
    # True only for unambiguous commands:
    assert is_imperative_command("do these for me using the computer use")       # agent phrase
    assert is_imperative_command("merge queue is set, run the watched test")     # step clause
    assert is_imperative_command("then start #1081 on a worktree")               # continuation
    assert is_imperative_command("rerun the failed CI job")                      # unambiguous verb
    assert is_imperative_command("Sim, dispatcha F15 e F16")                     # step clause (non-EN)
    # False for legit info-seeking queries (the regressions P2-1 was about):
    assert not is_imperative_command("how do I run the tests?")                  # question
    assert not is_imperative_command("merge conflict resolution strategy")       # verb-as-noun
    assert not is_imperative_command("add rate limiting to the auth endpoint")   # legit feature query
    assert not is_imperative_command("configure caching using the redis adapter")  # legit
    assert not is_imperative_command("update the user profile schema migration") # legit
    assert not is_imperative_command("find the test runner in this repo")        # find-the
    assert not is_imperative_command("definition of a work queue")               # definition-of

    # Self-check: gate decision — bucket_stale (code = coverage-gap/no-gate; others = fixable/gate)
    _b = bucket_stale([
        {"query": "a", "expect": "x", "scope": "memory"},
        {"query": "b", "expect": "y", "scope": "code"},
        {"query": "c", "expect": "z", "scope": "none"},
    ])
    assert len(_b["coverage_gap"]) == 1 and _b["coverage_gap"][0]["scope"] == "code", _b
    assert {s["scope"] for s in _b["fixable"]} == {"memory", "none"}, _b

    # Self-check: case_scope normalization (str / list / missing)
    assert case_scope({"expect_scope": "memory"}) == "memory"
    assert case_scope({"expect_scope": ["session", "memory"]}) == "session+memory"
    assert case_scope({}) == "none"

    # Self-check: stale-label detection
    test_cases = [
        {"query": "example query", "expect_path_contains": "memory/valid-note.md"},
        {"query": "another query", "expect_path_contains": "memory/DELETED-NOTE.md"},
        {"query": "multi-target", "expect_path_contains": ["memory/valid-note.md", "memory/DELETED.md"]},
    ]
    test_paths = {
        "memory/valid-note.md",
        "memory/other.md",
        "code/src/main.py",
    }
    stale = find_stale_labels(test_cases, test_paths)
    assert len(stale) == 1, f"Expected 1 stale case, got {len(stale)}"
    assert stale[0]["query"] == "another query", f"Wrong stale case: {stale[0]}"

    # Self-check: relevance_ok
    assert relevance_ok(["memory/note"], ["memory/note.md", "other.md"])
    assert relevance_ok(["note"], ["memory/note.md"])
    assert not relevance_ok(["nonexistent"], ["memory/note.md"])

    print("✓ All case_quality self-checks passed")
