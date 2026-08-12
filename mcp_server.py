#!/usr/bin/env python3
"""MCP stdio server exposing rag_query over the hybrid BM25+cosine index.

JSON-RPC 2.0. Supports: initialize, tools/list, tools/call (rag_query).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import retrieval
from retrieval import search


def _prewarm() -> None:
    """Load embedding model, reranker, and corpus caches in the background.

    Cold start is ~30-45s (torch import + HF weights + sqlite fetch + BM25
    build), which blows past MCP clients' per-request timeout on the first
    query. Warming at server start makes the first real query fast. Runs in a
    daemon thread so the JSON-RPC handshake is never blocked.
    """
    try:
        retrieval._get_model()
        retrieval._load(None, None)  # rag_query default scope
        retrieval._load(KNOWLEDGE_SCOPE, None)  # search_knowledge scope
        if os.environ.get("RAG_CODE_RERANK", "off").lower() in ("on", "1", "true"):
            retrieval._get_reranker()
        print("prewarm complete", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"prewarm failed (queries will cold-load instead): {e}", file=sys.stderr)

TOOL_SCHEMA = {
    "name": "rag_query",
    "description": (
        "Hybrid semantic + BM25 search over the user's configured corpus: notes, "
        "docs, repo docs + README + CHANGELOG, recent git commits, session "
        "transcripts, and source code (TS/JS/Python/Shell) from the configured "
        "repos. Returns top-K chunks with path:line + symbol + repo citations. "
        "Auto-scopes to the current repo when cwd is inside one — pass "
        "scope_repos=['all'] to disable. Use instead of grep for fuzzy or "
        "cross-file recall."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            "scope_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "source types: the labels from sources.yaml plus built-ins changelog, repo-docs, repo-readme, spec, roadmap, code, workstation-code, commit, session",
            },
            "scope_repos": {
                "type": "array",
                "items": {"type": "string"},
                "description": "configured repo names, or pass ['all'] to disable cwd auto-scoping",
            },
            "cwd": {"type": "string", "description": "working dir to drive auto-scoping (defaults to server cwd)"},
        },
        "required": ["query"],
    },
}

# Knowledge scope: which source_type labels count as durable knowledge for the
# search_knowledge tool. Override with RAG_KNOWLEDGE_SCOPE (csv of your labels).
KNOWLEDGE_SCOPE = [
    s.strip()
    for s in os.environ.get(
        "RAG_KNOWLEDGE_SCOPE", "memory,standards,plans,handoffs,adrs"
    ).split(",")
    if s.strip()
]

SEARCH_KNOWLEDGE_SCHEMA = {
    "name": "search_knowledge",
    "description": (
        "Semantic search over the knowledge layer — durable notes and decisions: "
        "memory notes, ADRs, plans, session handoffs, and standards (NOT source "
        "code or git commits). Cross-project by design (searches all repos, no "
        "cwd auto-scoping). Use for 'what did we decide / is there a note about "
        "X / did we hit this before'. For source-code or git-commit recall, use "
        "rag_query instead."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "top": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
    },
}


def _render(results: list[dict]) -> str:
    if not results:
        return "No matches."
    lines = []
    for r in results:
        tag = r["source_type"]
        if r.get("repo"):
            tag += f"/{r['repo']}"
        if r.get("symbol"):
            tag += f"::{r['symbol']}"
        lines.append(
            f"#{r['rank']}  rrf={r['rrf']} cos={r['cos']} bm={r['bm25']}  [{tag}]  {r['path']}:{r['start_line']}-{r['end_line']}"
        )
        snippet = r["text"][:500].replace("\n", " ")
        lines.append(f"  {snippet}")
    return "\n".join(lines)


def _clamp_top(raw, default=5, lo=1, hi=20):
    """Coerce the `top` arg to a sane int in [lo, hi]; never raises (bad input -> default).

    Guards the schema's declared bounds at runtime: a negative `top` would otherwise
    slice results as [:-n] (silently wrong count); a non-numeric string would raise.
    """
    try:
        t = int(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(t, hi))


def _respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def main() -> int:
    threading.Thread(target=_prewarm, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "initialize":
            _respond(
                msg_id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "shelfmark", "version": "0.1.0"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            _respond(msg_id, result={"tools": [TOOL_SCHEMA, SEARCH_KNOWLEDGE_SCHEMA]})
        elif method == "tools/call":
            tool = params.get("name")
            args = params.get("arguments") or {}
            try:
                if tool == "search_knowledge":
                    # Cross-project knowledge/decisions surface.
                    # scope_repos=["all"] -> no repo filter AND no cwd auto-scoping
                    # (scope_repos=None would auto-scope to the server's cwd repo).
                    # rerank=False: knowledge (non-code) scopes bypass the bge reranker by
                    # design — reranking memory/knowledge regressed -10.5pp (ADR-0011).
                    results = search(
                        query=args.get("query", ""),
                        top=_clamp_top(args.get("top")),
                        scope_types=KNOWLEDGE_SCOPE,
                        scope_repos=["all"],
                        cwd=None,
                        rerank=False,
                    )
                elif tool == "rag_query":
                    scope_types = args.get("scope_types") or None
                    scope_repos_raw = args.get("scope_repos") or None
                    if scope_repos_raw == ["all"]:
                        scope_repos = None
                        cwd = args.get("cwd")  # respect cwd hint but ignore repo match
                    elif scope_repos_raw:
                        scope_repos = scope_repos_raw
                        cwd = None
                    else:
                        scope_repos = None
                        cwd = args.get("cwd")
                    results = search(
                        query=args.get("query", ""),
                        top=_clamp_top(args.get("top")),
                        scope_types=scope_types,
                        scope_repos=scope_repos,
                        cwd=cwd,
                        rerank=bool(args.get("rerank", True)),
                    )
                else:
                    _respond(msg_id, error={"code": -32601, "message": f"unknown tool: {tool}"})
                    continue
                _respond(
                    msg_id,
                    result={
                        "content": [{"type": "text", "text": _render(results)}],
                        "isError": False,
                    },
                )
            except Exception as e:  # noqa: BLE001
                _respond(msg_id, error={"code": -32000, "message": str(e)})
        elif msg_id is not None:
            _respond(msg_id, error={"code": -32601, "message": f"unknown method: {method}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
