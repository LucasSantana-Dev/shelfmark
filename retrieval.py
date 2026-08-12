"""Shared retrieval logic: hybrid BM25 + cosine with Reciprocal Rank Fusion.

Loaded once per process (sqlite read + tokenize) and cached by scope signature.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from config import DB, DIM, MODEL_NAME, QLOG, ROOT
from config import CURATED_REPOS as REPO_ROOTS

RRF_K = 60
BM25_WEIGHT = float(os.environ.get("RAG_BM25_WEIGHT", "1.5"))  # >1 favors lexical (code) match
# Iter 2 lever (memory-first plan): gentle recency prior for memory-scope chunks only.
# Added at RRF-fusion time as a tie-break (NOT a rerank — memory is never reranked), scaled
# to one rank-step so fresh memory wins close calls without dominating relevance. Reference =
# max mtime among memory candidates (no wall-clock call -> eval-deterministic). Default 0 (off);
# the memory-first sweep tried {0.05, 0.10, 0.20}. Improves live autorecall freshness.
RAG_RECENCY_WEIGHT = float(os.environ.get("RAG_RECENCY_WEIGHT", "0"))
RECENCY_HALF_SECONDS = 90 * 86400.0  # linear decay window: notes 90d older than the freshest fade to 0

HOME = Path.home()

_TOKEN_RE = re.compile(r"[A-Za-z_][\w$]{1,}")
_SUB_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[a-z]+|[A-Z]+|\d+")
_SYM_STOP = {"where","what","when","does","with","from","this","that","which","used","call","calls","file","files","function","defined","definition","class","interface","type","setup","how","are","the","and","for","into","across","handler","implementation","usage","schema"}
_model = None
_reranker = None
# Serializes lazy loader paths (_get_model/_get_reranker/_load cache-miss) so a
# query arriving while the MCP server's prewarm thread is still loading blocks
# instead of double-loading multi-GB models (measured: 80s query + swap storm).
_loader_lock = threading.Lock()
_cache: dict[tuple, tuple[list[dict], np.ndarray, BM25Okapi]] = {}
# (mtime, size) of DB at last cache fill. Any write to index.sqlite (incremental
# reindex, full rebuild, export swap) invalidates every cached scope on the next
# _load: without this the long-lived MCP server serves a pre-write snapshot
# forever (found 2026-07-29: memory notes invisible to rag_query until restart).
_cache_db_stamp: tuple[float, int] | None = None


def _db_stamp() -> tuple[float, int]:
    try:
        st = os.stat(DB)
        return (st.st_mtime, st.st_size)
    except OSError:
        return (0.0, 0)
# Reranker model. Override with RAG_RERANK_MODEL env var.
# Default kept on ms-marco-MiniLM-L-6-v2 (lightweight, fast, no download
# needed — already cached). Swap to "BAAI/bge-reranker-v2-m3" for +5-10pp
# Hit@5 once K3 LongMemEval baseline confirms the gain on this corpus.
RERANK_MODEL_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", RERANK_MODEL_DEFAULT)

# Auto-rerank configuration for weak/ambiguous queries.
RERANK_AUTO = os.environ.get("RAG_RERANK_AUTO", "on").lower() in ("on", "1", "true")
RERANK_AUTO_THRESHOLD = float(os.environ.get("RAG_RERANK_AUTO_THRESHOLD", "0.35"))
RERANK_AUTO_MARGIN = float(os.environ.get("RAG_RERANK_AUTO_MARGIN", "0.08"))
# Selective reranking for code-scope queries (the measured weak spot). Code retrieval is
# lexical-dominant and the fused ranking often buries the right chunk at rank 6-20, where a
# strong cross-encoder recovers it. Validated 2026-06-15 (ADR 0011): selective
# bge-reranker-v2-m3 on code scope = +4.9pp code / +2.1pp overall vs the 0.693/0.557 floor;
# applying it to ALL scopes regressed memory/standards (net -1). Default OFF — the model is
# ~2.2GB and machine-local; enable (with RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3 + the model
# cached) only where present. Reranker failures fall back to the fused ranking (graceful).
RAG_CODE_RERANK = os.environ.get("RAG_CODE_RERANK", "off").lower() in ("on", "1", "true")


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _loader_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _get_model():
    global _model
    if _model is None:
        with _loader_lock:
            if _model is None:
                _model = _build_model()
    return _model


def _build_model():
    from sentence_transformers import SentenceTransformer, models

    try:
        return SentenceTransformer(MODEL_NAME)
    except TypeError as exc:
        if "embedding_dimension" not in str(exc):
            raise
        model_path = MODEL_NAME
        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(MODEL_NAME, local_files_only=True)
        except Exception:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
            # Convert MODEL_NAME (intfloat/multilingual-e5-small) to cache path format
            cache_dir_name = "models--" + MODEL_NAME.replace("/", "--")
            snapshots = (
                cache_root
                / cache_dir_name
                / "snapshots"
            )
            candidates = [p for p in snapshots.glob("*") if (p / "config.json").exists()]
            if candidates:
                model_path = str(candidates[0])
        transformer = models.Transformer(model_path)
        pooling = models.Pooling(DIM, pooling_mode="mean")
        normalize = models.Normalize()
        return SentenceTransformer(modules=[transformer, pooling, normalize])


def _tokenize(text: str) -> list[str]:
    # Code-aware: whole identifier (lowercased) + camelCase/snake_case/dotted sub-tokens,
    # so natural-language queries ("create player") match symbol names ("createPlayer").
    # Applied to both corpus and query sides. Validated +2.8pp code / +3.2pp overall, 0 regressions.
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        out.append(tok.lower())
        subs: list[str] = []
        for piece in re.split(r"[_$]+", tok):
            subs.extend(_SUB_RE.findall(piece))
        subs = [s.lower() for s in subs if len(s) >= 2]
        if len(subs) > 1:
            out.extend(subs)
    return out


def _safe_cwd() -> str:
    """os.getcwd() raises FileNotFoundError if the server's cwd was deleted
    (e.g. launched from a later-pruned worktree); every tools/call then dies
    with '[Errno 2]' while the server still reports Connected. Fall back to
    HOME: matches no REPO_ROOT, so auto-scoping just disables."""
    try:
        return os.getcwd()
    except OSError:
        return str(HOME)


def cwd_repo(cwd: str | None = None) -> str | None:
    """Detect which curated repo the cwd lives in, returning the repo.name."""
    path = Path(cwd or _safe_cwd()).resolve()
    for repo in REPO_ROOTS:
        try:
            path.relative_to(repo)
            return repo.name
        except ValueError:
            continue
    return None


def _load(scope_types: list[str] | None, scope_repos: list[str] | None) -> tuple:
    global _cache_db_stamp
    stamp = _db_stamp()
    if _cache_db_stamp is None:
        with _loader_lock:
            if _cache_db_stamp is None:
                _cache_db_stamp = stamp
    elif stamp != _cache_db_stamp:
        with _loader_lock:
            if stamp != _cache_db_stamp:  # re-check: another thread may have invalidated
                _cache.clear()
                _cache_db_stamp = stamp
                print("rag-index: index.sqlite changed; corpus cache invalidated",
                      file=sys.stderr)
    key = (
        tuple(sorted(scope_types)) if scope_types else None,
        tuple(sorted(scope_repos)) if scope_repos else None,
    )
    if key in _cache:
        return _cache[key]
    with _loader_lock:
        if key in _cache:  # re-check: prewarm thread may have filled it while we waited
            return _cache[key]
        return _load_uncached(key, scope_types, scope_repos)


def _load_uncached(key: tuple, scope_types: list[str] | None, scope_repos: list[str] | None) -> tuple:
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")  # WAL+timeout hardening (WAL set by writer)
    where: list[str] = []
    params: list[Any] = []
    if scope_types:
        where.append(f"source_type IN ({','.join('?' * len(scope_types))})")
        params.extend(scope_types)
    if scope_repos:
        where.append(f"repo IN ({','.join('?' * len(scope_repos))})")
        params.extend(scope_repos)
    sql = (
        "SELECT id, source_type, repo, language, symbol, path, start_line, end_line, text, embedding, mtime "
        "FROM chunks"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    if not rows:
        empty_meta: list[dict] = []
        empty_embs = np.zeros((0, DIM), dtype=np.float32)
        bm25 = BM25Okapi([[""]])
        _cache[key] = (empty_meta, empty_embs, bm25)
        return _cache[key]
    embs = np.frombuffer(b"".join(r[9] for r in rows), dtype=np.float32).reshape(-1, DIM)
    meta = [
        {
            "id": r[0],
            "source_type": r[1],
            "repo": r[2],
            "language": r[3],
            "symbol": r[4],
            "path": r[5],
            "start_line": r[6],
            "end_line": r[7],
            "text": r[8],
            "mtime": r[10],  # r[9] = embedding blob; mtime appended last in SELECT
        }
        for r in rows
    ]
    tokens = [_tokenize(f"{m['symbol']} {m['text']}") for m in meta]
    bm25 = BM25Okapi(tokens)
    _cache[key] = (meta, embs, bm25)
    return _cache[key]


def search(
    query: str,
    top: int = 5,
    scope_types: list[str] | None = None,
    scope_repos: list[str] | None = None,
    cwd: str | None = None,
    rerank: bool | None = None,
) -> list[dict]:
    # transformers imports lazily below and does Path("src").resolve() at import
    # time, which raises ENOENT when the process cwd was deleted (pruned worktree).
    # Recover once here so every downstream relative-path use is safe.
    try:
        os.getcwd()
    except OSError:
        os.chdir(HOME)
    if not query.strip():
        return []
    if isinstance(scope_types, str):  # defensive: a bare string would char-explode in _load's IN(...)
        scope_types = [scope_types] if scope_types else None
    if scope_repos == ["all"]:
        scope_repos = None
    elif scope_repos is None:
        detected = cwd_repo(cwd)
        if detected:
            scope_repos = [detected]
    meta, embs, bm25 = _load(scope_types, scope_repos)
    if not meta:
        return []

    # Cosine
    # E5 model requires "query: " prefix for queries
    prefixed_query = f"query: {query}"
    qv = _get_model().encode(
        [prefixed_query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)[0]
    cos = embs @ qv
    cos_order = np.argsort(-cos)

    # BM25
    q_tokens = _tokenize(query)
    bm_scores = bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(meta))
    bm_order = np.argsort(-bm_scores)

    # Retrieval: hybrid BM25+cosine via Reciprocal Rank Fusion OR pure cosine
    # Set RAG_HYBRID=1 to use hybrid (RRF); default is cosine-only
    use_hybrid = os.environ.get("RAG_HYBRID", "1").lower() in ("1", "on", "true")

    if use_hybrid:
        # Reciprocal Rank Fusion — take top (top*8, min 40) from each to bound work.
        fusion_window = min(len(meta), max(top * 16, 80))
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(cos_order[:fusion_window]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm_order[:fusion_window]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + BM25_WEIGHT / (RRF_K + rank + 1)
        # Symbol-definition boost: chunks whose defined symbol matches a query identifier get a
        # rank-0 signal — targets "where is X defined / X interface" queries (chunks.symbol is
        # populated; the symbols_* call-graph tables are not). Validated +2.8pp code, 0 regressions.
        q_idents = {t.lower() for t in _TOKEN_RE.findall(query) if len(t) >= 4 and t.lower() not in _SYM_STOP}
        if q_idents:
            for i, m in enumerate(meta):
                sym = (m.get("symbol") or "").lower()
                if sym and sym in q_idents:
                    rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (RRF_K + 1)
        # Recency prior — memory-scope chunks only (fresh memory wins ties). Deterministic:
        # reference = max mtime among memory candidates, linear decay over RECENCY_HALF_SECONDS.
        # Scaled by 1/(RRF_K+1) so the max boost equals one rank-step (gentle tie-break, not a
        # relevance override). Memory is never reranked downstream — this is the only freshness lever.
        if RAG_RECENCY_WEIGHT > 0:
            mem = [(i, m["mtime"]) for i, m in enumerate(meta)
                   if m.get("source_type") == "memory" and m.get("mtime")]
            if mem:
                ref = max(mt for _, mt in mem)
                for i, mt in mem:
                    decay = 1.0 - (ref - mt) / RECENCY_HALF_SECONDS
                    if decay > 0:
                        rrf[int(i)] = rrf.get(int(i), 0.0) + RAG_RECENCY_WEIGHT * decay / (RRF_K + 1)
        cos_scores_for_ranking = rrf
    else:
        # Cosine-only: rank by cosine similarity
        cos_scores_for_ranking = {int(idx): float(cos[idx]) for idx in range(len(cos))}

    if rerank is None:
        rerank = os.environ.get("RAG_RERANK", "off").lower() in ("on", "1", "true")
        # Rerank-eligible scopes = code (ADR-0011) + standards (2026-06-21 benchmark: rerank lifts
        # standards +14pp, hit@5 0.43->0.57). Gate on EVERY scoped type being rerank-friendly, so a
        # MIXED query that also includes memory never reranks: reranking memory regressed -10.5pp
        # (memory 0.921 -> 0.816 on the rerank=None path, ADR-0011). Hence search_knowledge
        # (memory+standards+plans+handoffs+adrs) stays un-reranked & protected, while a
        # standards-only or code-only rag_query reranks.
        is_rerank_scope = bool(scope_types) and all(
            any(rs in s for rs in ("code", "standards")) for s in scope_types
        )
        # Auto-trigger rerank on weak/ambiguous queries (if not explicitly disabled). In default
        # (ms-marco) mode, auto-rerank fires on any scope (net-positive there).
        auto_allowed = RERANK_AUTO and (not RAG_CODE_RERANK or is_rerank_scope)
        if not rerank and auto_allowed:
            top1 = float(cos[cos_order[0]]) if len(cos_order) > 0 else 0.0
            top2 = float(cos[cos_order[1]]) if len(cos_order) > 1 else 0.0
            if top1 < RERANK_AUTO_THRESHOLD or (top1 - top2) < RERANK_AUTO_MARGIN:
                rerank = True
        # Selective rerank (ADR 0011 + 2026-06-21 benchmark): the fused ranking is weakest for
        # code and standards; rerank them when scope is restricted to those (memory excluded).
        if not rerank and RAG_CODE_RERANK and is_rerank_scope:
            rerank = True

    if rerank:
        candidate_k = min(len(meta), max(top * 4, 20))
        candidate_order = sorted(cos_scores_for_ranking.items(), key=lambda kv: -kv[1])[:candidate_k]
        pairs = [(query, meta[idx]["text"][:1500]) for idx, _ in candidate_order]
        try:
            ce_scores = _get_reranker().predict(pairs, show_progress_bar=False)
            reranked = sorted(
                zip(candidate_order, ce_scores), key=lambda x: -float(x[1])
            )[:top]
            fused = [(idx_score[0][0], float(idx_score[1])) for idx_score in reranked]
        except Exception as e:
            # Reranker unavailable (e.g. the code-rerank model isn't cached on this machine) ->
            # fall back to the fused ranking instead of failing the query. Keeps machines without
            # the 2.2GB model working at the floor. (ADR 0011)
            import sys
            print(f"WARN: reranker unavailable, using fused ranking: {e}", file=sys.stderr)
            fused = sorted(cos_scores_for_ranking.items(), key=lambda kv: -kv[1])[:top]
    else:
        fused = sorted(cos_scores_for_ranking.items(), key=lambda kv: -kv[1])[:top]

    results: list[dict] = []
    for rank, (idx, score) in enumerate(fused, 1):
        m = meta[idx]
        results.append(
            {
                "rank": rank,
                "rrf": round(float(score), 4),
                "cos": round(float(cos[idx]), 3),
                "bm25": round(float(bm_scores[idx]), 2),
                "reranked": rerank,
                "source_type": m["source_type"],
                "repo": m["repo"],
                "language": m["language"],
                "symbol": m["symbol"],
                "path": m["path"],
                "start_line": m["start_line"],
                "end_line": m["end_line"],
                "text": m["text"],
            }
        )
    # Query telemetry (local sqlite, powers report.py). Default OFF in the
    # public distribution — opt in with RAG_QLOG=on.
    if os.environ.get("RAG_QLOG", "off").lower() in ("on", "1", "true"):
        _log_query(query, scope_types, scope_repos, cwd, rerank, results)
    return results


def _log_query(
    query: str,
    scope_types: list[str] | None,
    scope_repos: list[str] | None,
    cwd: str | None,
    rerank: bool,
    results: list[dict],
) -> None:
    try:
        conn = sqlite3.connect(QLOG, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA busy_timeout=10000")  # WAL+timeout hardening
        conn.execute(
            """CREATE TABLE IF NOT EXISTS queries (
                ts REAL NOT NULL,
                cwd TEXT, query TEXT, scope_types TEXT, scope_repos TEXT,
                rerank INTEGER, top_score REAL, top_path TEXT, n_results INTEGER
            )"""
        )
        top_score = float(results[0]["cos"]) if results else 0.0
        top_path = results[0]["path"] if results else ""
        conn.execute(
            "INSERT INTO queries VALUES (?,?,?,?,?,?,?,?,?)",
            (
                __import__("time").time(),
                cwd or _safe_cwd(),
                query[:500],
                ",".join(scope_types or []),
                ",".join(scope_repos or []),
                1 if rerank else 0,
                top_score,
                top_path,
                len(results),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
