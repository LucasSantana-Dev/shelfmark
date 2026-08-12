#!/usr/bin/env python3
"""Weekly RAG observability report. Zero-hit queries + stale chunks.

Reads $RAG_HOME/queries.sqlite + index.sqlite, writes $RAG_HOME/weekly.md.
Requires query telemetry (RAG_QLOG=on). Safe to run frequently; idempotent.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from config import QLOG, ROOT
DB = ROOT / "index.sqlite"
OUT = ROOT / "weekly.md"

WINDOW_DAYS = 7
ZERO_HIT_THRESHOLD = 0.25
SINCE = time.time() - WINDOW_DAYS * 86400


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n\n"


def zero_hits() -> list[tuple]:
    if not QLOG.exists():
        return []
    conn = sqlite3.connect(QLOG)
    rows = conn.execute(
        "SELECT query, top_score, top_path, ts FROM queries "
        "WHERE ts >= ? AND top_score < ? ORDER BY ts DESC LIMIT 50",
        (SINCE, ZERO_HIT_THRESHOLD),
    ).fetchall()
    conn.close()
    return rows


def freq_queries() -> list[tuple]:
    if not QLOG.exists():
        return []
    conn = sqlite3.connect(QLOG)
    rows = conn.execute(
        "SELECT query, COUNT(*) c, AVG(top_score) s FROM queries "
        "WHERE ts >= ? GROUP BY query ORDER BY c DESC LIMIT 15",
        (SINCE,),
    ).fetchall()
    conn.close()
    return rows


def file_sha(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def stale_chunks() -> list[tuple[str, str]]:
    if not DB.exists():
        return []
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT path, file_sha FROM chunks GROUP BY path, file_sha"
    ).fetchall()
    conn.close()
    stale: list[tuple[str, str]] = []
    for raw_path, indexed_sha in rows:
        if raw_path.startswith("git:"):
            continue
        path = Path(raw_path)
        if not path.exists():
            stale.append(("missing", raw_path))
            continue
        if not path.is_file():
            continue
        try:
            if file_sha(path) != indexed_sha:
                stale.append(("modified", raw_path))
        except OSError:
            stale.append(("unreadable", raw_path))
    return stale[:50]


def index_stats() -> dict:
    conn = sqlite3.connect(DB)
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    by_type = conn.execute(
        "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type ORDER BY 2 DESC"
    ).fetchall()
    by_repo = conn.execute(
        "SELECT repo, COUNT(*) FROM chunks WHERE repo IS NOT NULL GROUP BY repo ORDER BY 2 DESC"
    ).fetchall()
    conn.close()
    return {"total": total, "by_type": by_type, "by_repo": by_repo}


def main() -> int:
    stats = index_stats()
    zh = zero_hits()
    fq = freq_queries()
    stale = stale_chunks()

    parts = [f"# RAG Weekly Report — {time.strftime('%Y-%m-%d')}\n"]

    type_lines = "\n".join(f"- {t}: {n}" for t, n in stats["by_type"])
    repo_lines = "\n".join(f"- {r}: {n}" for r, n in stats["by_repo"])
    parts.append(
        section(
            "Index stats",
            f"Total chunks: **{stats['total']}**\n\n"
            f"By source type:\n{type_lines}\n\nBy repo:\n{repo_lines}",
        )
    )

    if fq:
        parts.append(
            section(
                "Most-run queries (last 7d)",
                "\n".join(f"- `{q[:80]}`  ×{c}  top_cos={s:.2f}" for q, c, s in fq),
            )
        )
    else:
        parts.append(section("Most-run queries (last 7d)", "_no queries logged yet_"))

    if zh:
        lines = [
            f"- `{q[:100]}`  top_cos={score:.2f}  "
            f"at {time.strftime('%m-%d %H:%M', time.localtime(ts))}"
            for q, score, _, ts in zh
        ]
        parts.append(
            section(
                f"Zero-hit queries (cos < {ZERO_HIT_THRESHOLD}, last 7d) — corpus gaps to close",
                "\n".join(lines),
            )
        )
    else:
        parts.append(
            section(
                f"Zero-hit queries (cos < {ZERO_HIT_THRESHOLD})",
                "_none — corpus covers all recent queries_",
            )
        )

    if stale:
        parts.append(
            section(
                "Stale chunks (missing or modified files)",
                "\n".join(f"- {status}: {path}" for status, path in stale)
                + "\n\nRebuild or run incremental reindex for these files.",
            )
        )
    else:
        parts.append(section("Stale chunks", "_none_"))

    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
