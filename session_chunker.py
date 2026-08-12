#!/usr/bin/env python3
"""Chunk Claude Code session JSONLs into the RAG index.

Each session JSONL contains an array of message objects. We group by
tool_use_id (input + result as one unit) and embed each group as a chunk.

Run:
    python3 session_chunker.py [days=30]

Idempotent: deletes prior `source_type='session'` chunks before inserting.
"""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

from config import DB as INDEX_DB

HOME = Path.home()
# Claude Code session transcripts (JSONL). Point RAG_SESSIONS_DIR elsewhere for
# other agent harnesses, or leave the default.
SESSIONS_DIR = Path(
    os.environ.get("RAG_SESSIONS_DIR", HOME / ".claude" / "projects")
).expanduser()
MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_MAX_CHARS = 4000  # ~1K tokens
SKIP_FILE_PATTERNS = ("acompact-", "agent-acompact-")  # auto-compaction summaries
BATCH = 32


def iter_session_files(days: int) -> Iterator[Path]:
    cutoff = time.time() - (days * 86400)
    for f in SESSIONS_DIR.rglob("*.jsonl"):
        if any(p in f.name for p in SKIP_FILE_PATTERNS):
            continue
        try:
            if f.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        yield f


def extract_text_from_message(msg: dict) -> str:
    """Pull the meaningful text from a Claude Code message envelope."""
    content = msg.get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool = block.get("name", "")
            inp = block.get("input", {})
            inp_str = json.dumps(inp, separators=(",", ":"))[:500]
            parts.append(f"[Tool: {tool}] {inp_str}")
        elif btype == "tool_result":
            c = block.get("content", "")
            if isinstance(c, list):
                c = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
            parts.append(f"[Result] {str(c)[:1000]}")
    text = "\n".join(p for p in parts if p)
    # Strip system reminders + persisted-output noise
    if "<system-reminder>" in text:
        text = text.split("<system-reminder>")[0]
    return text.strip()


def iter_session_chunks(days: int) -> Iterator[dict]:
    """Yield chunk dicts ready to embed/insert."""
    for f in iter_session_files(days):
        try:
            lines = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        # Aggregate into chunks of ~CHUNK_MAX_CHARS
        buffer: list[str] = []
        buffer_size = 0
        chunk_idx = 0

        def flush() -> dict | None:
            nonlocal chunk_idx, buffer, buffer_size
            if not buffer:
                return None
            text = "\n".join(buffer).strip()
            if len(text) < 100:
                buffer = []
                buffer_size = 0
                return None
            sha = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
            chunk = {
                "source_type": "session",
                "repo": None,
                "language": None,
                "symbol": None,
                "path": str(f),
                "start": chunk_idx,
                "end": chunk_idx + len(buffer),
                "text": text[:CHUNK_MAX_CHARS],
                "sha": sha,
                "mtime": f.stat().st_mtime,
            }
            chunk_idx += len(buffer)
            buffer = []
            buffer_size = 0
            return chunk

        for line in lines:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = extract_text_from_message(msg)
            if not text:
                continue
            if buffer_size + len(text) > CHUNK_MAX_CHARS and buffer:
                c = flush()
                if c:
                    yield c
            buffer.append(text)
            buffer_size += len(text)
        c = flush()
        if c:
            yield c


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"[sessions] scanning last {days}d of session files...", flush=True)

    chunks = list(iter_session_chunks(days))
    print(f"[sessions] collected {len(chunks)} chunks", flush=True)
    if not chunks:
        return 0

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    conn = sqlite3.connect(INDEX_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA busy_timeout=10000")  # WAL+timeout hardening
    conn.execute("DELETE FROM chunks WHERE source_type = 'session'")
    conn.commit()

    started = time.time()
    written = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        # E5 model requires "passage: " prefix for indexed chunks
        prefixed_texts = [f"passage: {t}" for t in texts]
        vecs = model.encode(prefixed_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        rows = [
            (c["source_type"], c["repo"], c["language"], c["symbol"],
             c["path"], c["start"], c["end"], c["text"],
             c["sha"], c["mtime"], vec.tobytes())
            for c, vec in zip(batch, vecs)
        ]
        conn.executemany(
            "INSERT INTO chunks (source_type, repo, language, symbol, path, "
            "start_line, end_line, text, file_sha, mtime, embedding) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        written += len(rows)
        if i % (BATCH * 8) == 0 and i > 0:
            print(f"[sessions] {written}/{len(chunks)} ({time.time()-started:.0f}s)", flush=True)
    conn.commit()
    conn.close()
    print(f"[sessions] done: {written} chunks in {time.time()-started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
