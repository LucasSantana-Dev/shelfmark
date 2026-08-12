#!/usr/bin/env python3
"""Chunk + embed configured sources into $RAG_HOME/index.sqlite.

Markdown corpora (notes, docs, standards — whatever sources.yaml declares) plus
source-code dirs for the configured repo whitelist. Language-aware chunkers
split by symbol where possible. See config.py / sources.yaml.example.

Usage:
  build.py                                           # full rebuild
  build.py --incremental <file> [...files]           # reindex specific files
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import subprocess
from chunkers import chunk_file, detect_language

# Iter 1 lever (memory-first RAG plan): emit one extra high-signal "purpose" chunk per
# memory/skills note from frontmatter (name + description + project/topic/type tags).
# Scope-selectable: RAG_PURPOSE_CHUNK = 0/off | 1/both | skills | memory | "skills,memory".
# VERDICT 2026-06-24 (ADR-0037): **REVERT, default off.** On correctly-labeled eval cases the
# chunk is +0.0pp — chunkers already dump frontmatter into the body, so it's redundant; the
# apparent skills "+12.5pp" was a label artifact (un-curated cases), erased once the skills
# eval was curated (0.250→0.625 was curation, not this chunk). memory was -2.3pp. Kept behind
# the flag (reversible, underpowered n=8 eval) in case a broader skills eval shows a gain.
def _purpose_chunk_types() -> tuple:
    v = os.environ.get("RAG_PURPOSE_CHUNK", "off").strip().lower()
    if v in ("0", "off", "false", ""):
        return ()
    if v in ("1", "on", "true", "both"):
        return ("memory", "skills")
    return tuple(t.strip() for t in v.split(",") if t.strip() in ("memory", "skills"))


PURPOSE_CHUNK_TYPES = _purpose_chunk_types()

# Token-efficiency (2026-06-25): ephemeral/operational records (handoffs, session snapshots) are
# large (~520 tok/chunk), time-bound, and low-reuse — indexing them in FULL dilutes ranking and
# inflates retrieval cost. For these types, index ONE concise CARD (title + first content line)
# instead of full chunks: still findable, ~10x lighter. Configurable via RAG_CARD_ONLY (csv).
CARD_ONLY_TYPES = tuple(
    t.strip() for t in os.environ.get("RAG_CARD_ONLY", "handoffs").split(",") if t.strip()
)
# Path-pattern card-ifying (env RAG_CARD_PATHS) — DEFAULT OFF. D2 (2026-06-25) tried carding
# session-snapshot memory notes (sessionend/precompact) to save tokens, but it REGRESSED the gate:
# memory-target 0.884→0.767 and holdout-memory 1.000→0.900 (−10pp). Snapshots are NOT purely
# ephemeral — ~3-4 eval memory cases retrieve their content. So the mechanism stays but defaults
# off; do NOT card source_type=memory paths. (Handoffs, which ARE pure ephemeral, stay carded via
# CARD_ONLY_TYPES.)
CARD_ONLY_PATH_PATTERNS = tuple(
    t.strip().lower() for t in os.environ.get("RAG_CARD_PATHS", "").split(",") if t.strip()
)


def _is_card_path(path) -> bool:
    n = path.name.lower()
    return any(pat in n for pat in CARD_ONLY_PATH_PATTERNS)

from config import CURATED_REPOS, DB, MODEL_NAME, ROOT, WORKSTATION_CODE_GLOBS
from config import SOURCES as _CONFIG_SOURCES

HOME = Path.home()
MAX_FILE_BYTES = 200_000
EXCLUDED_DIR_PARTS = {
    "site-packages",
    "htmlcov",
    ".eggs",
    ".tox",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "coverage",
    ".git",
    ".next",
    ".turbo",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "test-results",
    "playwright-report",
    ".storybook",
    ".docusaurus",
    ".worktrees",
    "worktrees",
    ".wt-res",
    ".wt-luckynotify",
    ".wt-notify",
    ".wt-ufw",
    ".wt-strict",
    ".wt-renovate",
    ".wt-disc",
    ".wt-fix",
    ".wt-notify-clean",
}
CODE_EXTS = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".sh", ".bash", ".zsh"}

# Markdown corpora come from sources.yaml (see config.py). Per-repo doc sources
# are derived below for every configured repo.
SOURCES: list[tuple[str, str]] = list(_CONFIG_SOURCES)
for repo in CURATED_REPOS:
    if not repo.is_dir():
        continue
    SOURCES.append(("changelog", str(repo / "CHANGELOG.md")))
    # spec MUST precede repo-docs: docs/**/*.md subsumes docs/specs/**, and the
    # seen_real dedup assigns first-match-wins. With repo-docs first, zero
    # spec-typed chunks exist and scope-filtered retrieval (eval + recall
    # --scope spec) misses everything (hit@5 spec 1.0 -> 0.0, found 2026-07-23).
    SOURCES.append(("spec", str(repo / "docs/specs/**/*.md")))
    # roadmap MUST also precede repo-docs, same subsumption: docs/**/*.md
    # matches docs/roadmap.md, so a later roadmap entry gets zero typed chunks
    # (found 2026-07-29: roadmap scope hit@5 1.0 -> 0.0).
    SOURCES.append(("roadmap", str(repo / "docs/roadmap.md")))
    SOURCES.append(("repo-docs", str(repo / "docs/**/*.md")))
    SOURCES.append(("repo-readme", str(repo / "README.md")))
    SOURCES.append(("serena", str(repo / ".serena/memories/*.md")))
for glob in WORKSTATION_CODE_GLOBS:
    SOURCES.append(("workstation-code", glob))


def has_graph(repo: Path) -> bool:
    """Repo has a graphify knowledge graph — its raw code is NOT embedded
    (graph queries answer structural questions better); only its distilled
    graph artifacts (GRAPH_REPORT.md, wiki) and markdown docs are indexed."""
    return (repo / "graphify-out" / "graph.json").exists()


# Graph artifacts: distilled summaries are high-signal embedding material.
GRAPH_ARTIFACT_ROOTS = CURATED_REPOS
for _root in GRAPH_ARTIFACT_ROOTS:
    if (_root / "graphify-out").is_dir():
        SOURCES.append(("graph", str(_root / "graphify-out/GRAPH_REPORT.md")))
        SOURCES.append(("graph", str(_root / "graphify-out/wiki/*.md")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    repo TEXT,
    language TEXT,
    symbol TEXT,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    file_sha TEXT NOT NULL,
    mtime REAL NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS chunks_type ON chunks(source_type);
CREATE INDEX IF NOT EXISTS chunks_repo ON chunks(repo);
"""


def iter_md_sources() -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    # Deduplicate by realpath — multiple configured globs may reach the same
    # file via symlinks; without this every such file would be indexed N×.
    seen_real: set[Path] = set()
    for stype, glob in SOURCES:
        base = Path(glob.split("*", 1)[0])
        if "**" in glob:
            pattern = glob.split("/**/", 1)[1]
            for p in base.rglob(pattern):
                if p.is_file():
                    rp = p.resolve()
                    if rp not in seen_real:
                        seen_real.add(rp)
                        results.append((stype, p))
        elif "*" in glob:
            parent = Path(glob).parent
            name_pat = Path(glob).name
            if "*" in str(parent):
                # Handle double-wildcard patterns like .../*/memory/*.md
                # Extract everything after the first wildcard (e.g., /memory/*.md)
                grandparent = Path(str(parent).split("*", 1)[0])
                rest_pattern = glob.split("*", 1)[1].lstrip("/")
                for sub in grandparent.glob("*"):
                    if sub.is_dir():
                        for p in sub.glob(rest_pattern):
                            if p.is_file():
                                rp = p.resolve()
                                if rp not in seen_real:
                                    seen_real.add(rp)
                                    results.append((stype, p))
            else:
                for p in parent.glob(name_pat):
                    if p.is_file():
                        rp = p.resolve()
                        if rp not in seen_real:
                            seen_real.add(rp)
                            results.append((stype, p))
        else:
            p = Path(glob)
            if p.is_file():
                rp = p.resolve()
                if rp not in seen_real:
                    seen_real.add(rp)
                    results.append((stype, p))
    # Quarantine tier (memory security gate, 2026-07-28): notes under any
    # quarantine/ dir are never indexed until promoted out. See memory-trust.sh.
    return [(s, p) for s, p in results if "/quarantine/" not in str(p)]


GIT_LOG_DAYS = 180


def collect_commit_chunks() -> list[dict]:
    """Run git log on each curated repo; one chunk per commit (subject + body head)."""
    rows: list[dict] = []
    for repo in CURATED_REPOS:
        if not (repo / ".git").exists():
            continue
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "log",
                    f"--since={GIT_LOG_DAYS}.days",
                    "--no-merges",
                    "--pretty=format:%H%x01%ai%x01%an%x01%s%x01%b%x02",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            continue
        raw = proc.stdout
        for entry in raw.split("\x02"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split("\x01", 4)
            if len(parts) < 4:
                continue
            sha, date_str, author, subject = parts[:4]
            body = parts[4] if len(parts) == 5 else ""
            body_head = body.strip().splitlines()
            body_top = "\n".join(body_head[:6])[:1200]
            text = f"{subject}\n\n{body_top}".strip()
            rows.append(
                {
                    "source_type": "commit",
                    "repo": repo.name,
                    "language": "git",
                    "symbol": sha[:7],
                    "path": f"git:{repo.name}@{sha}",
                    "start": 0,
                    "end": 0,
                    "text": text[:4000],
                    "sha": sha,
                    "mtime": 0.0,
                    "meta": f"{author} · {date_str}",
                }
            )
    return rows


def iter_code_sources() -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for repo in CURATED_REPOS:
        if not repo.is_dir():
            continue
        if has_graph(repo):
            continue  # graph answers structure; raw code not embedded
        for p in repo.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in CODE_EXTS:
                continue
            if is_excluded_path(p.relative_to(repo)):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            results.append(("code", p))
    return results


def is_excluded_path(relative_path: Path) -> bool:
    for part in relative_path.parts:
        if part in EXCLUDED_DIR_PARTS:
            return True
        if part.startswith(".venv"):   # .venv_ci, .venv_dev, etc.
            return True
        if part.startswith(".wt-"):
            return True
    return False


def file_sha(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def classify_repo(path: Path) -> str | None:
    # Compare against both the configured repo path and its resolved target —
    # a configured path may be a symlink, and resolve() expands it, so matching
    # only the unresolved path never hits.
    rp = path.resolve()
    for repo in CURATED_REPOS:
        for base in (repo, repo.resolve()):
            try:
                rp.relative_to(base)
                return repo.name
            except ValueError:
                continue
    return None


def classify_type(path: Path) -> str:
    s = str(path)
    if "/memory/" in s:
        return "memory"
    if "/plans/" in s:
        return "plans"
    if "/adrs/" in s and s.endswith(".md"):
        return "adrs"
    if "/handoffs/" in s:
        return "handoffs"
    # Only SKILL.md is a "skill" — a non-SKILL.md file under a skills dir (e.g. skill-creator
    # benchmark workspace outputs like .../<skill>-workspace/iteration-*/.../response.md) is NOT
    # a skill and must not pollute the skills scope. Matches SOURCES (skills = */SKILL.md only).
    if ("/.claude/skills/" in s or "/.agents/skills/" in s) and s.endswith("/SKILL.md"):
        return "skills"
    if "/.codex/" in s:
        return "codex"
    if "/standards/" in s and s.endswith(".md"):
        return "standards"
    if s.endswith("CHANGELOG.md"):
        return "changelog"
    if s.endswith("README.md"):
        return "repo-readme"
    if s.endswith("/docs/roadmap.md") or s.endswith("docs/roadmap.md"):
        return "roadmap"
    if "/graphify-out/" in s:
        return "graph"
    if "/docs/specs/" in s and s.endswith(".md"):
        return "spec"
    if "/docs/" in s and s.endswith(".md"):
        return "repo-docs"
    if any(s.startswith(g.split("*", 1)[0]) for g in WORKSTATION_CODE_GLOBS):
        return "workstation-code"
    if path.suffix.lower() in CODE_EXTS:
        return "code"
    return "other"


def _parse_frontmatter(text: str) -> dict:
    """Best-effort YAML frontmatter parse. Returns {} when absent or malformed."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        fm = yaml.safe_load(text[3:end])
    except Exception:
        return {}
    return fm if isinstance(fm, dict) else {}


# Tag axes that carry ranking signal (the Second-Brain taxonomy). status/* is dropped:
# status/active sits on nearly every note, so it has no discriminative value.
_PURPOSE_TAG_AXES = ("project/", "topic/", "type/")


def _collect_tags(fm: dict) -> list[str]:
    raw = fm.get("tags")
    if not isinstance(raw, list):
        meta = fm.get("metadata")
        raw = meta.get("tags") if isinstance(meta, dict) else None
    tags = [str(t) for t in raw] if isinstance(raw, list) else []
    return [t for t in tags if t.startswith(_PURPOSE_TAG_AXES)]


def build_purpose_chunk(stype: str, path: Path, text: str):
    """Synthesize one high-signal chunk from a note's frontmatter so its human-written
    description and Second-Brain tags become directly matchable — "quality memory, not
    just MD files". Returns a Chunk (start, end, body, symbol) or None when there is no
    usable frontmatter signal (so we never inject a bare-filename chunk that adds noise)."""
    fm = _parse_frontmatter(text)
    if not fm:
        return None
    name = (str(fm.get("name") or "")).strip() or path.stem
    desc = (str(fm.get("description") or "")).strip()
    parts = [f"{name}: {desc}" if desc else name]
    if stype == "memory":
        tags = _collect_tags(fm)
        if tags:
            parts.append("tags: " + " ".join(tags))
    elif stype == "skills":
        trig = fm.get("triggers")
        if isinstance(trig, list) and trig:
            parts.append("triggers: " + ", ".join(str(t) for t in trig))
    # Worth a chunk only if it carries more than the bare filename.
    if not desc and len(parts) == 1:
        return None
    body = " · ".join(p for p in parts if p).strip()
    if not body:
        return None
    # (0, 0) marks a SYNTHETIC purpose chunk — real chunkers are 1-indexed, so line 0 never
    # collides with a genuine 1-line chunk (lets `start_line=0` count purpose chunks exactly).
    return (0, 0, body[:1000], name)


def build_card_chunk(stype: str, path: Path, text: str):
    """One concise CARD for an ephemeral/operational note — title + first content line —
    in place of full chunks, so it stays findable at ~10x lower token cost. Returns a Chunk
    (0, 0, body, title) or None. (0,0) = synthetic marker (chunkers are 1-indexed)."""
    title = ""
    summary = ""
    in_fm = False
    for line in text.splitlines():
        ls = line.strip()
        if ls == "---":
            in_fm = not in_fm
            continue
        if in_fm or not ls:
            continue
        if ls.startswith("#"):
            if not title:
                title = ls.lstrip("#").strip()
            continue
        if not summary:
            summary = ls
        if title and summary:
            break
    title = title or path.stem
    body = " · ".join(p for p in (title, summary[:240]) if p).strip(" ·")
    if not body:
        return None
    return (0, 0, body[:400], title)


def connect() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        # Corruption probe (router pattern): quick_check first; a broken DB is
        # renamed aside and rebuilt from sources instead of crashing mid-build.
        try:
            probe = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
            ok = probe.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            probe.close()
            if not ok:
                raise sqlite3.DatabaseError("quick_check failed")
        except sqlite3.DatabaseError:
            aside = DB.with_suffix(f".corrupt-{int(time.time())}")
            DB.rename(aside)
            print(f"corrupt index moved aside: {aside.name} (will rebuild fresh)")
    conn = sqlite3.connect(DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA busy_timeout=10000")  # WAL+timeout hardening
    conn.executescript(SCHEMA)
    return conn


def backup_db(keep: int = 3) -> None:
    """VACUUM INTO snapshot after a successful build; keep newest `keep`."""
    backups = sorted(ROOT.glob("index.backup-*.sqlite"))
    dest = ROOT / f"index.backup-{int(time.time())}.sqlite"
    try:
        src = sqlite3.connect(DB)
        dst = sqlite3.connect(dest)
        src.backup(dst)
        dst.close(); src.close()
        for old in backups[: max(0, len(backups) + 1 - keep)]:
            old.unlink()
        print(f"backup: {dest.name}")
    except sqlite3.Error as e:
        print(f"backup skipped: {e}")


def embed(model, texts: list[str]) -> np.ndarray:
    # E5 model requires "passage: " prefix for indexed chunks; this is added
    # at embed time only, not stored in the database.
    prefixed_texts = [f"passage: {t}" for t in texts]
    vecs = model.encode(
        prefixed_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return vecs.astype(np.float32)


def index_files(
    conn: sqlite3.Connection, model, files: list[tuple[str, Path]], purge_paths: list[str]
) -> int:
    for p in purge_paths:
        conn.execute("DELETE FROM chunks WHERE path = ?", (p,))
    total = 0
    batch_texts: list[str] = []
    batch_meta: list[dict] = []
    for stype_hint, path in files:
        if stype_hint == "code":
            _repo = next(
                (r for r in CURATED_REPOS if str(path).startswith(str(r) + "/")), None
            )
            if _repo is not None and has_graph(_repo):
                continue  # graphed repo: code chunks excluded (incremental path too)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if stype_hint == "memory":
            # Bitemporal supersession (2026-07-28): a note with valid_to set was
            # superseded by a newer fact. It stays on disk (supersede-not-overwrite)
            # but leaves the index so stale facts stop competing as distractors.
            # Regex, not YAML: notes with malformed frontmatter (e.g. unquoted
            # colons in description) make yaml.safe_load raise, the dict parse
            # returns {}, and the note leaks back into the index (observed in
            # production with an archived note). \S requires an actual value so an empty
            # "valid_to:" (still valid) does not match.
            if re.search(r"^valid_to:\s*\S", text, re.M):
                continue
        sha = file_sha(path)
        mtime = path.stat().st_mtime
        stype = stype_hint if stype_hint != "code" else "code"
        repo = classify_repo(path)
        language = detect_language(path)
        if stype in CARD_ONLY_TYPES or _is_card_path(path):
            card = build_card_chunk(stype, path, text)
            chunks = [card] if card is not None else []  # one card, not full chunks
        else:
            chunks = list(chunk_file(path, text))
            if stype in PURPOSE_CHUNK_TYPES:
                pc = build_purpose_chunk(stype, path, text)
                if pc is not None:
                    chunks.insert(0, pc)  # high-signal frontmatter chunk first
        for start, end, body, symbol in chunks:
            # Contextual chunk prefix for embedding: adds source type, repo, file name, symbol
            # Format: "<source_type> | <repo or ''> | <file name or doc title> | <symbol or ''>\n<chunk body>"
            # This is only used for embedding, not stored in the database.
            file_name = path.name
            context_prefix = f"{stype} | {repo or ''} | {file_name} | {symbol or ''}"
            contextualized_text = f"{context_prefix}\n{body[:4000]}"

            batch_texts.append(contextualized_text)
            batch_meta.append(
                {
                    "source_type": stype,
                    "repo": repo,
                    "language": language,
                    "symbol": symbol,
                    "path": str(path),
                    "start": start,
                    "end": end,
                    "text": body[:4000],
                    "sha": sha,
                    "mtime": mtime,
                }
            )
            if len(batch_texts) >= 64:
                _flush(conn, model, batch_texts, batch_meta)
                total += len(batch_texts)
                batch_texts.clear()
                batch_meta.clear()
    if batch_texts:
        _flush(conn, model, batch_texts, batch_meta)
        total += len(batch_texts)
    conn.commit()
    return total


def _flush(conn, model, texts, meta):
    vecs = embed(model, texts)
    rows = []
    for m, vec in zip(meta, vecs):
        rows.append(
            (
                m["source_type"],
                m["repo"],
                m["language"],
                m["symbol"],
                m["path"],
                m["start"],
                m["end"],
                m["text"],
                m["sha"],
                m["mtime"],
                vec.tobytes(),
            )
        )
    conn.executemany(
        "INSERT INTO chunks (source_type, repo, language, symbol, path, start_line, end_line, text, file_sha, mtime, embedding) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", nargs="+", help="specific files to reindex")
    ap.add_argument("--no-code", action="store_true", help="skip source code ingestion")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    conn = connect()
    started = time.time()

    if args.incremental:
        targets: list[tuple[str, Path]] = []
        purge: list[str] = []
        for raw in args.incremental:
            p = Path(raw).expanduser().resolve()
            purge.append(str(p))
            if not p.exists():
                continue
            if "/quarantine/" in str(p):
                # Quarantine tier: purge any stale chunks but never index.
                continue
            stype = classify_type(p)
            targets.append((stype, p))
        written = index_files(conn, model, targets, purge)
        print(f"incremental: {len(targets)} files, {written} chunks, {time.time()-started:.1f}s")
    else:
        conn.execute("DELETE FROM chunks")
        md_files = iter_md_sources()
        code_files = [] if args.no_code else iter_code_sources()
        files = md_files + code_files
        written = index_files(conn, model, files, [])
        commits_written = 0
        if not args.no_code:
            commit_rows = collect_commit_chunks()
            if commit_rows:
                texts = [r["text"] for r in commit_rows]
                # Embed in batches to share with index_files pathway
                for i in range(0, len(commit_rows), 64):
                    batch = commit_rows[i : i + 64]
                    # Add contextual prefix for commits: "commit | <repo>"
                    contextualized_texts = [
                        f"commit | {r['repo']}\n{r['text']}" for r in batch
                    ]
                    vecs = embed(model, contextualized_texts)
                    sql_rows = []
                    for m, vec in zip(batch, vecs):
                        sql_rows.append(
                            (
                                m["source_type"],
                                m["repo"],
                                m["language"],
                                m["symbol"],
                                m["path"],
                                m["start"],
                                m["end"],
                                m["text"],
                                m["sha"],
                                m["mtime"],
                                vec.tobytes(),
                            )
                        )
                    conn.executemany(
                        "INSERT INTO chunks (source_type, repo, language, symbol, path, start_line, end_line, text, file_sha, mtime, embedding) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        sql_rows,
                    )
                    commits_written += len(batch)
                conn.commit()
        print(
            f"full rebuild: md={len(md_files)} code={len(code_files)} commits={commits_written} chunks={written+commits_written} t={time.time()-started:.1f}s"
        )
    backup_db()
    return 0


if __name__ == "__main__":
    sys.exit(main())
