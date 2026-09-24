"""Central configuration for shelfmark.

Everything the engine needs to know about YOUR machine lives in two places:

1. Environment variables (all optional):
   RAG_HOME     data dir for index.sqlite etc.   (default: ~/.shelfmark)
   RAG_DB       explicit index path               (default: $RAG_HOME/index.sqlite)
   RAG_SOURCES  path to sources.yaml              (default: $RAG_HOME/sources.yaml)
   RAG_MODEL    sentence-transformers model name  (default: intfloat/multilingual-e5-small)
   RAG_DIM      embedding dimension of RAG_MODEL  (default: 384)

2. sources.yaml — what to index (auto-created on first run if missing):
   repos:      code repos to index (code + docs + CHANGELOG + git commits)
   sources:    [{type: <label>, glob: <pattern>}] markdown corpora (notes, docs, ...)
   code_globs: loose script globs indexed as source_type=workstation-code
   clients:    per-client knowledge layers, each in its own index file
               (see "Client layers" below)

Client layers: a client's business knowledge is indexed into its own file,
never into the general index, so dropping that file is the purge and no query
filter can be forgotten. A query reads the general index plus the ACTIVE
client's file only. The active client comes from the process (RAG_CLIENT, or
the process cwd under a client root), never from a tool-call argument.

Import from here; never hardcode paths in engine modules.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RAG_HOME", "~/.shelfmark")).expanduser()
DB = Path(os.environ.get("RAG_DB") or ROOT / "index.sqlite")
CLIENT_ENV = "RAG_CLIENT"
QLOG = ROOT / "queries.sqlite"
MODEL_NAME = os.environ.get("RAG_MODEL", "intfloat/multilingual-e5-small")
DIM = int(os.environ.get("RAG_DIM", "384"))

SOURCES_FILE = Path(
    os.environ.get("RAG_SOURCES") or ROOT / "sources.yaml"
).expanduser()

_STARTER_SOURCES_YAML = """\
# shelfmark corpus configuration.
# Fill in what to index, then rerun. ~ and $VARS are expanded in every path.

# Code repos: indexed for source code (py/ts/js/sh), docs/**/*.md, README.md,
# CHANGELOG.md, docs/specs/**, docs/roadmap.md, and the last 180 days of git
# commit messages.
repos: []
# repos:
#   - ~/dev/my-main-project

# Markdown corpora: type is a free label you filter on at query time
# (--scope <type>).
sources: []
# sources:
#   - type: memory
#     glob: ~/notes/memory/**/*.md

# Loose scripts outside any repo, indexed as source_type=workstation-code.
code_globs: []
# code_globs:
#   - ~/scripts/*.sh

# Client layers. Files under a client's roots (or with `client: <slug>` in
# their frontmatter) go to that client's own index file, never the general one.
# `client: none` in frontmatter sends a note to the general index even from
# inside a client root. Queries read general + the active client only.
clients: {}
# clients:
#   acme:
#     roots:
#       - ~/dev/acme-app
#       - ~/notes/acme
#     db: ~/.shelfmark/index.client-acme.sqlite   # optional; this is the default
"""


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(p)))


def _load_sources_file() -> dict:
    if not SOURCES_FILE.exists():
        SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
        SOURCES_FILE.write_text(_STARTER_SOURCES_YAML, encoding="utf-8")
        print(
            f"shelfmark: no config found — wrote a starter one to {SOURCES_FILE}. "
            "Add your repos/notes there, then rerun.",
            file=sys.stderr,
        )
        return {}
    try:
        data = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SystemExit(f"shelfmark: malformed {SOURCES_FILE}: {e}")
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"shelfmark: {SOURCES_FILE} must be a YAML mapping")
    return data


_raw = _load_sources_file()

# Code repos: indexed for code files, docs/**/*.md, README, CHANGELOG, specs,
# roadmap, and recent git commits.
CURATED_REPOS: list[Path] = [Path(_expand(p)) for p in _raw.get("repos", [])]

# Markdown corpora: (source_type, glob). source_type is a free label you filter
# on at query time (--scope), e.g. memory, notes, docs, standards.
SOURCES: list[tuple[str, str]] = [
    (str(s["type"]), _expand(s["glob"]))
    for s in _raw.get("sources", [])
    if isinstance(s, dict) and "type" in s and "glob" in s
]

# Loose script globs outside any repo (indexed as workstation-code).
WORKSTATION_CODE_GLOBS: list[str] = [_expand(g) for g in _raw.get("code_globs", [])]

# Slugs become file names: restrict them so a slug can never escape ROOT.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _load_clients(raw: object) -> dict[str, dict]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise SystemExit(f"shelfmark: 'clients' in {SOURCES_FILE} must be a mapping")
    out: dict[str, dict] = {}
    for slug, spec in raw.items():
        slug = str(slug)
        if not _SLUG_RE.match(slug) or slug == "none":
            raise SystemExit(f"shelfmark: invalid client slug {slug!r} (use a-z, 0-9, '-'; 'none' is reserved)")
        spec = spec if isinstance(spec, dict) else {}
        roots = [Path(_expand(r)).resolve() for r in spec.get("roots", []) or []]
        db = Path(_expand(spec["db"])) if spec.get("db") else ROOT / f"index.client-{slug}.sqlite"
        out[slug] = {"roots": roots, "db": db}
    return out


CLIENTS: dict[str, dict] = _load_clients(_raw.get("clients"))


def client_db(slug: str) -> Path:
    return CLIENTS[slug]["db"]


def client_for_path(path: Path | str) -> str | None:
    """Client whose roots contain path (deepest root wins), else None."""
    p = Path(path).resolve()
    best: tuple[int, str] | None = None
    for slug, spec in CLIENTS.items():
        for root in spec["roots"]:
            if p == root or root in p.parents:
                depth = len(root.parts)
                if best is None or depth > best[0]:
                    best = (depth, slug)
    return best[1] if best else None


def active_client() -> str | None:
    """Client for this process: RAG_CLIENT wins ('none' = general only), else
    the process cwd. Deliberately takes no argument: a tool-call supplied cwd
    must never be able to switch a session into another client's layer."""
    env = os.environ.get(CLIENT_ENV, "").strip()
    if env:
        if env == "none":
            return None
        if env not in CLIENTS:
            # Fail closed per query (ValueError, not SystemExit: a long-lived MCP
            # server must report the error, not die or fall back to general).
            raise ValueError(f"shelfmark: {CLIENT_ENV}={env!r} is not a configured client")
        return env
    try:
        cwd = os.getcwd()
    except OSError:
        return None
    return client_for_path(cwd)


def query_dbs() -> list[Path]:
    """Index files a query may read: general + the active client's, nothing else."""
    slug = active_client()
    return [DB] if slug is None else [DB, client_db(slug)]


def all_dbs() -> list[Path]:
    return [DB, *(spec["db"] for spec in CLIENTS.values())]
