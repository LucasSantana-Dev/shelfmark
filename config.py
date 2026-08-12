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

Import from here; never hardcode paths in engine modules.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RAG_HOME", "~/.shelfmark")).expanduser()
DB = Path(os.environ.get("RAG_DB") or ROOT / "index.sqlite")
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
