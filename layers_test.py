#!/usr/bin/env python3
"""Tests for client layers: routing into per-client index files and query isolation.

No model download: a bag-of-words fake stands in for the embedding model, so
ranking here is lexical. What is under test is WHICH index files a note lands
in and which files a query may read, not ranking quality.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

TMP = Path(tempfile.mkdtemp(prefix="shelfmark-layers-")).resolve()
os.environ["RAG_HOME"] = str(TMP)
os.environ["RAG_SOURCES"] = str(TMP / "sources.yaml")
os.environ["RAG_SESSIONS_DIR"] = str(TMP / "sessions")
os.environ["RAG_RERANK_AUTO"] = "off"  # no cross-encoder download; routing is under test
os.environ.pop("RAG_CLIENT", None)

NOTES, ACME, BETA = TMP / "notes", TMP / "acme", TMP / "beta"
REAL, LINK = TMP / "real", TMP / "link"  # a notes dir reached through a symlink
for d in (NOTES, ACME, BETA, REAL, TMP / "sessions" / "p"):
    d.mkdir(parents=True)
LINK.symlink_to(REAL)
(TMP / "sources.yaml").write_text(textwrap.dedent(f"""\
    sources:
      - {{type: memory, glob: "{NOTES}/**/*.md"}}
      - {{type: memory, glob: "{ACME}/**/*.md"}}
      - {{type: memory, glob: "{BETA}/**/*.md"}}
      - {{type: memory, glob: "{LINK}/**/*.md"}}
    clients:
      acme:
        roots: ["{ACME}"]
      beta:
        roots: ["{BETA}"]
    """))


def note(path: Path, body: str, client: str | None = None) -> Path:
    fm = f"---\nname: {path.stem}\nclient: {client}\n---\n" if client else ""
    path.write_text(f"{fm}# {path.stem}\n\n{body}\n")
    return path


GENERAL = note(NOTES / "general.md", "rollback lesson: always keep a tested rollback path")
TAGGED_BETA = note(NOTES / "tagged-beta.md", "beta pricing zebra tiers are confidential", client="beta")
GHOST = note(NOTES / "ghost.md", "ghost client typo note octopus", client="ghost")
ACME_RULE = note(ACME / "acme-rule.md", "acme billing walrus rule charges on the fifth")
ACME_LESSON = note(ACME / "lesson.md", "retry lesson: make handlers idempotent", client="none")
LINKED = note(REAL / "linked.md", "linked lesson reached through a symlinked glob")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import indexer  # noqa: E402
import retrieval  # noqa: E402
import session_chunker  # noqa: E402


class FakeModel:
    def encode(self, texts, **_kw):
        out = np.zeros((len(texts), config.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in retrieval._tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % config.DIM
                out[i, h] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


MODEL = FakeModel()
retrieval._model = MODEL
CONNS: dict[Path, sqlite3.Connection] = {}


def conn_for(db: Path) -> sqlite3.Connection:
    if db not in CONNS:
        CONNS[db] = indexer.connect(db)
    return CONNS[db]


def paths_in(db: Path) -> set[str]:
    if not db.exists():
        return set()
    return {Path(r[0]).name for r in sqlite3.connect(db).execute("SELECT DISTINCT path FROM chunks")}


def found(query: str, cwd: str | None = None) -> set[str]:
    retrieval._cache.clear()
    return {Path(r["path"]).name for r in retrieval.search(query, top=10, scope_repos=["all"], cwd=cwd)}


GEN_DB, ACME_DB, BETA_DB = config.DB, config.client_db("acme"), config.client_db("beta")
indexer.index_files(conn_for, MODEL, indexer.iter_md_sources(), [])


def test_routing():
    assert paths_in(GEN_DB) == {"general.md", "lesson.md", "linked.md"}, paths_in(GEN_DB)
    assert paths_in(ACME_DB) == {"acme-rule.md"}, paths_in(ACME_DB)
    assert paths_in(BETA_DB) == {"tagged-beta.md"}, paths_in(BETA_DB)


def test_unknown_client_is_skipped_not_general():
    everywhere = paths_in(GEN_DB) | paths_in(ACME_DB) | paths_in(BETA_DB)
    assert "ghost.md" not in everywhere


def test_no_active_client_reads_general_only():
    os.chdir(TMP)
    assert config.active_client() is None
    assert "acme-rule.md" not in found("acme billing walrus rule")
    assert "tagged-beta.md" not in found("beta pricing zebra tiers")
    assert "general.md" in found("rollback lesson")


def test_active_client_sees_own_layer_never_other():
    os.chdir(ACME)
    assert config.active_client() == "acme"
    assert "acme-rule.md" in found("acme billing walrus rule")
    assert "tagged-beta.md" not in found("beta pricing zebra tiers")
    assert "lesson.md" in found("retry idempotent handlers")  # client: none lesson stays reachable
    os.chdir(TMP)


def test_tool_cwd_argument_cannot_switch_client():
    os.chdir(TMP)
    assert "tagged-beta.md" not in found("beta pricing zebra tiers", cwd=str(BETA))


def test_env_override_and_fail_closed():
    os.chdir(ACME)
    os.environ["RAG_CLIENT"] = "beta"
    try:
        hits = found("beta pricing zebra tiers acme billing walrus")
        assert "tagged-beta.md" in hits and "acme-rule.md" not in hits, hits
        os.environ["RAG_CLIENT"] = "none"
        assert "acme-rule.md" not in found("acme billing walrus rule")
        os.environ["RAG_CLIENT"] = "ghost"
        try:
            found("anything")
            raise AssertionError("unknown RAG_CLIENT must raise, not fall back")
        except ValueError:
            pass
    finally:
        os.environ.pop("RAG_CLIENT", None)
        os.chdir(TMP)


def test_note_changing_layer_leaves_old_index():
    note(GENERAL, "rollback lesson moved under acme", client="acme")
    indexer.index_files(conn_for, MODEL, [("memory", GENERAL)], [str(GENERAL)])
    assert "general.md" not in paths_in(GEN_DB)
    assert "general.md" in paths_in(ACME_DB)


def test_sessions_route_by_recorded_cwd():
    s = TMP / "sessions" / "p"
    (s / "in-acme.jsonl").write_text(
        json.dumps({"cwd": str(ACME), "type": "user", "message": {"role": "user", "content": "x" * 200}}) + "\n")
    (s / "no-cwd.jsonl").write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "y" * 200}}) + "\n")
    chunks = list(session_chunker.iter_session_chunks(days=1))
    by_file = {Path(c["path"]).name: c["db"] for c in chunks}
    assert "no-cwd.jsonl" not in by_file, by_file  # clients configured: unroutable -> skipped
    assert by_file.get("in-acme.jsonl") == ACME_DB, by_file


def test_frontmatter_variants_fail_closed():
    out = NOTES / "x.md"  # outside every client root: path routing would say general
    beta = config.client_db("beta")
    cases = {
        "\ufeff---\nclient: beta\n---\nbody": beta,               # BOM
        "---\nclient: beta  # primary\n---\nbody": beta,           # YAML comment
        "---\nmetadata:\n  client: beta\n---\nbody": beta,        # nested under metadata
        "---\r\nclient: beta\r\n---\r\nbody": beta,             # CRLF
        "---\nclient: [beta]\n---\nbody": None,                  # list: skip, not general
        "---\nclient:\n---\nbody": None,                         # empty: skip
        "---\nclient: beta: x: [\n---\nbody": None,              # unparseable YAML: skip
        "---\nclient: Beta\n---\nbody": None,                    # slugs are lowercase: skip
        '---\n"client": beta\n---\nbody': beta,                  # quoted key
        "---\n{client: beta}\n---\nbody": beta,                  # flow mapping
        "---\nproject:\n  client: beta\n---\nbody": None,       # client line nested elsewhere: skip
        "---\nname: x\n---\nclient: beta in the body": config.DB,  # body text is not frontmatter
    }
    for text, want in cases.items():
        got = indexer.route(out, text)
        assert got == want, (text, got, want)


def test_roots_and_db_validation():
    for bad in ({"x": {"roots": str(NOTES)}}, {"x": {"roots": ["/"]}}, {"x": {"roots": [str(Path.home())]}},
                {"x": {"roots": [str(Path.home().parent)]}}, {"x": {"roots": [str(TMP)]}},  # contains $HOME / RAG_HOME
                {"x": {"db": str(config.DB)}}, {"x": {"db": str(TMP / "d.sqlite")}, "y": {"db": str(TMP / "d.sqlite")}},
                *([{"x": {"db": str(config.DB).upper()}}] if sys.platform in ("darwin", "win32") else [])):
        try:
            config._load_clients(bad)
            raise AssertionError(f"must reject {bad}")
        except SystemExit:
            pass


def test_case_insensitive_roots():
    if sys.platform not in ("darwin", "win32"):
        return
    assert config.client_for_path(str(ACME).upper() + "/a.md") == "acme"


def test_symlinked_glob_layer_move():
    note(LINKED, "linked lesson now belongs to acme", client="acme")
    # main --incremental purges by resolved path; storage must use the same key.
    indexer.index_files(conn_for, MODEL, [("memory", LINKED.resolve())], [str(LINKED.resolve())])
    assert "linked.md" not in paths_in(GEN_DB), paths_in(GEN_DB)
    assert "linked.md" in paths_in(ACME_DB)


def test_orphan_client_index_detected():
    orphan = TMP / "index.client-gone.sqlite"
    backup = TMP / "index.client-acme.backup-1.sqlite"
    custom = TMP / "elsewhere" / "old-client.sqlite"  # custom db: only the registry knows it
    custom.parent.mkdir()
    for f in (orphan, backup, custom):
        f.touch()
    indexer.REGISTRY.write_text(json.dumps({"old": {"db": str(custom), "roots": []}}))
    try:
        assert indexer.orphan_client_indexes() == sorted([orphan, custom]), indexer.orphan_client_indexes()
        try:
            indexer.check_orphans()
            raise AssertionError("orphans must stop the build")
        except SystemExit:
            pass
    finally:
        for f in (orphan, backup, custom, indexer.REGISTRY):
            f.unlink()


def test_slug_cannot_escape_root():
    try:
        config._load_clients({"../evil": {}})
        raise AssertionError("path-like slug must be rejected")
    except SystemExit:
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    order = ["test_routing", "test_unknown_client_is_skipped_not_general", "test_no_active_client_reads_general_only",
             "test_active_client_sees_own_layer_never_other", "test_tool_cwd_argument_cannot_switch_client",
             "test_env_override_and_fail_closed", "test_note_changing_layer_leaves_old_index",
             "test_sessions_route_by_recorded_cwd", "test_slug_cannot_escape_root",
             "test_frontmatter_variants_fail_closed", "test_roots_and_db_validation",
             "test_case_insensitive_roots", "test_symlinked_glob_layer_move", "test_orphan_client_index_detected"]
    try:
        for name in order:
            globals()[name]()
            print(f"ok  {name}")
    finally:
        os.chdir(Path(__file__).resolve().parent)
        for c in CONNS.values():
            c.close()
        shutil.rmtree(TMP, ignore_errors=True)
    assert len(order) == len(tests), "update the ordered list"
    print(f"\n{len(order)} passed")
