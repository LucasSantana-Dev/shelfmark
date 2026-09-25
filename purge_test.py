#!/usr/bin/env python3
"""Tests for purge.py: one client's knowledge leaves every index-side surface,
harvested lessons (`client: none`) survive, and the archive is recoverable."""

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path

import numpy as np

TMP = Path(tempfile.mkdtemp(prefix="shelfmark-purge-")).resolve()
os.environ["RAG_HOME"] = str(TMP)
os.environ["RAG_SOURCES"] = str(TMP / "sources.yaml")
os.environ.pop("RAG_CLIENT", None)

NOTES, ACME, BETA, GAMMA = TMP / "notes", TMP / "acme", TMP / "beta", TMP / "gamma"
SESS = TMP / "sessions"
for d in (NOTES, ACME / "acme-app", BETA, GAMMA, SESS):
    d.mkdir(parents=True)
(TMP / "sources.yaml").write_text(textwrap.dedent(f"""\
    repos: ["{ACME}/acme-app"]
    sources:
      - {{type: memory, glob: "{NOTES}/**/*.md"}}
      - {{type: memory, glob: "{ACME}/**/*.md"}}
      - {{type: memory, glob: "{BETA}/**/*.md"}}
      - {{type: memory, glob: "{GAMMA}/**/*.md"}}
    clients:
      acme:
        roots: ["{ACME}"]
      beta:
        roots: ["{BETA}"]
      gamma:
        roots: ["{GAMMA}"]
    """))
(NOTES / "general.md").write_text("# general\n\nrollback lesson kept forever: always keep a tested rollback path\n")
(ACME / "rule.md").write_text("# rule\n\nacme walrus billing rule: invoices are charged on the fifth of each month\n")
(ACME / "lesson.md").write_text("---\nclient: none\n---\n# lesson\n\nidempotent retry lesson learned at acme\n")
(GAMMA / "g.md").write_text("# g\n\ngamma client notes long enough to be indexed as a chunk\n")
(BETA / "b.md").write_text("# b\n\nbeta zebra pricing tiers are confidential and change every quarter\n")
LEXICON = TMP / "acme-terms.txt"
LEXICON.write_text("# acme canaries\nwalrus\n")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import indexer  # noqa: E402
import purge  # noqa: E402


class ZeroModel:
    def encode(self, texts, **_kw):
        return np.zeros((len(texts), config.DIM), dtype=np.float32)


def build() -> None:
    conns = {}

    def conn_for(db):
        if db not in conns:
            conns[db] = indexer.connect(db)
        return conns[db]

    indexer.index_files(conn_for, ZeroModel(), indexer.iter_md_sources(), [])
    for c in conns.values():
        c.close()
    for db in conns:
        indexer.backup_db(db)


def paths(db: Path) -> set[str]:
    if not db.exists():
        return set()
    return {Path(r[0]).name for r in sqlite3.connect(db).execute("SELECT DISTINCT path FROM chunks")}


def digest(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.name.encode() + (f.read_bytes() if f.is_file() else b""))
    return h.hexdigest()


build()
GEN, ACME_DB, BETA_DB = config.DB, config.client_db("acme"), config.client_db("beta")

# Simulate a pre-layers index: acme knowledge sitting in general (a note, a
# commit of its repo, a session recorded in its root) and in a general backup.
(SESS / "acme-session.jsonl").write_text('{"cwd": "%s", "message": {"content": "walrus chat"}}\n' % ACME)
(SESS / "general-session.jsonl").write_text('{"cwd": "%s", "message": {"content": "rollback chat"}}\n' % TMP)
_zero = np.zeros(config.DIM, dtype=np.float32).tobytes()
_conn = sqlite3.connect(GEN)
_conn.executemany(
    "INSERT INTO chunks (source_type, repo, language, symbol, path, start_line, end_line, text, file_sha, mtime, embedding) "
    "VALUES (?, ?, NULL, NULL, ?, 1, 2, ?, 'x', 0, ?)",
    [
        ("memory", None, str(ACME / "rule.md"), "acme walrus billing rule leaked", _zero),
        ("commit", "acme-app", "git:acme-app@abc1234", "fix walrus invoice rounding", _zero),
        ("session", None, str(SESS / "acme-session.jsonl"), "walrus chat", _zero),
        ("session", None, str(SESS / "general-session.jsonl"), "rollback chat kept", _zero),
    ],
)
_conn.commit()
_conn.close()
shutil.copyfile(GEN, TMP / "index.backup-1.sqlite")

# Query log: one row by recorded client, one by cwd, one unrelated.
_q = sqlite3.connect(config.QLOG)
_q.execute("CREATE TABLE queries (ts REAL, cwd TEXT, query TEXT, scope_types TEXT, scope_repos TEXT, "
           "rerank INTEGER, top_score REAL, top_path TEXT, n_results INTEGER, client TEXT)")
_q.executemany("INSERT INTO queries VALUES (0,?,?,'','',0,0,?,1,?)", [
    (str(TMP), "walrus billing?", "", "acme"),
    (str(ACME), "when is the fifth charge", "", None),
    (str(TMP), "rollback lesson", str(NOTES / "general.md"), None),
])
_q.commit()
_q.close()


def test_dry_run_changes_nothing():
    before = digest(TMP.glob("*.sqlite*"))
    assert purge.main(["acme", "--lexicon", str(LEXICON)]) == 0
    assert digest(TMP.glob("*.sqlite*")) == before


def test_apply_requires_archive_decision():
    try:
        purge.main(["acme", "--apply"])
        raise AssertionError("--apply without --archive/--no-archive must refuse")
    except SystemExit:
        pass


def test_purge_acme():
    assert "rule.md" in paths(GEN)  # the planted residue
    (ACME / "lesson.md").unlink()  # the lesson's source is gone; the index remembers it was general
    (TMP / "index.client-acme.corrupt-1").write_bytes(b"walrus corrupt copy")
    (TMP / "weekly.md").write_text("top query: walrus billing?\n")
    assert purge.main(["acme", "--apply", "--no-archive", "--lexicon", str(LEXICON)]) == 0
    assert not ACME_DB.exists()
    assert not list(TMP.glob("index.client-acme.backup-*"))
    assert "rule.md" not in paths(GEN)
    assert "lesson.md" in paths(GEN), "harvested client: none lesson must survive the purge"
    assert "acme-app@abc1234" not in paths(GEN), "commit of the client's repo is residue"
    assert "acme-session.jsonl" not in paths(GEN), "session recorded in the client's root is residue"
    assert "general-session.jsonl" in paths(GEN), "unrelated session untouched"
    assert not (TMP / "index.client-acme.corrupt-1").exists()
    assert not (TMP / "weekly.md").exists()
    assert "general.md" in paths(GEN)
    assert not (TMP / "index.backup-1.sqlite").exists(), "general backup holding residue must go"
    assert list(TMP.glob("index.backup-*.sqlite")), "a fresh clean general backup must exist"
    assert paths(BETA_DB) == {"b.md"}, "other clients untouched"
    left = [r[0] for r in sqlite3.connect(config.QLOG).execute("SELECT query FROM queries")]
    assert left == ["rollback lesson"], left
    assert purge.tombstone_path("acme").exists()
    for f in [*TMP.glob("index*.sqlite*"), *TMP.glob("queries.sqlite*")]:
        if "client-beta" in f.name:
            continue
        assert b"walrus" not in f.read_bytes().lower(), f"canary bytes survive in {f.name}"


def test_tombstone_blocks_general_routing_once_client_removed():
    spec = config.CLIENTS.pop("acme")
    try:
        assert indexer.route(ACME / "rule.md", (ACME / "rule.md").read_text()) is None
        assert indexer.route(NOTES / "general.md", "") == GEN
    finally:
        config.CLIENTS["acme"] = spec


def test_partial_failure_then_rerun():
    real = purge.scrub
    calls = []

    def boom(*a, **k):
        calls.append(a)
        raise RuntimeError("simulated: database is locked")

    # Plant gamma residue in general so apply() has a scrub to fail on.
    conn = sqlite3.connect(GEN)
    conn.execute(
        "INSERT INTO chunks (source_type, repo, language, symbol, path, start_line, end_line, text, file_sha, mtime, embedding) "
        "VALUES ('memory', NULL, NULL, NULL, ?, 1, 2, 'gamma leaked', 'x', 0, ?)",
        (str(GAMMA / "g.md"), np.zeros(config.DIM, dtype=np.float32).tobytes()),
    )
    conn.commit()
    conn.close()
    purge.scrub = boom
    try:
        assert purge.main(["gamma", "--apply", "--no-archive"]) == 1
    finally:
        purge.scrub = real
    assert calls, "the simulated failure must have been hit"
    assert purge.tombstone_path("gamma").exists(), "tombstone must exist before anything is deleted"
    assert purge.main(["gamma", "--apply", "--no-archive"]) == 0
    assert "g.md" not in paths(GEN)


def test_archive_is_recoverable():
    if not (shutil.which("age") and shutil.which("age-keygen")):
        print("    (age not installed: archive test skipped)")
        return
    key = TMP / "key.txt"
    subprocess.run(["age-keygen", "-o", str(key)], check=True, capture_output=True)
    recipient = next(l.split(": ", 1)[1] for l in key.read_text().splitlines() if l.startswith("# public key: "))
    assert purge.main(["beta", "--apply", "--archive", recipient]) == 0
    assert not BETA_DB.exists()
    (archive,) = (TMP / "archive").glob("client-beta-*.tar.gz.age")
    plain = TMP / "beta.tar.gz"
    subprocess.run(["age", "-d", "-i", str(key), "-o", str(plain), str(archive)], check=True)
    with tarfile.open(plain) as tar:
        names = tar.getnames()
    assert "index.client-beta.sqlite" in names, names


if __name__ == "__main__":
    order = ["test_dry_run_changes_nothing", "test_apply_requires_archive_decision", "test_purge_acme",
             "test_tombstone_blocks_general_routing_once_client_removed",
             "test_partial_failure_then_rerun", "test_archive_is_recoverable"]
    assert len(order) == len([k for k in globals() if k.startswith("test_")]), "update the ordered list"
    try:
        for name in order:
            globals()[name]()
            print(f"ok  {name}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(order)} passed")
