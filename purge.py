#!/usr/bin/env python3
"""Purge one client's knowledge from every shelfmark surface.

Dry run by default: prints what would go. `--apply` does it, and needs either
`--archive <age-recipient>` (the default path: nothing is lost by accident, the
client's index and query-log rows are kept in one age-encrypted tarball) or an
explicit `--no-archive`.

What goes, in order:
  1. archive (streamed into `age`, never written in plaintext)
  2. a tombstone, written before anything is deleted: once the client leaves
     sources.yaml, builds skip its files instead of routing them to general,
     even if this purge dies halfway
  3. the client's index file, its -wal/-shm, its backups and corrupt asides
  4. any residue of the client in the general index AND in general backups:
     notes under its roots, commits of its repos, sessions recorded in its
     roots; scrubbed with secure_delete + WAL checkpoint + VACUUM
  5. the client's rows in the query log (by recorded client, cwd or top path)
     and the derived weekly.md report
  6. verification: 0 residue rows, client files gone, and (with --lexicon) no
     lexicon term in the raw bytes of any remaining general/log file

Notes indexed from inside the client's roots with `client: none` (harvested
lessons) are never residue: the index records that decision per row, so they
survive even if the source file is already gone.

Source files under the client's roots are yours and are never touched: this
purges the index side. File-level only: file system snapshots, Time Machine or
Spotlight copies are out of scope.

A failure midway returns 1 with "partial"; re-running the same command
finishes the job (every step is idempotent).

Usage:
  shelfmark-purge acme                                  # dry run
  shelfmark-purge acme --lexicon acme-terms.txt         # dry run + canary scan
  shelfmark-purge acme --apply --archive age1... --lexicon acme-terms.txt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path

from config import CLIENTS, CURATED_REPOS, DB, QLOG, ROOT, client_db, client_for_path

BATCH = 500  # stay under SQLite's variable limit on old builds (999)


def backups_of(db: Path) -> list[Path]:
    return sorted(db.parent.glob(f"{db.stem}.backup-*.sqlite"))


def corrupt_asides_of(db: Path) -> list[Path]:
    return sorted(db.parent.glob(f"{db.stem}.corrupt-*"))


def with_sidecars(p: Path) -> list[Path]:
    return [p, *(Path(f"{p}{s}") for s in ("-wal", "-shm", "-journal"))]


def tombstone_path(slug: str) -> Path:
    return ROOT / f"index.client-{slug}.purged"


def _session_routes_to(path: str, slug: str) -> bool:
    from session_chunker import recorded_cwd

    src = Path(path)
    if not src.exists():
        return True  # transcript gone: nothing proves it is not the client's
    cwd = recorded_cwd(src.read_text(errors="replace").splitlines())
    return bool(cwd) and client_for_path(cwd) == slug


def residue_paths(db: Path, slug: str) -> list[str]:
    """Distinct chunk paths in a general index (or backup) that belong to slug."""
    import indexer

    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    try:
        has_layer = "layer" in {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
        rows = conn.execute(
            f"SELECT DISTINCT path, source_type, repo, {'layer' if has_layer else 'NULL'} FROM chunks"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    client_repos = {r.name for r in CURATED_REPOS if client_for_path(r) == slug}
    out = set()
    for path, stype, repo, layer in rows:
        if stype == "commit":
            if repo in client_repos:
                out.add(path)
            continue
        if stype == "session":
            if _session_routes_to(path, slug):
                out.add(path)
            continue
        if path.startswith("git:") or client_for_path(path) != slug:
            continue
        if layer == "none":
            continue  # harvested lesson, recorded at index time
        src = Path(path)
        if layer is None and src.exists():
            # Rows indexed before the layer column: re-derive from the source.
            if indexer.route(src, src.read_text(encoding="utf-8", errors="replace")) == DB:
                continue
        out.add(path)
    return sorted(out)


def qlog_rows(slug: str) -> list[int]:
    """rowids of query-log rows belonging to slug."""
    if not QLOG.exists():
        return []
    conn = sqlite3.connect(QLOG)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(queries)")}
        rows = conn.execute(
            f"SELECT rowid, cwd, top_path, {'client' if 'client' in cols else 'NULL'} FROM queries"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [
        rid for rid, cwd, top, client in rows
        if client == slug or (cwd and client_for_path(cwd) == slug) or (top and client_for_path(top) == slug)
    ]


def load_lexicon(path: Path) -> list[str]:
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            if len(t) < 3:
                raise SystemExit(f"shelfmark-purge: lexicon term {t!r} is too short to scan for (min 3 chars)")
            terms.append(t.casefold())
    return terms


def scan(files: list[Path], terms: list[str]) -> dict[str, list[str]]:
    """Raw-byte canary scan: catches text in free pages, WAL and backups, not just live rows."""
    hits: dict[str, list[str]] = {}
    for f in files:
        if not f.is_file():
            continue
        data = f.read_bytes().decode("utf-8", "ignore").casefold()
        found = [t for t in terms if t in data]
        if found:
            hits[str(f)] = found
    return hits


def scrub(db: Path, table: str, column: str, values: list) -> int:
    """Delete rows so their bytes do not survive in free pages or the WAL."""
    conn = sqlite3.connect(db, timeout=10)
    try:
        conn.execute("PRAGMA secure_delete=ON")
        n = 0
        for i in range(0, len(values), BATCH):
            part = values[i:i + BATCH]
            n += conn.execute(
                f"DELETE FROM {table} WHERE {column} IN ({','.join('?' * len(part))})", part
            ).rowcount
        conn.commit()
        conn.execute("VACUUM")
        busy, _log, _done = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy:
            # Old WAL frames may still hold the deleted rows in plaintext.
            raise RuntimeError(f"{db.name}: WAL checkpoint busy (another process has it open); close it and re-run")
        return n
    finally:
        conn.close()


def archive(slug: str, recipient: str, files: list[Path], qlog_rowids: list[int]) -> Path:
    if not shutil.which("age"):
        raise SystemExit("shelfmark-purge: `age` not found on PATH (needed for --archive; or pass --no-archive)")
    out_dir = ROOT / "archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"client-{slug}-{int(time.time())}.tar.gz.age"
    proc = subprocess.Popen(["age", "-r", recipient, "-o", str(out)], stdin=subprocess.PIPE)
    ok = True
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|gz") as tar:
            for f in files:
                if f.is_file():
                    tar.add(f, arcname=f.name)
            if qlog_rowids:
                conn = sqlite3.connect(QLOG)
                marks = ",".join("?" * len(qlog_rowids))
                cur = conn.execute(f"SELECT * FROM queries WHERE rowid IN ({marks})", qlog_rowids)
                cols = [d[0] for d in cur.description]
                payload = json.dumps([dict(zip(cols, r)) for r in cur.fetchall()], ensure_ascii=False).encode()
                conn.close()
                info = tarfile.TarInfo("queries.json")
                info.size = len(payload)
                tar.addfile(info, BytesIO(payload))
    except (BrokenPipeError, OSError):
        ok = False  # age died early (bad recipient, disk full)
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            ok = False
    if proc.wait() != 0 or not ok or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise SystemExit("shelfmark-purge: age failed; nothing was deleted")
    return out


def unlink_all(paths: list[Path]) -> None:
    for p in paths:
        for f in with_sidecars(p):
            f.unlink(missing_ok=True)


def apply(slug: str, client_files: list[Path], residue: dict[str, list[str]], qrows: list[int]) -> None:
    unlink_all(client_files)
    for db_path, paths in residue.items():
        db = Path(db_path)
        if db == DB:
            scrub(db, "chunks", "path", paths)
        else:
            unlink_all([db])  # a general backup holding client residue is dropped whole
    if residue:
        import indexer  # fresh clean snapshot of the scrubbed general index

        indexer.backup_db(DB)
    if qrows:
        scrub(QLOG, "queries", "rowid", qrows)
    (ROOT / "weekly.md").unlink(missing_ok=True)  # derived from the query log; report.py regenerates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Purge one client's knowledge from the index side.")
    ap.add_argument("client", help="configured client slug")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--archive", metavar="AGE_RECIPIENT", help="age recipient for the encrypted archive")
    ap.add_argument("--no-archive", action="store_true", help="delete without keeping an archive")
    ap.add_argument("--lexicon", type=Path, help="canary terms, one per line, scanned in the raw bytes of remaining files")
    args = ap.parse_args(argv)

    slug = args.client
    if slug not in CLIENTS:
        raise SystemExit(f"shelfmark-purge: {slug!r} is not a configured client (purge runs before removing it from sources.yaml)")
    if args.apply and not (args.archive or args.no_archive):
        raise SystemExit("shelfmark-purge: --apply needs --archive <age-recipient> or an explicit --no-archive")
    if args.archive and args.no_archive:
        raise SystemExit("shelfmark-purge: --archive and --no-archive are exclusive")
    terms = load_lexicon(args.lexicon) if args.lexicon else []

    cdb = client_db(slug)
    client_files = [f for f in [cdb, *backups_of(cdb), *corrupt_asides_of(cdb)] if f.exists()]
    general_copies = [*backups_of(DB), *corrupt_asides_of(DB)]
    residue = {str(db): residue_paths(db, slug) for db in [DB, *backups_of(DB)]}
    residue = {k: v for k, v in residue.items() if v}
    qrows = qlog_rows(slug)

    def kept_files() -> list[Path]:
        return [*with_sidecars(DB), *backups_of(DB), *corrupt_asides_of(DB), *with_sidecars(QLOG), ROOT / "weekly.md"]

    print(f"client {slug}: {len(client_files)} index file(s), "
          f"{sum(len(v) for v in residue.values())} residue path(s) in general/backups, "
          f"{len(qrows)} query-log row(s)")
    for f in client_files:
        print(f"  delete  {f}")
    for db, paths in residue.items():
        print(f"  scrub   {db}: {len(paths)} path(s)")
    for f in general_copies:
        if f.name.startswith(f"{DB.stem}.corrupt-"):
            print(f"  note    {f}: corrupt aside, only the canary scan can check it")
    if terms:
        for f, found in scan(kept_files(), terms).items():
            print(f"  canary  {f}: {', '.join(found)}")

    if not args.apply:
        print("dry run: nothing changed (add --apply)")
        return 0

    archived = archive(slug, args.archive, client_files, qrows) if args.archive else None
    if archived:
        print(f"archived: {archived}")
    # Tombstone before any deletion: inert while the slug is still configured,
    # and it keeps a half-finished purge from reopening the leak later.
    tombstone_path(slug).write_text(json.dumps({
        "client": slug,
        "roots": [str(r) for r in CLIENTS[slug]["roots"]],
        "purged_at": int(time.time()),
        "archive": str(archived) if archived else None,
    }, indent=2))
    try:
        apply(slug, client_files, residue, qrows)
    except (sqlite3.Error, OSError, RuntimeError) as e:
        print(f"PARTIAL: {e}\nre-run the same command to finish (every step is idempotent)", file=sys.stderr)
        return 1

    # Verify.
    problems = []
    problems += [f"still present: {f}" for f in client_files if f.exists()]
    for db in [DB, *backups_of(DB)]:
        left = residue_paths(db, slug)
        if left:
            problems.append(f"residue left in {db}: {len(left)} path(s)")
    if qlog_rows(slug):
        problems.append("query-log rows left")
    if terms:
        for f, found in scan(kept_files(), terms).items():
            problems.append(f"canary in {f}: {', '.join(found)}")
    if problems:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"purged {slug}: verified. Next: remove {slug!r} from sources.yaml together with its globs "
          f"(the tombstone keeps builds from routing its files into general).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
