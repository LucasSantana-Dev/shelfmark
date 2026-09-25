#!/usr/bin/env python3
"""Purge one client's knowledge from every shelfmark surface.

Dry run by default: prints what would go. `--apply` does it, and needs either
`--archive <age-recipient>` (the default path: nothing is lost by accident, the
client's index and query-log rows are kept in one age-encrypted tarball) or an
explicit `--no-archive`.

What goes, in order:
  1. archive (streamed into `age`, never written in plaintext)
  2. the client's index file, its -wal/-shm, and its backups
  3. any residue of the client in the general index AND in general backups
     (rows whose path routes to the client), with secure_delete + VACUUM
  4. the client's rows in the query log (by recorded client, cwd or top path)
  5. a tombstone, so a later build refuses to route the client's files into
     general if the client leaves sources.yaml but its globs stay
  6. verification: 0 residue rows, client files gone, and (with --lexicon) no
     lexicon term in the raw bytes of any remaining general/log file

Source files under the client's roots are yours and are never touched: this
purges the index side. Remove or archive them separately.

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

from config import CLIENTS, DB, QLOG, ROOT, client_db, client_for_path


def backups_of(db: Path) -> list[Path]:
    return sorted(db.parent.glob(f"{db.stem}.backup-*.sqlite"))


def with_sidecars(p: Path) -> list[Path]:
    return [p, *(Path(f"{p}{s}") for s in ("-wal", "-shm", "-journal"))]


def tombstone_path(slug: str) -> Path:
    return ROOT / f"index.client-{slug}.purged"


def residue_paths(db: Path, slug: str) -> list[str]:
    """Distinct chunk paths in db that route to slug (should be none in general)."""
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    try:
        paths = [r[0] for r in conn.execute("SELECT DISTINCT path FROM chunks")]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    import indexer

    out = []
    for p in paths:
        if p.startswith("git:") or client_for_path(p) != slug:
            continue
        src = Path(p)
        if src.exists():
            # A note under the client's root tagged `client: none` is a
            # harvested lesson that belongs in general: never residue.
            if indexer.route(src, src.read_text(encoding="utf-8", errors="replace")) == DB:
                continue
        out.append(p)
    return out


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
        if not f.exists():
            continue
        data = f.read_bytes().decode("utf-8", "ignore").casefold()
        found = [t for t in terms if t in data]
        if found:
            hits[str(f)] = found
    return hits


def scrub(db: Path, sql: str, params: list) -> int:
    """Delete rows so their bytes do not survive in free pages or the WAL."""
    conn = sqlite3.connect(db, timeout=10)
    try:
        conn.execute("PRAGMA secure_delete=ON")
        n = conn.execute(sql, params).rowcount
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return n
    finally:
        conn.close()


def delete_paths(db: Path, paths: list[str]) -> int:
    if not paths:
        return 0
    marks = ",".join("?" * len(paths))
    return scrub(db, f"DELETE FROM chunks WHERE path IN ({marks})", paths)


def archive(slug: str, recipient: str, files: list[Path], qlog_rowids: list[int]) -> Path:
    if not shutil.which("age"):
        raise SystemExit("shelfmark-purge: `age` not found on PATH (needed for --archive; or pass --no-archive)")
    out_dir = ROOT / "archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"client-{slug}-{int(time.time())}.tar.gz.age"
    proc = subprocess.Popen(["age", "-r", recipient, "-o", str(out)], stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|gz") as tar:
            for f in files:
                if f.exists():
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
    finally:
        proc.stdin.close()
    if proc.wait() != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise SystemExit("shelfmark-purge: age failed; nothing was deleted")
    return out


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
    client_files = [f for f in [*with_sidecars(cdb), *backups_of(cdb)] if f.exists()]
    general_backups = backups_of(DB)
    residue = {str(db): residue_paths(db, slug) for db in [DB, *general_backups]}
    residue = {k: v for k, v in residue.items() if v}
    qrows = qlog_rows(slug)
    kept_files = [*with_sidecars(DB), *general_backups, *with_sidecars(QLOG)]

    print(f"client {slug}: {len(client_files)} index file(s), "
          f"{sum(len(v) for v in residue.values())} residue path(s) in general/backups, "
          f"{len(qrows)} query-log row(s)")
    for f in client_files:
        print(f"  delete  {f}")
    for db, paths in residue.items():
        print(f"  scrub   {db}: {len(paths)} path(s)")
    if terms:
        pre = scan(kept_files, terms)
        for f, found in pre.items():
            print(f"  canary  {f}: {', '.join(found)}")

    if not args.apply:
        print("dry run: nothing changed (add --apply)")
        return 0

    archived = archive(slug, args.archive, client_files, qrows) if args.archive else None
    if archived:
        print(f"archived: {archived}")

    for f in client_files:
        f.unlink(missing_ok=True)
    for db_path, paths in residue.items():
        db = Path(db_path)
        if db == DB:
            delete_paths(db, paths)
        else:
            db.unlink()  # a general backup holding client residue is dropped whole
    if residue:
        import indexer  # fresh clean snapshot of the scrubbed general index

        indexer.backup_db(DB)
    if qrows:
        marks = ",".join("?" * len(qrows))
        scrub(QLOG, f"DELETE FROM queries WHERE rowid IN ({marks})", qrows)
    tombstone_path(slug).write_text(json.dumps({
        "client": slug,
        "roots": [str(r) for r in CLIENTS[slug]["roots"]],
        "purged_at": int(time.time()),
        "archive": str(archived) if archived else None,
    }, indent=2))

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
        for f, found in scan([*with_sidecars(DB), *backups_of(DB), *with_sidecars(QLOG)], terms).items():
            problems.append(f"canary in {f}: {', '.join(found)}")
    if problems:
        print("VERIFY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"purged {slug}: verified. Next: remove {slug!r} from sources.yaml together with its globs "
          f"(the tombstone blocks builds that would route its files into general).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
