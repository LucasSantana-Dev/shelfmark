#!/usr/bin/env python3
"""One-off + reusable population of the call-graph tables from indexed Python files.

No re-embed: iterates the indexed python code files and fills symbols_definitions /
symbols_calls_to / symbols_imports via ast (symbols.py). Refines the (empty) calls_to schema
to add a `file` column (needed to map a call back to its chunk) and makes called_by a VIEW.
Idempotent: full repopulate. Run after index changes, or call populate_file() incrementally.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import symbols

from config import DB as _DB

DB = str(_DB)


def _refine_schema(cur):
    # Type-aware drop (called_by may currently be a TABLE or, on re-run, a VIEW).
    for name in ("symbols_called_by", "symbols_calls_to"):
        row = cur.execute("SELECT type FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if row:
            cur.execute(f"DROP {'VIEW' if row[0] == 'view' else 'TABLE'} {name}")
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS symbols_definitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT NOT NULL, symbol_name TEXT NOT NULL,
            language TEXT NOT NULL, start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            UNIQUE(file, symbol_name)
        );
        CREATE INDEX IF NOT EXISTS symbols_def_file ON symbols_definitions(file);
        CREATE TABLE IF NOT EXISTS symbols_imports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, importing_file TEXT NOT NULL,
            imported_module TEXT NOT NULL, imported_name TEXT NOT NULL,
            UNIQUE(importing_file, imported_module, imported_name)
        );
        CREATE INDEX IF NOT EXISTS symbols_imported_by ON symbols_imports(importing_file);
        CREATE TABLE symbols_calls_to (
            file TEXT NOT NULL, caller TEXT NOT NULL, callee TEXT NOT NULL,
            line_in_caller INTEGER, UNIQUE(file, caller, callee)
        );
        CREATE INDEX idx_calls_callee ON symbols_calls_to(callee);
        CREATE INDEX idx_calls_file   ON symbols_calls_to(file);
        CREATE VIEW symbols_called_by AS
            SELECT callee, caller, file, line_in_caller FROM symbols_calls_to;
        """
    )


def populate_file(cur, path: str, text: str, language: str = "python"):
    """Idempotently (re)populate one file's symbol rows. Incremental-safe."""
    d, c, i = symbols.extract(text, language)
    cur.execute("DELETE FROM symbols_definitions WHERE file = ?", (path,))
    cur.execute("DELETE FROM symbols_calls_to    WHERE file = ?", (path,))
    cur.execute("DELETE FROM symbols_imports     WHERE importing_file = ?", (path,))
    cur.executemany("INSERT OR IGNORE INTO symbols_definitions(file,symbol_name,language,start_line,end_line) VALUES(?,?,?,?,?)",
                    [(path, n, l, s, e) for (n, l, s, e) in d])
    cur.executemany("INSERT OR IGNORE INTO symbols_calls_to(file,caller,callee,line_in_caller) VALUES(?,?,?,?)",
                    [(path, ca, ce, ln) for (ca, ce, ln) in c])
    cur.executemany("INSERT OR IGNORE INTO symbols_imports(importing_file,imported_module,imported_name) VALUES(?,?,?)",
                    [(path, m, n) for (m, n) in i])
    return len(d), len(c), len(i)


def main():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA busy_timeout=10000")
    cur = con.cursor()
    _refine_schema(cur)
    cur.execute("DELETE FROM symbols_definitions")
    cur.execute("DELETE FROM symbols_imports")
    files = [r[0] for r in cur.execute(
        "SELECT DISTINCT path FROM chunks WHERE language='python' AND source_type IN ('code','workstation-code') "
        "AND path NOT LIKE '%/site-packages/%' AND path NOT LIKE '%/.venv%' AND path NOT LIKE '%/_vendor/%'")]
    skip = 0
    for f in files:
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            skip += 1
            continue
        populate_file(cur, f, text, "python")
    con.commit()
    print(f"files={len(files)} unreadable={skip}")
    for t in ("symbols_definitions", "symbols_calls_to", "symbols_imports"):
        print(f"  {t}: {cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows")
    print(f"  symbols_called_by (view): {cur.execute('SELECT COUNT(*) FROM symbols_called_by').fetchone()[0]} rows")
    con.close()


if __name__ == "__main__":
    main()
