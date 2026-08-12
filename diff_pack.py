#!/usr/bin/env python3
"""Change-scoped retrieval: given a git diff, return relevant context.

Input:  git diff output (stdin) OR --files <path1,path2> OR --symbols <sym1,sym2> OR --pr <N>
Output: Markdown bundle with modified symbols + 1-hop callers + tests + standards

Usage:
  git diff HEAD~1 | diff_pack.py --budget 3000
  diff_pack.py --pr 579 --budget 3000
  diff_pack.py --files src/commands/play.ts,src/utils/cache.ts "fix player crash"
  diff_pack.py --symbols playerFactory::createPlayer,playerFactory::cleanup --budget 2500
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import search, cwd_repo

# Unified diff hunk header pattern: @@ -L1,C1 +L2,C2 @@
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class FileChange(NamedTuple):
    path: str
    old_start: int
    new_start: int
    added_lines: set[int]


def parse_diff(diff_text: str) -> dict[str, FileChange]:
    """Parse unified diff output to extract file + changed line ranges.

    Returns: {file_path: FileChange}
    """
    changes: dict[str, FileChange] = {}
    current_file = None
    current_change = None
    line_num = 0

    for line in diff_text.splitlines():
        # Detect file headers (--- a/path or +++ b/path)
        if line.startswith("--- a/"):
            current_file = line[6:]
            current_change = None
        elif line.startswith("--- /dev/null"):
            # New file case
            current_file = None
            current_change = None
        elif line.startswith("+++ b/"):
            # File to be added or modified
            path = line[6:]
            current_change = FileChange(path=path, old_start=0, new_start=0, added_lines=set())
            changes[path] = current_change
        elif line.startswith("diff --git"):
            # Reset on new file in diff
            current_file = None
            current_change = None

        # Detect hunk headers to reset line numbering
        elif line.startswith("@@"):
            match = _HUNK_HEADER.match(line)
            if match and current_change:
                current_change = FileChange(
                    path=current_change.path,
                    old_start=int(match.group(1)),
                    new_start=int(match.group(2)),
                    added_lines=set(),
                )
                changes[current_change.path] = current_change
                line_num = int(match.group(2)) - 1

        # Track added lines (those starting with +, excluding +++b/...)
        elif line.startswith("+") and not line.startswith("+++"):
            if current_change:
                line_num += 1
                current_change.added_lines.add(line_num)
        # Other lines (context or deletions) still advance line counter
        elif line.startswith("-") and not line.startswith("---"):
            if current_change:
                line_num += 1
        else:
            if current_change and line and not line.startswith("\\"):
                line_num += 1

    return changes


def fetch_pr_diff(pr_number: int) -> str:
    """Fetch git diff for a GitHub PR using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "pr", "diff", str(pr_number)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error fetching PR {pr_number}: {e}", file=sys.stderr)
        return ""


def find_symbols_in_ranges(
    db_path: Path, file_path: str, line_ranges: set[int]
) -> list[str]:
    """Find symbol definitions that overlap with changed lines.

    Returns: list of "file:symbol" strings
    """
    if not line_ranges:
        return []

    conn = sqlite3.connect(db_path)
    # Find symbols where [start_line, end_line] overlaps any changed line
    min_line = min(line_ranges)
    max_line = max(line_ranges)

    symbols = []
    sql = """
        SELECT symbol_name FROM symbols_definitions
        WHERE file = ? AND start_line <= ? AND end_line >= ?
    """

    for line in sorted(line_ranges):
        cur = conn.execute(sql, (file_path, max_line, min_line))
        for (sym,) in cur:
            symbol_id = f"{file_path}::{sym}"
            if symbol_id not in symbols:
                symbols.append(symbol_id)

    conn.close()
    return symbols


def get_callers(db_path: Path, symbol_id: str, max_depth: int = 1) -> list[str]:
    """Fetch 1-hop callers of a symbol via call-graph."""
    callers = []
    if max_depth <= 0:
        return callers

    conn = sqlite3.connect(db_path)
    # symbol_id is "file::name"; look it up in symbols_called_by
    symbol_name = symbol_id.split("::")[-1] if "::" in symbol_id else symbol_id
    
    sql = """
        SELECT DISTINCT caller FROM symbols_called_by
        WHERE callee = ? OR callee LIKE ?
    """
    # Match both exact "file::name" and "name" patterns
    cur = conn.execute(sql, (symbol_id, f"%.{symbol_name}"))
    for (caller,) in cur:
        if caller not in callers:
            callers.append(caller)

    conn.close()
    return callers


def find_test_files(
    db_path: Path, modified_symbols: list[str]
) -> list[str]:
    """Find test files that import or reference modified symbols."""
    test_paths = set()
    conn = sqlite3.connect(db_path)

    for symbol_id in modified_symbols:
        symbol_name = symbol_id.split("::")[-1] if "::" in symbol_id else symbol_id
        # Look for test chunks that mention this symbol
        sql = """
            SELECT DISTINCT path FROM chunks
            WHERE (path LIKE '%.test.ts' OR path LIKE '%.spec.ts' OR
                   path LIKE '%_test.py' OR path LIKE '%.spec.tsx')
            AND (text LIKE ? OR text LIKE ?)
        """
        cur = conn.execute(sql, (f"%{symbol_name}%", f"%{symbol_id}%"))
        for (path,) in cur:
            test_paths.add(path)

    conn.close()
    return list(test_paths)


def chunks_for_files(
    file_paths: list[str], query: str, budget_tokens: int
) -> tuple[list[dict], int]:
    """Search for chunks in specific files, respecting budget."""
    chunks = []
    budget = budget_tokens

    for fpath in file_paths:
        # Search with file hint
        search_query = f"{query} {fpath}" if query else fpath
        results = search(search_query, top=3, cwd=None)
        results = [r for r in results if fpath in r["path"]]

        for r in results:
            snippet_size = min(900, budget // 4)  # Rough token to char conversion
            snippet = r["text"][:snippet_size]
            cost = max(1, len(snippet) // 4)
            if budget - cost < 0:
                break

            chunks.append(r)
            budget -= cost

        if budget <= 100:  # Stop when budget is low
            break

    return chunks, budget


def build_pack(
    changes: dict[str, FileChange],
    query: str = "",
    budget_tokens: int = 3000,
) -> str:
    """Build context pack from diff changes."""
    from config import DB as db_path
    if not db_path.exists():
        return "(error: rag-index database not found)"

    budget = budget_tokens
    out: list[str] = []

    if not changes:
        return "(no changes detected)"

    # 1. Identify modified symbols
    all_modified_symbols: list[str] = []
    for file_path, change in changes.items():
        symbols = find_symbols_in_ranges(db_path, file_path, change.added_lines)
        all_modified_symbols.extend(symbols)

    # 2. Identify 1-hop callers
    all_callers: list[str] = []
    for symbol in all_modified_symbols:
        callers = get_callers(db_path, symbol)
        all_callers.extend(callers)

    # 3. Find test files
    test_files = find_test_files(db_path, all_modified_symbols)

    # 4. Emit sections respecting budget
    def emit_section(header: str, file_list: list[str], search_query: str, per_chunk_cap: int) -> None:
        nonlocal budget
        if not file_list or budget <= 100:
            return
        out.append(f"## {header}\n")
        chunks, budget = chunks_for_files(file_list, search_query, budget)
        for r in chunks:
            snippet = r["text"][:per_chunk_cap]
            sym = f"::{r['symbol']}" if r.get("symbol") else ""
            repo = f"/{r['repo']}" if r.get("repo") else ""
            header_line = (
                f"### {r['source_type']}{repo}{sym} `{r['path']}:{r['start_line']}-{r['end_line']}`"
            )
            block = f"{header_line}\n```\n{snippet}\n```\n"
            out.append(block)

    # Emit modified symbols
    if all_modified_symbols and budget > 100:
        modified_files = list(set(f.split("::")[0] for f in all_modified_symbols))
        emit_section("Modified symbols", modified_files, query or "implementation", 900)

    # Emit callers
    if all_callers and budget > 100:
        caller_files = list(set(f.split("::")[0] for f in all_callers))
        emit_section("Callers (1-hop)", caller_files, query or "caller context", 800)

    # Emit tests
    if test_files and budget > 100:
        emit_section("Tests", test_files, query or "test", 700)

    # 5. Apply standards
    if query and budget > 100:
        out.append("## Applicable standards\n")
        try:
            std_chunks = search(query, top=2, scope_types=["standards"], cwd=None)
            for r in std_chunks:
                snippet = r["text"][:600]
                block = f"### `{r['path']}:{r['start_line']}-{r['end_line']}`\n```\n{snippet}\n```\n"
                cost = max(1, len(block) // 4)
                if budget - cost >= 0:
                    out.append(block)
                    budget -= cost
        except Exception:
            pass

    if not out:
        return "(no relevant context found)"

    remaining = max(0, budget)
    header = f"# Diff-scoped context pack\n_Budget: {budget_tokens} tokens · remaining ≈ {remaining}_\n"
    return header + "\n" + "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Change-scoped retrieval from git diff or file list"
    )
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--diff",
        action="store_true",
        help="read unified diff from stdin",
    )
    input_group.add_argument(
        "--pr",
        type=int,
        help="fetch diff for GitHub PR by number",
    )
    input_group.add_argument(
        "--files",
        help="comma-separated file paths",
    )
    input_group.add_argument(
        "--symbols",
        help="comma-separated symbol names (file::symbol format)",
    )

    ap.add_argument(
        "--query",
        default="",
        help="optional natural language context (for standards matching)",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=3000,
        help="token budget (chars/4)",
    )
    args = ap.parse_args()

    changes: dict[str, FileChange] = {}

    if args.diff:
        diff_text = sys.stdin.read()
        changes = parse_diff(diff_text)
    elif args.pr:
        diff_text = fetch_pr_diff(args.pr)
        if diff_text:
            changes = parse_diff(diff_text)
    elif args.files:
        # Treat --files as list of paths with all lines in range
        for fpath in args.files.split(","):
            fpath = fpath.strip()
            p = Path(fpath).expanduser().resolve()
            if p.exists():
                # All lines in file are "changed" for context purposes
                num_lines = len(p.read_text(errors="replace").splitlines())
                changes[str(p)] = FileChange(
                    path=str(p),
                    old_start=1,
                    new_start=1,
                    added_lines=set(range(1, num_lines + 1)),
                )
    elif args.symbols:
        # Direct symbol names: "file::symbol" format
        # Create fake FileChange entries to trigger symbol lookup
        for sym in args.symbols.split(","):
            sym = sym.strip()
            if "::" in sym:
                file_part, sym_part = sym.rsplit("::", 1)
                # For direct symbols, we assume lines 1-999 as range
                changes[file_part] = FileChange(
                    path=file_part,
                    old_start=1,
                    new_start=1,
                    added_lines=set(range(1, 1000)),
                )

    if not changes:
        print("(no changes detected)", file=sys.stderr)
        return 1

    pack_output = build_pack(changes, query=args.query, budget_tokens=args.budget)
    print(pack_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
