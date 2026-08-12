#!/usr/bin/env python3
"""Tests for diff_pack.py change-scoped retrieval."""

import sqlite3
import tempfile
from pathlib import Path

from diff_pack import parse_diff, find_symbols_in_ranges, FileChange


def test_parse_diff_simple():
    """Test parsing a simple unified diff."""
    diff = """diff --git a/src/commands/play.ts b/src/commands/play.ts
index abc1234..def5678 100644
--- a/src/commands/play.ts
+++ b/src/commands/play.ts
@@ -12,7 +12,9 @@ import { playerFactory } from '../utils/playerFactory';
 
 export async function handlePlay(context: CommandContext) {
   const player = playerFactory.createPlayer();
-  await player.play(context.url);
+  const duration = await player.getDuration(context.url);
+  if (duration < 10000) return;
+  await player.play(context.url);
   return { status: 'ok' };
 }
"""
    changes = parse_diff(diff)
    assert "src/commands/play.ts" in changes, "File should be in changes"
    change = changes["src/commands/play.ts"]
    assert isinstance(change.added_lines, set)
    # We expect at least 2 added lines from the diff
    assert len(change.added_lines) > 0, f"Expected added lines, got {change.added_lines}"
    print(f"  Added lines: {sorted(change.added_lines)}")


def test_parse_diff_new_file():
    """Test parsing diff for a new file."""
    diff = """diff --git a/new.ts b/new.ts
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/new.ts
@@ -0,0 +1,5 @@
+export function greet(name: string) {
+  return `Hello, ${name}!`;
+}
+
+greet('world');
"""
    changes = parse_diff(diff)
    assert "new.ts" in changes, "new.ts should be in changes"
    print(f"  New file added_lines: {len(changes['new.ts'].added_lines)}")


def test_parse_diff_empty():
    """Test parsing empty diff."""
    changes = parse_diff("")
    assert changes == {}, "Empty diff should return empty dict"


def test_find_symbols_in_ranges():
    """Test symbol lookup from line ranges."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = Path(tmp.name)
    
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE symbols_definitions (
                id INTEGER PRIMARY KEY,
                file TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                language TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL
            )
        """)
        
        conn.execute(
            "INSERT INTO symbols_definitions VALUES (1, 'test.ts', 'myFunc', 'typescript', 10, 20)"
        )
        conn.commit()
        conn.close()
        
        # Test: line 15 should match myFunc (10-20)
        symbols = find_symbols_in_ranges(db_path, "test.ts", {15})
        assert "test.ts::myFunc" in symbols, f"Expected myFunc in {symbols}"
        
        # Test: empty range
        symbols = find_symbols_in_ranges(db_path, "test.ts", set())
        assert symbols == [], f"Expected empty list, got {symbols}"
        
        print(f"  Symbols found correctly")
    
    finally:
        db_path.unlink()


if __name__ == "__main__":
    test_parse_diff_simple()
    print("✓ test_parse_diff_simple")
    
    test_parse_diff_new_file()
    print("✓ test_parse_diff_new_file")
    
    test_parse_diff_empty()
    print("✓ test_parse_diff_empty")
    
    test_find_symbols_in_ranges()
    print("✓ test_find_symbols_in_ranges")
    
    print("\nAll tests passed!")
