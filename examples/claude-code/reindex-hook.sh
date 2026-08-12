#!/usr/bin/env bash
# PostToolUse hook: if the just-written file lives in a tracked source dir,
# reindex it incrementally. Silent on non-matches. Never blocks or errors out.
#
# EDIT the case patterns below to match the globs in your sources.yaml.
#
# Install (~/.claude/settings.json):
#   "hooks": { "PostToolUse": [ { "matcher": "Write|Edit", "hooks": [ { "type":
#     "command", "command": "bash /path/to/reindex-hook.sh" } ] } ] }
set -u
SHELFMARK_DIR="${SHELFMARK_DIR:-$HOME/shelfmark}"
RAG_PY="${SHELFMARK_PY:-$SHELFMARK_DIR/venv/bin/python}"
LOG="${RAG_HOME:-$HOME/.shelfmark}/hook.log"
# Cap the log so per-reindex model-load output can't grow it unbounded.
[ -f "$LOG" ] && [ "$(wc -l < "$LOG" 2>/dev/null || echo 0)" -gt 2000 ] && { tail -n 500 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"; }
INPUT=$(cat 2>/dev/null || true)

FILE=$(printf '%s' "$INPUT" | python3 -c 'import json,sys
try:
  d=json.loads(sys.stdin.read() or "{}")
  p=d.get("tool_input",{}).get("file_path") or d.get("file_path") or ""
  print(p)
except Exception:
  print("")
' 2>/dev/null)

[ -z "$FILE" ] && exit 0

# EDIT ME: patterns should mirror your sources.yaml globs.
case "$FILE" in
  "$HOME"/notes/memory/*.md|\
  "$HOME"/notes/plans/*.md|\
  "$HOME"/notes/standards/*.md)
    TRANSFORMERS_VERBOSITY=error HF_HUB_VERBOSITY=error \
    "$RAG_PY" "$SHELFMARK_DIR/indexer.py" --incremental "$FILE" \
      >>"$LOG" 2>&1 &
    ;;
esac
exit 0
