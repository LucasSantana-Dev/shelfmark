#!/usr/bin/env bash
# SessionStart hook: detect drifted (modified-on-disk-since-indexed) files
# and kick a background incremental reindex. Cap N to avoid burning CPU
# on every session start. Silent unless something runs.
#
# Closes the report→fix loop: files modified outside the agent harness
# (manual vim, IDE, git pull) would otherwise drift until the next full
# rebuild — this reindexes them incrementally on the next session.
set -u

SHELFMARK_DIR="${SHELFMARK_DIR:-$HOME/shelfmark}"
RAG_HOME="${RAG_HOME:-$HOME/.shelfmark}"
PY="${SHELFMARK_PY:-$SHELFMARK_DIR/venv/bin/python3}"
THRESHOLD="${RAG_DRIFT_THRESHOLD:-3}"   # only reindex if ≥ N files drift
MAX_FILES="${RAG_DRIFT_MAX:-25}"        # cap per session
LOG="$RAG_HOME/drift-reindex.log"

[ -x "$PY" ] || exit 0
[ -f "$RAG_HOME/index.sqlite" ] || exit 0

# Throttle: don't run more than once per 6h
MARKER="$RAG_HOME/.last-drift-reindex"
if [ -f "$MARKER" ]; then
  age=$(( $(date +%s) - $(stat -f %m "$MARKER" 2>/dev/null || stat -c %Y "$MARKER" 2>/dev/null || echo 0) ))
  [ "$age" -lt 21600 ] && exit 0
fi

# Pull the modified list from report.stale_chunks() — reuse existing logic
DRIFT_FILES=$("$PY" - "$SHELFMARK_DIR" <<'PY' 2>/dev/null
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from report import stale_chunks
for kind, path in stale_chunks():
    if kind == "modified":
        print(path)
PY
)

[ -z "$DRIFT_FILES" ] && exit 0
COUNT=$(printf "%s\n" "$DRIFT_FILES" | wc -l | tr -d ' ')
[ "$COUNT" -lt "$THRESHOLD" ] && exit 0

# Take the first MAX_FILES, kick incremental reindex in background
PICKED=$(printf "%s\n" "$DRIFT_FILES" | head -"$MAX_FILES")
{
  echo "[$(date '+%Y-%m-%dT%H:%M:%S')] drift=$COUNT picked=$(echo "$PICKED" | wc -l | tr -d ' ') threshold=$THRESHOLD"
  # shellcheck disable=SC2086
  echo "$PICKED" | xargs "$PY" "$SHELFMARK_DIR/indexer.py" --incremental
  # Eval gate: after a drift reindex, verify retrieval quality against the
  # golden baseline (if one exists for your corpus).
  if [ -x "$SHELFMARK_DIR/eval/check.sh" ] && [ -f "$SHELFMARK_DIR/eval/baseline-golden.json" ]; then
    RAG_QLOG=off "$SHELFMARK_DIR/eval/check.sh" "drift-$(date '+%Y%m%d-%H%M')" || \
      echo "[$(date '+%Y-%m-%dT%H:%M:%S')] EVAL GATE REGRESSION after drift reindex" >> "$LOG"
  fi
  touch "$MARKER"
} >>"$LOG" 2>&1 &
exit 0
