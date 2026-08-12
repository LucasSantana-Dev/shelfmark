#!/usr/bin/env bash
# SessionStart hook: refresh $RAG_HOME/weekly.md once per 24h.
# Requires query telemetry (RAG_QLOG=on). Non-blocking, backgrounded.
set -u
SHELFMARK_DIR="${SHELFMARK_DIR:-$HOME/shelfmark}"
RAG_HOME="${RAG_HOME:-$HOME/.shelfmark}"
PY="${SHELFMARK_PY:-$SHELFMARK_DIR/venv/bin/python}"
REPORT="$RAG_HOME/weekly.md"
MAX_AGE=86400  # 24h

if [ -f "$REPORT" ]; then
  age=$(( $(date +%s) - $(stat -f %m "$REPORT" 2>/dev/null || stat -c %Y "$REPORT" 2>/dev/null || echo 0) ))
  [ "$age" -lt "$MAX_AGE" ] && exit 0
fi

"$PY" "$SHELFMARK_DIR/report.py" >>"$RAG_HOME/report.log" 2>&1 &
exit 0
