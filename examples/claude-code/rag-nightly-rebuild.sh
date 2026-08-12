#!/usr/bin/env bash
# Nightly full index rebuild + eval gate. Schedule via launchd/cron at e.g. 02:00.
# Optional webhook alert on regression: set RAG_ALERT_WEBHOOK (Discord-style JSON embed).
set -uo pipefail

SHELFMARK_DIR="${SHELFMARK_DIR:-$HOME/shelfmark}"
RAG_HOME="${RAG_HOME:-$HOME/.shelfmark}"
PY="${SHELFMARK_PY:-$SHELFMARK_DIR/venv/bin/python3}"
LOG="$RAG_HOME/nightly-rebuild.log"
BASELINE="$SHELFMARK_DIR/eval/baseline-golden.json"

log() { echo "[$(date '+%Y-%m-%dT%H:%M:%S')] $*" | tee -a "$LOG"; }

notify() {
    local title="$1" msg="$2" color="${3:-3066993}"
    [ -z "${RAG_ALERT_WEBHOOK:-}" ] && return 0
    curl -sf -H "Content-Type: application/json" -X POST "$RAG_ALERT_WEBHOOK" \
        -d "{\"embeds\":[{\"title\":\"$title\",\"description\":\"$msg\",\"color\":$color}]}" \
        >> "$LOG" 2>&1 || true
}

[ -x "$PY" ] || { log "ERROR: python not at $PY"; exit 1; }

log "=== nightly full rebuild start ==="

# 1. Full rebuild (not incremental)
if ! "$PY" "$SHELFMARK_DIR/build.py" >> "$LOG" 2>&1; then
    log "ERROR: build.py exited non-zero"
    notify "shelfmark rebuild FAILED" "build.py exited non-zero — check nightly-rebuild.log" "15158332"
    exit 1
fi
log "build.py complete"

# 2. Eval gate (requires an eval dataset + frozen baseline for YOUR corpus)
if [ -f "$BASELINE" ]; then
    LABEL="nightly-$(date '+%Y%m%d-%H%M')"
    if ! RAG_QLOG=off "$SHELFMARK_DIR/eval/check.sh" "$LABEL" >> "$LOG" 2>&1; then
        log "REGRESSION: eval gate failed vs baseline-golden"
        notify "shelfmark nightly: REGRESSION" "hit@5 regression vs baseline — check nightly-rebuild.log" "15158332"
        exit 1
    fi
    log "eval gate passed"
fi

notify "shelfmark nightly: OK" "Full rebuild complete." "3066993"
log "=== done ==="
