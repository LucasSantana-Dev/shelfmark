#!/usr/bin/env bash
# UserPromptSubmit hook: inject top sqlrecall hits for the user's prompt.
# Silent on low-confidence / short / long prompts. Never blocks.
#
# Install (~/.claude/settings.json):
#   "hooks": { "UserPromptSubmit": [ { "hooks": [ { "type": "command",
#     "command": "bash /path/to/autorecall-hook.sh" } ] } ] }
#
# Env dials:
#   SQLRECALL_DIR                 → repo checkout (default ~/sqlrecall)
#   SQLRECALL_PY                  → python with deps (default $SQLRECALL_DIR/venv/bin/python)
#   CLAUDE_RAG_AUTORECALL=off     → disabled entirely
#   CLAUDE_RAG_AUTORECALL=quiet   → cosine threshold 0.55 (default)
#   CLAUDE_RAG_AUTORECALL=loud    → cosine threshold 0.40
set -u

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Fail-fast: if the engine moved or is broken, exit immediately (one stderr
# line) instead of failing inside the hook timeout on every prompt.
SQLRECALL_DIR="${SQLRECALL_DIR:-${SHELFMARK_DIR:-$HOME/sqlrecall}}"
# Checkouts made before the rename live at ~/shelfmark.
[ -d "$SQLRECALL_DIR" ] || [ ! -d "$HOME/shelfmark" ] || SQLRECALL_DIR="$HOME/shelfmark"
RAG_PY="${SQLRECALL_PY:-${SHELFMARK_PY:-$SQLRECALL_DIR/venv/bin/python}}"
RAG_QUERY="$SQLRECALL_DIR/query.py"
if [ ! -x "$RAG_PY" ] || [ ! -f "$RAG_QUERY" ]; then
  echo "autorecall: sqlrecall not found at $SQLRECALL_DIR (moved?) — skipping recall" >&2
  exit 0
fi

MODE="${CLAUDE_RAG_AUTORECALL:-quiet}"
[ "$MODE" = "off" ] && exit 0

INPUT=$(cat 2>/dev/null || true)

PROMPT=$(printf '%s' "$INPUT" | python3 -c 'import json,sys
try:
  d=json.loads(sys.stdin.read() or "{}")
  print(d.get("prompt","") or d.get("user_prompt","") or "")
except Exception:
  print("")
' 2>/dev/null)

LEN=${#PROMPT}
[ "$LEN" -lt 15 ] && exit 0
[ "$LEN" -gt 2000 ] && exit 0

# Skip harness task-notifications (autonomous-tick noise, not a user question).
if printf '%s' "$PROMPT" | grep -qE '^[[:space:]]*<task-notification'; then
  exit 0
fi

THRESHOLD="0.55"
[ "$MODE" = "loud" ] && THRESHOLD="0.40"

RAG_LINES=$("$RAG_PY" "$RAG_QUERY" \
	--top 2 --format json --scope-repo all --fast "$PROMPT" 2>/dev/null |
	python3 -c "
import json, sys
data = json.loads(sys.stdin.read() or '[]')
threshold = float('$THRESHOLD')
keep = [r for r in data if r.get('cos', 0) >= threshold]
if not keep:
    sys.exit(0)
lines = []
for r in keep:
    tag = r['source_type']
    if r.get('repo'): tag += '/' + r['repo']
    if r.get('symbol'): tag += '::' + r['symbol']
    text = (r['text'] or '')[:300].replace('\n', ' ')
    lines.append(f\"- [{tag}] {r['path']}:{r['start_line']}-{r['end_line']} (cos={r['cos']:.2f}) — {text}\")
if lines:
    print('<!-- RAG (cos>=' + str(threshold) + ') -->')
    print('\n'.join(lines))
" 2>/dev/null)

[ -z "$RAG_LINES" ] && exit 0

RAG_COUNT=$(printf '%s\n' "$RAG_LINES" | grep -c '^- \[' || true)

printf '<!-- Auto-recall (%d RAG hits). Treat as hints. -->\n' "$RAG_COUNT"
printf '%s\n' "$RAG_LINES"

exit 0
