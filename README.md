# shelfmark

Local hybrid retrieval for agent memory and codebases. One SQLite file, no
services, no cloud. BM25 + dense embeddings fused with Reciprocal Rank Fusion,
an optional cross-encoder reranker, and an MCP server so coding agents
(Claude Code, Codex, anything MCP) can recall your notes, decisions, docs, and
code instead of guessing.

A shelfmark is the code a librarian writes on a book so it can be found again.
That is the whole product: before your agent answers, the librarian fetches the
handful of notes and code chunks most likely to matter and puts them on the
desk.

## Why this exists

Agent harnesses accumulate knowledge: memory notes, ADRs, plans, standards,
handoffs, and the code itself. The useful answer to most questions is already
written down somewhere. shelfmark indexes all of it into one local SQLite file
and answers "what did we decide about X / where did we handle Y" in a single
query, with `path:line` citations.

Design choices that fell out of a year of measured iteration (see
`eval/holdout-policy.md` and inline rationale comments):

- **Hybrid by default.** BM25 catches identifiers and exact terms; embeddings
  catch paraphrase. RRF fusion beats either alone on mixed corpora.
- **Code-aware tokenization.** camelCase/snake_case sub-tokens on both corpus
  and query sides ("create player" matches `createPlayer`). Measured +2.8pp
  code / +3.2pp overall.
- **Symbol-definition boost.** Chunks whose defined symbol matches a query
  identifier get a rank-0 signal (+2.8pp code, zero regressions).
- **Selective reranking.** Cross-encoder rerank helps code and standards,
  *hurts* memory recall (measured -10.5pp). The rerank policy is scope-aware;
  memory is never reranked.
- **Contextual chunk prefixes.** Every chunk embeds with a
  `type | repo | file | symbol` header, not raw text.
- **Freshness without wall-clock.** Optional recency prior for memory scope,
  deterministic (reference = max mtime among candidates), so evals stay
  reproducible.

## Quickstart

```bash
git clone https://github.com/LucasSantana-Dev/shelfmark && cd shelfmark
python3 -m venv venv && venv/bin/pip install -r requirements.txt

mkdir -p ~/.shelfmark
cp sources.yaml.example ~/.shelfmark/sources.yaml   # edit: your repos + note globs

venv/bin/python build.py                            # index everything
venv/bin/python query.py "how do we handle retry timeouts"
```

First build downloads `intfloat/multilingual-e5-small` (~120MB). Everything
after that runs offline.

## MCP server (agent integration)

```jsonc
// e.g. Claude Code: .mcp.json or ~/.claude.json
{
  "mcpServers": {
    "shelfmark": {
      "command": "/path/to/shelfmark/venv/bin/python",
      "args": ["/path/to/shelfmark/mcp_server.py"]
    }
  }
}
```

Two tools:

- `rag_query` — full-corpus hybrid search (code, docs, commits, notes).
  Auto-scopes to the repo your agent is working in.
- `search_knowledge` — cross-project search over durable knowledge only
  (memory/standards/plans/handoffs/adrs; configurable via
  `RAG_KNOWLEDGE_SCOPE`). Never reranked, by measurement.

`examples/claude-code/` has the full loop: auto-recall on every prompt
(UserPromptSubmit), incremental reindex on file writes (PostToolUse), drift
reindex + weekly report at session start, and a nightly rebuild with an eval
regression gate.

## What gets indexed

`sources.yaml` declares everything (see `sources.yaml.example`):

| kind | what |
|------|------|
| `repos` | source code (py/ts/js/sh, symbol-aware chunking), `docs/**`, README, CHANGELOG, `docs/specs/**`, roadmap, last 180d of commit messages |
| `sources` | arbitrary markdown globs, each under a free `type` label you filter on at query time |
| `code_globs` | loose scripts outside any repo |

Incremental reindex (`build.py --incremental <files>`) keeps writes cheap;
`session_chunker.py` can additionally index agent session transcripts.

## Evaluation

The eval harness is the part most RAG setups skip. `eval/run.py` scores
Hit@1/3/5 + MRR per scope against a JSONL dataset; `eval/check.sh` gates any
change at >5pp regression vs a frozen baseline; `eval/holdout-policy.md`
documents the train/holdout discipline (the holdout set is never used for
tuning — numbers quoted from it are honest).

This repo ships a public, reproducible dataset (`eval/dataset-public.jsonl`)
whose queries target this repository's own code and docs:

```bash
venv/bin/python build.py                 # index this repo (sources.yaml.example works as-is)
venv/bin/python eval/run.py --dataset eval/dataset-public.jsonl --label mine
```

Benchmark results and methodology: [BENCHMARK.md](BENCHMARK.md).

To evaluate on YOUR corpus, write ~50 `{"query", "expect_path_contains",
"expect_scope"}` lines, freeze a fifth of them as holdout, and wire
`eval/check.sh` into your nightly rebuild. That regression gate is what keeps
retrieval quality from silently rotting as the corpus grows.

## Configuration

All optional — see `.env.example` for the full list. Highlights:

| var | default | effect |
|-----|---------|--------|
| `RAG_HOME` | `~/.shelfmark` | data dir (index, sources.yaml) |
| `RAG_MODEL` / `RAG_DIM` | e5-small / 384 | embedding model |
| `RAG_BM25_WEIGHT` | 1.5 | >1 favors lexical match |
| `RAG_RERANK_AUTO` | on | rerank weak/ambiguous queries |
| `RAG_CODE_RERANK` | off | bge-reranker-v2-m3 for code scopes (+4.9pp, ~2.2GB) |
| `RAG_QLOG` | off | local query telemetry (powers `report.py`) |

## License

MIT
