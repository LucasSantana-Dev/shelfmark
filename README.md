# sqlrecall

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/LucasSantana-Dev/sqlrecall?style=flat)](https://github.com/LucasSantana-Dev/sqlrecall/stargazers)
[![MCP](https://img.shields.io/badge/MCP-server-6b5ce7)](https://modelcontextprotocol.io)

Agent memory and code recall in one local SQLite file. No services, no cloud.
BM25 + dense embeddings fused with Reciprocal Rank Fusion, an optional cross-encoder reranker, and an MCP server so coding agents
(Claude Code, Codex, anything MCP) can recall your notes, decisions, docs, and
code instead of guessing.

Pairs with session-memory tools like
[ai-memory](https://github.com/akitaonrails/ai-memory): they remember what
happened, sqlrecall finds what was written down.

> **Formerly `shelfmark`.** Renamed in 1.2.0. `pip install sqlrecall` replaces
> `shelfmark-rag`; the old `shelfmark-*` commands still work as aliases, and an
> existing `~/.shelfmark` data dir is picked up automatically when
> `~/.sqlrecall` does not exist.

**[Why this exists](#why-this-exists)** · **[Quickstart](#quickstart)** ·
**[MCP server](#mcp-server-agent-integration)** ·
**[What gets indexed](#what-gets-indexed)** · **[Evaluation](#evaluation)** ·
**[How it compares](#how-it-compares)** · **[Configuration](#configuration)**

## Why this exists

Agent harnesses accumulate knowledge: memory notes, ADRs, plans, standards,
handoffs, and the code itself. The useful answer to most questions is already
written down somewhere. sqlrecall indexes all of it into one local SQLite file
and answers "what did we decide about X / where did we handle Y" in a single
query, with `path:line` citations.

sqlrecall is the retrieval engine that came out of a year of running AI coding
agents daily — most recently distilled from
[forgekit](https://github.com/LucasSantana-Dev/forgekit), an AI dev toolkit for
coding agents. Every design choice below was forced by a real failure mode in
that daily use, not picked off a paper.

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
pip install sqlrecall   # or: pipx install sqlrecall

sqlrecall-build      # first run writes a starter ~/.sqlrecall/sources.yaml — edit it, then rerun
sqlrecall-query "how do we handle retry timeouts"
```

First build downloads `intfloat/multilingual-e5-small` (~120MB). Everything
after that runs offline.

<details>
<summary>Prefer a local checkout instead? (contributing, editing the source)</summary>

```bash
git clone https://github.com/LucasSantana-Dev/sqlrecall && cd sqlrecall
python3 -m venv venv && venv/bin/pip install -e .

venv/bin/sqlrecall-build
venv/bin/sqlrecall-query "how do we handle retry timeouts"
```

</details>

## MCP server (agent integration)

```jsonc
// e.g. Claude Code: .mcp.json or ~/.claude.json
{
  "mcpServers": {
    "sqlrecall": {
      "command": "sqlrecall-mcp"
    }
  }
}
```

(Local checkout instead of `pipx install`? Use `"command": "/path/to/sqlrecall/venv/bin/sqlrecall-mcp"`.)

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

## Session capture and handoffs

One command, called from each agent's lifecycle hooks, records what a session
did and passes a baton to the next session in the same project, across
harnesses (start in Claude Code, continue in Codex or OpenCode):

```bash
shelfmark-hook --agent claude-code --event session-start   # reads the hook JSON on stdin
```

- **Capture.** Prompts, tool calls (name, target, pass/fail, short output) and
  the last reply of each turn go to a SQLite store next to the index
  (`index.memory.sqlite`, or `index.client-<slug>.memory.sqlite` under a
  client's roots). Secret-shaped strings are redacted and bodies capped at
  16 KiB before anything is written. A cwd under a purged client is never
  captured. Zero LLM calls.
- **Automatic handoff.** Every `stop` refreshes one open handoff per session:
  last request, last reply, files edited, recent commands with failures.
- **Delivery.** At `session-start` the newest open handoff from another session
  in the project (worktrees count as their main checkout) is claimed once and
  printed, fenced as untrusted data; Claude Code and Codex add that output to
  the new session's context. Written handoffs win over automatic ones;
  claiming an automatic one expires the older automatic ones. Open handoffs
  expire after `SHELFMARK_HANDOFF_TTL_DAYS` (14).
- **MCP tools.** `handoff_begin` (leave a written handoff), `handoff_list`,
  `handoff_accept`. The layer always comes from the server process, never
  from a tool argument.

Adapters: `examples/claude-code/capture-hooks.json` (merge into
`settings.json`), `examples/codex/hooks.json`, and
`examples/opencode/shelfmark.ts` (plugin, not yet run against a live
OpenCode). `SHELFMARK_CAPTURE=off` disables capture;
`SHELFMARK_HANDOFF_ON_START=off` keeps capture but skips delivery.
`shelfmark-purge <client>` drops the client's memory store with its index.

## What gets indexed

`sources.yaml` declares everything (see `sources.yaml.example`):

| kind | what |
|------|------|
| `repos` | source code (py/ts/js/sh, symbol-aware chunking), `docs/**`, README, CHANGELOG, `docs/specs/**`, roadmap, last 180d of commit messages |
| `sources` | arbitrary markdown globs, each under a free `type` label you filter on at query time |
| `code_globs` | loose scripts outside any repo |

Incremental reindex (`indexer.py --incremental <files>`) keeps writes cheap;
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
venv/bin/python indexer.py                 # index this repo (sources.yaml.example works as-is)
venv/bin/python eval/run.py --dataset eval/dataset-public.jsonl --label mine
```

Benchmark results and methodology: [BENCHMARK.md](BENCHMARK.md).

To evaluate on YOUR corpus, write ~50 `{"query", "expect_path_contains",
"expect_scope"}` lines, freeze a fifth of them as holdout, and wire
`eval/check.sh` into your nightly rebuild. That regression gate is what keeps
retrieval quality from silently rotting as the corpus grows.

## How it compares

No cross-tool benchmark exists yet (see [Methodology & honest
limitations](BENCHMARK.md#methodology--honest-limitations) — we'd genuinely
like to see one run). Qualitative trade-offs, no fabricated numbers:

| Option | When to pick sqlrecall instead | Why |
|---|---|---|
| **mem0** (managed, cloud-first memory layer) | You want full local data ownership and cross-repo search | One SQLite file, zero setup, MCP native. mem0 adds a hosted service you may not need |
| **Letta / MemGPT** (stateful agent framework) | You want a retriever, not a framework | Standalone tool that plugs into any MCP client; Letta expects you to adopt its agent runtime |
| **Zep** (hosted conversation memory) | You want permanent, local, cross-repo recall, not just chat threads | Single machine, no hosted dependency; Zep targets conversation history, not code/docs |
| **Cursor / Continue / Cody built-in indexing** | You want search outside one editor, or from a non-IDE agent | Runs anywhere MCP runs; editor-built-in indexes don't leave the editor |
| **DIY LangChain + Chroma/Weaviate** | You want hybrid retrieval and an eval gate without wiring it yourself | Reranking, RRF fusion, and `eval/check.sh` regression gates ship in the box |
| **Claude Code's built-in project memory** | You want hybrid (lexical + semantic) search across repos, not one workspace | File-based single-workspace memory has no ranking and no cross-repo scope |
| **Chroma / Weaviate raw, or Pinecone** (no framework) | You want zero infrastructure to stand up | One SQLite file vs. a vector DB service + embedding pipeline + glue code |
| **grep / ripgrep** | You need paraphrase recall, not just exact substrings | Lexical-only; sqlrecall fuses BM25 with embeddings so "retry timeout" also matches "backoff on failure" |

Pick a hosted vector DB when you need multi-tenant scale across millions of
documents — that's a different problem than agent recall over your own repos.

Want to measure and improve retrieval ranking quality on your own pipeline,
independent of any specific agent? [hitgate](https://github.com/LucasSantana-Dev/hitgate)
provides label-free regression testing for hybrid retrievers. sqlrecall and
hitgate share a common hybrid-retrieval foundation (BM25 + embeddings + RRF)
but serve complementary use cases: sqlrecall for zero-setup agent memory with
MCP integration, hitgate for ranking evaluation and quality gates on any
retriever you already have.

## Configuration

All optional — see `.env.example` for the full list. Highlights:

| var | default | effect |
|-----|---------|--------|
| `RAG_HOME` | `~/.sqlrecall` | data dir (index, sources.yaml) |
| `RAG_MODEL` / `RAG_DIM` | e5-small / 384 | embedding model |
| `RAG_BM25_WEIGHT` | 1.5 | >1 favors lexical match |
| `RAG_RERANK_AUTO` | on | rerank weak/ambiguous queries |
| `RAG_CODE_RERANK` | off | bge-reranker-v2-m3 for code scopes (+4.9pp, ~2.2GB) |
| `RAG_QLOG` | off | local query telemetry (powers `report.py`) |
| `RAG_CLIENT` | from cwd | active client layer; `none` = general only |

### Client layers

If you work for more than one client, keep each client's business knowledge
out of the index every session reads. Declare clients in `sources.yaml`:

```yaml
clients:
  acme:
    roots: [~/dev/acme-app, ~/notes/acme]
```

- Files under a client's `roots`, or with `client: acme` in their frontmatter
  (top level or under `metadata:`), are indexed into that client's own file
  (`index.client-acme.sqlite`), never the general index. `client: none` keeps a
  note general even inside a root (logged at index time). Once a `client` key
  is present, anything else (unknown slug, list, empty, broken YAML) skips the
  file instead of sending it to general.
- A query reads the general index plus the **active** client's file only. The
  active client comes from the process (`RAG_CLIENT`, or the process cwd under
  a client root), never from a tool-call argument, so an agent cannot switch
  itself into another client's layer.
- Session transcripts are routed by the cwd they recorded. Known limit: a
  session started outside every client root that then reads a client's files
  is indexed as general. Start client sessions inside the client's root, or
  leave `session_chunker.py` off.
- BM25 is scored per file, so one client's vocabulary never shapes another's
  term statistics; results are then ranked together. With no clients declared,
  ranking is unchanged.
- A client's data lives only in its file and its `index.client-<slug>.backup-*`
  snapshots. Removing a client from `sources.yaml` while its file still exists
  stops the next build (its untagged files would otherwise land in general):
  archive or remove the file and its `sources` globs first. The last build's
  clients are recorded in `$RAG_HOME/clients.json`, so this also holds for a
  client with a custom `db:` path.
- Upgrading from a version without client layers: run one full rebuild.
  Chunks now store the resolved path, and rows written before keep the old
  spelling until rebuilt.
- Root matching ignores case on macOS/Windows (`str.casefold`, close to but
  not exactly the file system's own folding).

To offboard a client, purge before removing it from `sources.yaml`:

```bash
sqlrecall-purge acme --lexicon acme-terms.txt                       # dry run: what would go
sqlrecall-purge acme --apply --archive age1... --lexicon acme-terms.txt
```

The purge streams the client's index, its backups and its query-log rows into
one [age](https://age-encryption.org)-encrypted archive (never plaintext on
disk). Then it deletes them and scrubs any residue of the client from the
general index and its backups with `secure_delete` + `VACUUM`, so nothing
survives in free pages or the WAL. It then checks that the lexicon's canary
terms appear nowhere in the raw bytes of what remains. Notes under the
client's roots tagged `client: none` (lessons you decided are general) are
kept. `--no-archive` deletes without an archive. A tombstone keeps later
builds from routing the client's files into general if its globs stay.
Source files are never touched.

## Contributing

Issues and PRs welcome — especially a real cross-tool benchmark (see [How it
compares](#how-it-compares)), support for more embedding models, or
non-Claude MCP client examples. If sqlrecall saves your agent a guess, a star
helps others find it.

## License

MIT
