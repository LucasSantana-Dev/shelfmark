# shelfmark — retrieval benchmark

**Config:** `intfloat/multilingual-e5-small` (384d) + BM25 (code-aware tokenizer), RRF fusion, no reranker (`[FAST]` mode).
**Corpus:** this repository itself — code + README indexed with `sources.yaml.example` semantics (~120 chunks).
**Datasets:** `eval/dataset-public.jsonl` (42 cases, train) · `eval/holdout-public.jsonl` (10 cases, frozen — never used for tuning, per `eval/holdout-policy.md`).

## Reproduce

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
mkdir -p ~/.shelfmark && printf 'repos:\n  - %s\n' "$(pwd)" > ~/.shelfmark/sources.yaml
venv/bin/python build.py
RAG_QLOG=off venv/bin/python eval/run.py --dataset eval/dataset-public.jsonl --label train
RAG_QLOG=off venv/bin/python eval/run.py --dataset eval/holdout-public.jsonl --label holdout
```

## Results

| Set | n | Hit@1 | Hit@3 | Hit@5 | MRR |
|-----|---|-------|-------|-------|-----|
| Train | 42 | 0.762 | 0.881 | **0.905** | 0.827 |
| Holdout (frozen) | 10 | 0.800 | 1.000 | **1.000** | 0.883 |

Committed baselines: `eval/baseline-public-train.json`, `eval/baseline-public-holdout.json`
(metrics only). `eval/check.sh` fails at >5pp regression vs a baseline:

```bash
RAG_EVAL_DATASET=eval/dataset-public.jsonl \
RAG_EVAL_BASELINE=eval/baseline-public-train.json \
  eval/check.sh my-change
```

## Methodology & honest limitations

- **Small self-referential corpus.** Queries target this repo's own code and
  docs so anyone can reproduce the table with three commands. A ~120-chunk
  corpus is far easier than a real knowledge base; treat these numbers as a
  smoke-test ceiling, not a claim about your corpus.
- **Author-written queries.** Cases were written by the engine's maintainers.
  They are natural-language paraphrases (no copied identifiers except where a
  developer would genuinely type one), and 4 known-hard cases that the engine
  currently MISSES are kept in the train set on purpose.
- **Holdout discipline.** The holdout split was frozen before any post-split
  tuning and is never used to pick parameters. If we ever tune against it, the
  numbers here stop being honest — `eval/holdout-policy.md` is the contract.
- **No cross-tool comparison here.** This table measures shelfmark against
  itself over time (regression gate), not against other retrieval tools. A
  fair cross-tool comparison needs identical corpora and query sets — if you
  run one, we'd genuinely like to see it.

## Production numbers (private corpus, for context)

On the maintainer's real corpus — ~7,800 chunks across memory notes, ADRs,
plans, standards, and five code repos — the same engine measured
**Hit@5 0.587 / MRR 0.478 on a 46-case frozen holdout** (no-rerank config,
2026-06). Real corpora are much harder than this repo's self-test: more
distractors, older notes, cross-repo ambiguity. Those eval sets contain
private content and are not distributable, which is exactly why the public
dataset above exists.
