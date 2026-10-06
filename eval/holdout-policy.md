# Holdout Set Policy

## Rationale

Automatic expansion of eval sets during tuning is a survivorship-bias trap. The
eval set becomes optimized for the **current** retriever, and tuning against an
auto-grown set gives a false sense of generalization.

## Governance

- **Locked:** the holdout file (`eval/holdout-public.jsonl` here; whatever you
  name yours) is a **frozen, deterministic ~20% group split** of the dataset.
  It is never auto-grown or modified in response to retrieval results.
- **Split rule (since 2026-10-06): group by target file.** The group key is
  `expect_path_contains`. Every target file lives entirely in train or entirely
  in holdout. A per-case split leaks: file-dependent tuning (symbol boost, chunk
  prefix, BM25 weight) sees the target file through its training cases, so the
  holdout inherits the advantage. Before 2026-10-06 this repo used a per-case
  split and 8 of the 10 holdout cases shared their target file with a training
  case. The split is reproducible with `python3 eval/split_holdout.py` (seed
  `shelfmark-holdout-v2`; groups ordered by sha256 of seed and key, added while
  the holdout stays <= 10 cases). Cases with a list target, files named in a
  list target, and groups of more than 3 cases (`retrieval.py`, `indexer.py`,
  `mcp_server.py`) stay in train. Consequence: the holdout never exercises those
  large files. Verify zero file overlap after any re-curation.
- **Tuning:** all configuration and model selection is done against the train
  set (`eval/dataset-public.jsonl`).
- **Final validation:** the holdout set is used ONLY for final acceptance
  testing after tuning is complete. Numbers quoted from it are honest; numbers
  quoted from the train set are not evidence of generalization.
- **Re-curation:** the holdout set may be manually re-curated on a quarterly
  basis if:
  - New scope types emerge (e.g., novel query patterns).
  - The dataset composition shifts significantly (>30% of cases change).
  - A regression > 0.05 is detected in holdout vs. train on a deployed config.
  Every re-curation resets the baseline (`eval/run.py --dataset <holdout>
  --label baseline-...`) and is a commit, so the history of the contract is
  auditable.

## For your own corpus

Write ~50 `{"query", "expect_path_contains", "expect_scope"}` cases against
your real corpus, split ~20% out deterministically **by target file** (not by case, see Locked),
freeze baselines for both files, and wire `eval/check.sh` into your rebuild
cadence. Keep known-hard MISS cases in the train set — a dataset the engine
scores 100% on cannot detect regressions.

## Re-curation log

- 2026-10-06: per-case split replaced by a target-file group split (leak fix).
  Baselines for both files re-frozen. Holdout numbers published before this date
  are not comparable. n is still about 10, so the holdout only detects large
  effects (roughly 30pp or more); growing it is a separate follow-up.
