# Holdout Set Policy

## Rationale

Automatic expansion of eval sets during tuning is a survivorship-bias trap. The
eval set becomes optimized for the **current** retriever, and tuning against an
auto-grown set gives a false sense of generalization.

## Governance

- **Locked:** the holdout file (`eval/holdout-public.jsonl` here; whatever you
  name yours) is a **frozen, deterministic ~20% sample** of the train dataset.
  It is never auto-grown or modified in response to retrieval results.
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
your real corpus, split ~20% out deterministically (e.g. every 5th case),
freeze baselines for both files, and wire `eval/check.sh` into your rebuild
cadence. Keep known-hard MISS cases in the train set — a dataset the engine
scores 100% on cannot detect regressions.
