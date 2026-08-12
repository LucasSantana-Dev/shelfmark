## What

<!-- One paragraph: what changes and why. Link the issue if one exists. -->

## Checklist

- [ ] Title follows conventional commits (`feat:`, `fix:`, `docs:`, ...)
- [ ] `venv/bin/python diff_pack_test.py` passes
- [ ] `venv/bin/python eval/test_harness.py` passes
- [ ] `bash eval/check.sh` passes (retrieval regression gate)

## Eval impact (only if this touches retrieval behavior)

<!--
Before/after numbers from:
  venv/bin/python eval/run.py --dataset eval/dataset-public.jsonl --label my-change
Never tune against the holdout set (eval/holdout-policy.md).
-->
