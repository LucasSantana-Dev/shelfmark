# Contributing to shelfmark

Thanks for considering a contribution. Issues and PRs are welcome; the ones
we want most right now:

- A real **cross-tool benchmark** (see
  [How it compares](README.md#how-it-compares) and
  [BENCHMARK.md](BENCHMARK.md#methodology--honest-limitations))
- Support for **more embedding models**
- **Non-Claude MCP client examples** (Codex, Cursor, anything MCP)

## Dev setup

```bash
git clone https://github.com/LucasSantana-Dev/shelfmark && cd shelfmark
python3 -m venv venv && venv/bin/pip install -e .

cp sources.yaml.example sources.yaml   # works as-is: indexes this repo
venv/bin/shelfmark-build
venv/bin/shelfmark-query "how does rerank policy work"
```

First build downloads `intfloat/multilingual-e5-small` (~120MB); everything
after runs offline.

## Tests and the eval gate

CI runs three things on every PR (`.github/workflows/eval.yml`):

```bash
venv/bin/python diff_pack_test.py       # unit tests
venv/bin/python eval/test_harness.py    # eval harness self-tests
bash eval/check.sh                      # retrieval regression gate
```

The eval gate is the important one: it scores Hit@1/3/5 + MRR on
`eval/dataset-public.jsonl` against a frozen baseline and fails on a >5pp
regression. If your change intentionally shifts retrieval quality, say so in
the PR and include before/after numbers from:

```bash
venv/bin/python eval/run.py --dataset eval/dataset-public.jsonl --label my-change
```

Never tune against the holdout set; `eval/holdout-policy.md` explains the
discipline.

## Code style

- Python >= 3.10, stdlib-first. New runtime dependencies need a strong
  justification (the current list is 4 packages; keeping install light is a
  feature).
- Flat module layout (top-level `.py` files); match the existing style of the
  file you touch.
- Retrieval behavior changes need a rationale comment near the code and
  measured eval numbers in the PR, mirroring the existing inline
  "measured +X.Xpp" comments.

## Commits and PRs

- Conventional commits: `feat:`, `fix:`, `docs:`, optional scope like
  `fix(eval): ...`.
- Keep PRs focused: one behavior change per PR, eval numbers where relevant.
- For security issues, do not open a PR or issue; see
  [SECURITY.md](SECURITY.md).
