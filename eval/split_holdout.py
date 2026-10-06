#!/usr/bin/env python3
"""Deterministic group split of the public eval set by target file.

Every case whose expect_path_contains is the same file lands on the same side, so
file-dependent tuning on train cannot see a holdout target. Rule (see
eval/holdout-policy.md):
  1. group key = expect_path_contains (a single string). Cases with a list target
     always stay in train, and so does every file named in a list target (a list
     case would otherwise leak that file into train).
  2. groups with more than MAX_GROUP cases stay in train (too big to hold out
     without blowing the ~20% budget).
  3. remaining groups are ordered by sha256(SEED + ":" + key) and added to the
     holdout while the holdout stays <= TARGET cases.
Usage: python3 eval/split_holdout.py [--write]   (default: dry run + overlap check)
Case text is never edited; cases only move between the two files, in original order.
"""
import hashlib
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
TRAIN, HOLD = D / "dataset-public.jsonl", D / "holdout-public.jsonl"
SEED, TARGET, MAX_GROUP = "shelfmark-holdout-v2", 10, 3


def load(p):
    return [l for l in p.read_text().splitlines() if l.strip()]


def key(line):
    t = json.loads(line)["expect_path_contains"]
    return t if isinstance(t, str) else None


def main():
    # Stable original order: train file first, then the old holdout cases.
    lines = load(TRAIN) + load(HOLD)
    groups = {}
    for ln in lines:
        k = key(ln)
        if k is not None:
            groups.setdefault(k, []).append(ln)
    listed = set()
    for ln in lines:
        t = json.loads(ln)["expect_path_contains"]
        if isinstance(t, list):
            listed.update(t)
    eligible = [k for k, v in groups.items() if len(v) <= MAX_GROUP and k not in listed]
    eligible.sort(key=lambda k: hashlib.sha256(f"{SEED}:{k}".encode()).hexdigest())
    chosen, n = set(), 0
    for k in eligible:
        if n + len(groups[k]) <= TARGET:
            chosen.add(k)
            n += len(groups[k])
    hold = [ln for ln in lines if key(ln) in chosen]
    train = [ln for ln in lines if key(ln) not in chosen]

    def files(ls):
        out = set()
        for ln in ls:
            t = json.loads(ln)["expect_path_contains"]
            out.update([t] if isinstance(t, str) else t)
        return out

    overlap = files(train) & files(hold)
    print(f"train={len(train)} holdout={len(hold)} overlapping target files={len(overlap)} {sorted(overlap)}")
    print("holdout groups:", sorted(chosen))
    if "--write" in sys.argv:
        TRAIN.write_text("\n".join(train) + "\n")
        HOLD.write_text("\n".join(hold) + "\n")


if __name__ == "__main__":
    main()
