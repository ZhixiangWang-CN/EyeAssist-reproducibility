#!/usr/bin/env python3
"""Summarize saliency-transfer results at the unique-partition level.

The original experiment contains three execution records for each of 17 random
case partitions. This script verifies that nesting and summarizes that original
partition-level audit. The submission's primary uncertainty analysis, including
the coverage-completion partition and all 75 cases, is implemented in
``17_saliency_case_cluster_bootstrap.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTRASTS = {
    "session_second_target": ("pre_report", "post_report"),
    "session_first_target": ("post_report", "pre_report"),
    "reader_group_subspecialty_target": ("generalist_consensus", "expert_consensus"),
    "reader_group_general_radiologist_target": ("expert_consensus", "generalist_consensus"),
}


def nss(run: dict, trained_on: str, scored_against: str) -> float:
    values = np.asarray(run["matrix"][trained_on][scored_against]["NSS"], dtype=float)
    return float(np.nanmean(values))


def bootstrap_mean(values: np.ndarray, *, seed: int, resamples: int) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(resamples, len(values)))
    distribution = values[indices].mean(axis=1)
    low, high = np.percentile(distribution, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, help="Glob matching expA_shard*.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--resamples", type=int, default=20_000)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.shards))]
    if not paths:
        raise FileNotFoundError(f"No shard files matched {args.shards!r}")
    shards = [json.loads(path.read_text()) for path in paths]
    counts = {len(shard["runs"]) for shard in shards}
    if len(counts) != 1:
        raise ValueError(f"Shard run counts differ: {sorted(counts)}")
    n_partitions = counts.pop()

    rows = []
    for index in range(n_partitions):
        runs = [shard["runs"][index] for shard in shards]
        test_sets = {tuple(run["test_cases"]) for run in runs}
        seeds = {int(run["seed"]) for run in runs}
        if len(test_sets) != 1 or len(seeds) != 1:
            raise ValueError(f"Execution records do not share partition {index + 1}")
        row = {
            "partition": index + 1,
            "seed": seeds.pop(),
            "test_cases": ";".join(next(iter(test_sets))),
            "executions": len(runs),
        }
        for name, (alternative, target) in CONTRASTS.items():
            execution_values = np.asarray(
                [nss(run, target, target) - nss(run, alternative, target) for run in runs]
            )
            row[name] = float(execution_values.mean())
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = []
    for name in CONTRASTS:
        values = frame[name].to_numpy(float)
        low, high = bootstrap_mean(values, seed=args.seed, resamples=args.resamples)
        summary.append(
            {
                "contrast": name,
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "positive_partitions": int((values > 0).sum()),
                "n_partitions": len(values),
                "executions_per_partition": len(shards),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".partitions.csv"), index=False)
    args.output.write_text(json.dumps({"contrasts": summary}, indent=2))
    print(json.dumps({"contrasts": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
