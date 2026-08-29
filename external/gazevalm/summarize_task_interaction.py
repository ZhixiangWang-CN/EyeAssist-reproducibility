#!/usr/bin/env python3
"""Stratify the GazeVaLM task-direction interaction by stimulus authenticity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ci(values: np.ndarray, indices: np.ndarray) -> list[float]:
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimulus-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    data = pd.read_csv(args.stimulus_scores)
    required = {
        "stimulus",
        "authenticity",
        "study_id",
        "target_task",
        "matched_minus_opposite",
        "matched_minus_half_mixed",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    table = data.pivot_table(
        index=["study_id", "authenticity"],
        columns="target_task",
        values=["matched_minus_opposite", "matched_minus_half_mixed"],
        aggfunc="first",
    ).dropna()
    rng = np.random.default_rng(args.seed)
    result: dict[str, object] = {
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
        "strata": {},
    }
    stratum_arrays: dict[str, dict[str, np.ndarray]] = {}
    for authenticity in ("real", "fake"):
        frame = table.xs(authenticity, level="authenticity").sort_index()
        arrays = {
            "matched_minus_opposite_task1": frame[("matched_minus_opposite", "Task1")].to_numpy(float),
            "matched_minus_opposite_task2": frame[("matched_minus_opposite", "Task2")].to_numpy(float),
            "matched_minus_half_task1": frame[("matched_minus_half_mixed", "Task1")].to_numpy(float),
            "matched_minus_half_task2": frame[("matched_minus_half_mixed", "Task2")].to_numpy(float),
        }
        arrays["opposite_interaction"] = (
            arrays["matched_minus_opposite_task1"] - arrays["matched_minus_opposite_task2"]
        )
        arrays["half_mixed_interaction"] = (
            arrays["matched_minus_half_task1"] - arrays["matched_minus_half_task2"]
        )
        stratum_arrays[authenticity] = arrays
        indices = rng.integers(0, len(frame), size=(args.bootstrap, len(frame)))
        result["strata"][authenticity] = {
            "source_study_clusters": int(len(frame)),
            "estimates": {
                name: {"mean": float(values.mean()), "ci95": ci(values, indices)}
                for name, values in arrays.items()
            },
        }

    paired = table.reset_index().pivot(index="study_id", columns="authenticity")
    common = paired.dropna().index
    indices = rng.integers(0, len(common), size=(args.bootstrap, len(common)))
    result["real_minus_fake_interaction"] = {}
    for metric in ("matched_minus_opposite", "matched_minus_half_mixed"):
        real = (
            paired.loc[common, (metric, "Task1", "real")].to_numpy(float)
            - paired.loc[common, (metric, "Task2", "real")].to_numpy(float)
        )
        fake = (
            paired.loc[common, (metric, "Task1", "fake")].to_numpy(float)
            - paired.loc[common, (metric, "Task2", "fake")].to_numpy(float)
        )
        delta = real - fake
        result["real_minus_fake_interaction"][metric] = {
            "mean": float(delta.mean()),
            "ci95": ci(delta, indices),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
