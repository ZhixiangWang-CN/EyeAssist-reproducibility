#!/usr/bin/env python3
"""Summarize GazeVaLM task-pool concentration and authenticity-stratified interactions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_fixed_pool import Config, find_task_roots, load_stimulus_records, official_participant_set


def percentile_ci(values: np.ndarray, indices: np.ndarray) -> list[float]:
    distribution = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(distribution, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Unpacked GazeVaLM root")
    parser.add_argument("--score-csv", type=Path, required=True, help="Stimulus-level score table")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_primary.json"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    config = Config.from_json(args.config)
    task_roots = find_task_roots(args.data_root)
    participants, _ = official_participant_set(task_roots)
    rows: list[dict[str, object]] = []
    stimuli = sorted(
        {path.name for root in task_roots.values() for path in root.iterdir()
         if path.is_dir() and path.name.startswith(("real_", "fake_"))}
    )
    for stimulus in stimuli:
        loaded = {}
        for task in ("Task1", "Task2"):
            path = task_roots[task] / stimulus / "scanpaths.csv"
            if path.exists():
                loaded[task], _, _ = load_stimulus_records(
                    path, task, stimulus, config, args.cache, participants
                )
        if set(loaded) != {"Task1", "Task2"}:
            continue
        common = sorted(set(loaded["Task1"]) & set(loaded["Task2"]))
        if not common:
            continue
        authenticity, study_id = stimulus.split("_", 1)
        for task in ("Task1", "Task2"):
            pooled = np.mean([loaded[task][reader].astype(float) for reader in common], axis=0)
            pooled /= pooled.sum()
            entropy = float(-(pooled * np.log2(pooled)).sum())
            rows.append({
                "stimulus": stimulus,
                "authenticity": authenticity,
                "study_id": study_id,
                "task": task,
                "n_common_readers": len(common),
                "entropy_bits": entropy,
                "shannon_support": float(2 ** entropy),
                "simpson_support": float(1 / np.square(pooled).sum()),
            })

    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    pivot = frame.pivot(
        index=["study_id", "authenticity", "stimulus"],
        columns="task",
        values=["entropy_bits", "shannon_support", "simpson_support"],
    )
    rng = np.random.default_rng(args.seed)
    studies = sorted(frame.study_id.unique())
    indices = rng.integers(0, len(studies), size=(args.bootstrap, len(studies)))
    summary: dict[str, object] = {
        "config": {"sigma_native_px": config.sigma_native_px, "alpha": config.alpha,
                   "seed": args.seed, "bootstrap": args.bootstrap},
        "audit": {"stimuli": int(frame.stimulus.nunique()), "rows": int(len(frame)),
                  "readers_min": int(frame.n_common_readers.min()),
                  "readers_max": int(frame.n_common_readers.max())},
        "task_means": {},
        "task1_minus_task2": {},
    }
    for metric in ("entropy_bits", "shannon_support", "simpson_support"):
        summary["task_means"][metric] = {
            task: float(frame.loc[frame.task == task, metric].mean())
            for task in ("Task1", "Task2")
        }
        difference = pivot[(metric, "Task1")] - pivot[(metric, "Task2")]
        values = difference.groupby(level="study_id").mean().reindex(studies).to_numpy(float)
        summary["task1_minus_task2"][metric] = {
            "mean": float(values.mean()), "ci95": percentile_ci(values, indices)
        }

    scores = pd.read_csv(args.score_csv)
    score_pivot = scores.pivot_table(
        index=["study_id", "authenticity"], columns="target_task",
        values=["matched_minus_opposite", "matched_minus_half_mixed"], aggfunc="first"
    ).dropna()
    summary["task_interaction_by_authenticity"] = {}
    for authenticity in ("real", "fake"):
        part = score_pivot.xs(authenticity, level="authenticity").sort_index()
        auth_indices = rng.integers(0, len(part), size=(args.bootstrap, len(part)))
        summary["task_interaction_by_authenticity"][authenticity] = {}
        for metric in ("matched_minus_opposite", "matched_minus_half_mixed"):
            values = (part[(metric, "Task1")] - part[(metric, "Task2")]).to_numpy(float)
            summary["task_interaction_by_authenticity"][authenticity][metric] = {
                "mean": float(values.mean()), "ci95": percentile_ci(values, auth_indices)
            }

    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
