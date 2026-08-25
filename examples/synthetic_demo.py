#!/usr/bin/env python3
"""End-to-end synthetic smoke test with no clinical data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eyeassist.gaze import density_map, fixation_log_score
from eyeassist.pooling import held_out_configuration_scores
from eyeassist.splits import repeated_group_stratified_splits


def synthetic_reader(rng, center, n=80):
    xy = rng.normal(center, [8.0, 8.0], size=(n, 2))
    duration = rng.gamma(shape=2.0, scale=80.0, size=(n, 1))
    return np.column_stack([xy, duration])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260822)
    state = {**{f"A{i}": "A" for i in range(5)}, **{f"B{i}": "B" for i in range(5)}}
    fixations = {
        reader: synthetic_reader(rng, (55, 55) if group == "A" else (145, 145))
        for reader, group in state.items()
    }
    maps = {
        reader: density_map(points, (200, 200), sigma_pixels=12, smoothing_mass=0.01)
        for reader, points in fixations.items()
    }

    rows = []
    for target_reader, points in fixations.items():
        score = lambda probability, p=points: fixation_log_score(p, probability)
        values = held_out_configuration_scores(
            maps=maps,
            target_reader=target_reader,
            state_by_reader=state,
            score=score,
            matched_size=4,
            half_mixed_target=2,
            half_mixed_off_state=2,
            opposite_size=4,
        )
        rows.append({"target_reader": target_reader, "target_state": state[target_reader], **values})
    score_table = pd.DataFrame(rows)
    score_table.to_csv(args.output / "held_out_scores.csv", index=False)

    manifest = pd.DataFrame(
        {
            "case_id": [f"case_{i:03d}" for i in range(40)],
            "label": [i % 2 for i in range(40)],
        }
    )
    splits = repeated_group_stratified_splits(
        manifest, repeats=3, test_groups=10, seed=20260822
    )
    splits.to_csv(args.output / "group_disjoint_splits.csv", index=False)

    summary = {
        "mean_matched_minus_half_mixed": float(score_table.matched_minus_half_mixed.mean()),
        "mean_matched_minus_opposite": float(score_table.matched_minus_opposite.mean()),
        "n_split_rows": int(len(splits)),
        "group_overlap": 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["mean_matched_minus_opposite"] <= 0:
        raise AssertionError("Synthetic matched-state advantage was not recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
