#!/usr/bin/env python3
"""Case-cluster bootstrap for stored saliency-transfer predictions.

The input is a data-derived local table and is not distributed with the code
release. It contains one row per case, training target, evaluation target and
metric. ``mean`` averages the stored execution-level predictions for that
case, and ``appearances`` records how many predictions contributed. Resampling
cases while retaining this weight preserves the original equal-partition
point estimand and accounts for cases appearing in multiple test partitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("NSS", "CC", "KLDiv", "sAUC")
LOWER_IS_BETTER = {"NSS": False, "CC": False, "KLDiv": True, "sAUC": False}
CONTRASTS = {
    "session_second_target": ("post_report", "pre_report", "post_report"),
    "session_first_target": ("pre_report", "post_report", "pre_report"),
    "reader_group_subspecialty_target": (
        "expert_consensus",
        "generalist_consensus",
        "expert_consensus",
    ),
    "reader_group_general_radiologist_target": (
        "generalist_consensus",
        "expert_consensus",
        "generalist_consensus",
    ),
}


def _case_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def validate(frame: pd.DataFrame) -> list[object]:
    required = {"case", "trained_on", "scored_against", "metric", "mean", "appearances"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing values in required columns")
    if frame.duplicated(["case", "trained_on", "scored_against", "metric"]).any():
        raise ValueError("Duplicate case/training/evaluation/metric rows")
    if (frame["appearances"] <= 0).any():
        raise ValueError("Every case-level row must have at least one appearance")
    cases = sorted(frame["case"].unique(), key=_case_sort_key)
    return cases


def analyse(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 20260824,
) -> pd.DataFrame:
    cases = validate(frame)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(cases), size=(n_bootstrap, len(cases)))
    rows: list[dict[str, object]] = []

    for metric in METRICS:
        for contrast, (matched_train, alternative_train, target) in CONTRASTS.items():
            subset = frame[
                (frame["metric"] == metric)
                & (frame["scored_against"] == target)
                & (frame["trained_on"].isin([matched_train, alternative_train]))
            ]
            values = subset.pivot(index="case", columns="trained_on", values="mean").loc[cases]
            weights = subset.pivot(index="case", columns="trained_on", values="appearances").loc[cases]
            if values.isna().any().any() or weights.isna().any().any():
                raise ValueError(f"Incomplete matrix for {metric}: {contrast}")
            if not np.array_equal(weights[matched_train], weights[alternative_train]):
                raise ValueError(f"Appearance mismatch for {metric}: {contrast}")

            if LOWER_IS_BETTER[metric]:
                difference = values[alternative_train].to_numpy() - values[matched_train].to_numpy()
            else:
                difference = values[matched_train].to_numpy() - values[alternative_train].to_numpy()
            case_weights = weights[matched_train].to_numpy(dtype=float)
            draws = difference[sampled]
            draw_weights = case_weights[sampled]
            weighted_draws = np.sum(draws * draw_weights, axis=1) / np.sum(draw_weights, axis=1)
            equal_case_draws = draws.mean(axis=1)

            rows.append(
                {
                    "metric": metric,
                    "contrast": contrast,
                    "evaluation_target": target,
                    "n_case_clusters": len(cases),
                    "weighted_point": float(np.average(difference, weights=case_weights)),
                    "weighted_ci_low": float(np.percentile(weighted_draws, 2.5)),
                    "weighted_ci_high": float(np.percentile(weighted_draws, 97.5)),
                    "equal_case_point": float(difference.mean()),
                    "equal_case_ci_low": float(np.percentile(equal_case_draws, 2.5)),
                    "equal_case_ci_high": float(np.percentile(equal_case_draws, 97.5)),
                    "positive_case_clusters": int((difference > 0).sum()),
                    "zero_case_clusters": int((difference == 0).sum()),
                    "negative_case_clusters": int((difference < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Local per-case saliency CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output contrast CSV")
    parser.add_argument("--metadata", type=Path, help="Optional JSON audit record")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    frame = pd.read_csv(args.source)
    result = analyse(frame, n_bootstrap=args.bootstrap, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if args.metadata:
        payload = {
            "source": str(args.source),
            "case_clusters": int(result["n_case_clusters"].iloc[0]),
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.seed,
            "primary_estimand": "case-cluster bootstrap with retained appearance weights",
        }
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
