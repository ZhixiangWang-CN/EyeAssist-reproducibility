#!/usr/bin/env python3
"""Case-clustered classifier sensitivity at fixed specificity.

This complements AUROC and the fixed 0.5-threshold operating metrics.  It
interpolates each stored split's empirical ROC curve at declared specificity
levels, then resamples the 75 unique cases while retaining every case's
repeated-split membership and all four model arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = {
    "No gaze": "image_only_last",
    "General-radiologist-group gaze": "generalist_gaze_last",
    "Subspecialist, first-session pre-report": "cold_read_gaze_last",
    "Subspecialist, second-session post-report": "informed_gaze_last",
}


def sensitivity_at_specificity(
    y: np.ndarray,
    score: np.ndarray,
    specificity: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate the weighted empirical ROC at one specificity.

    ``sample_weights`` has shape bootstrap x observations.  With ``None``, a
    single unweighted estimate is returned.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if sample_weights is None:
        weights = np.ones((1, len(y)), dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    w = weights[:, order]
    positive = w * (y_sorted == 1)[None, :]
    negative = w * (y_sorted == 0)[None, :]
    pos_total = positive.sum(axis=1)
    neg_total = negative.sum(axis=1)
    valid = (pos_total > 0) & (neg_total > 0)
    tpr = np.column_stack([np.zeros(len(w)), np.cumsum(positive, axis=1)])
    fpr = np.column_stack([np.zeros(len(w)), np.cumsum(negative, axis=1)])
    tpr = np.divide(tpr, pos_total[:, None], out=np.full_like(tpr, np.nan), where=pos_total[:, None] > 0)
    fpr = np.divide(fpr, neg_total[:, None], out=np.full_like(fpr, np.nan), where=neg_total[:, None] > 0)
    target = 1.0 - specificity

    out = np.full(len(w), np.nan, dtype=float)
    exact_mask = np.isclose(fpr, target, atol=1e-12)
    exact_any = exact_mask.any(axis=1) & valid
    if np.any(exact_any):
        exact_tpr = np.where(exact_mask, tpr, -np.inf)
        out[exact_any] = np.max(exact_tpr[exact_any], axis=1)
    interpolate = valid & ~exact_any
    if np.any(interpolate):
        rows = np.flatnonzero(interpolate)
        x = fpr[rows]
        z = tpr[rows]
        lo = np.sum(x < target, axis=1) - 1
        hi = np.argmax(x > target, axis=1)
        row_index = np.arange(len(rows))
        x_lo, x_hi = x[row_index, lo], x[row_index, hi]
        z_lo, z_hi = z[row_index, lo], z[row_index, hi]
        fraction = (target - x_lo) / (x_hi - x_lo)
        out[rows] = z_lo + fraction * (z_hi - z_lo)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--specificity", type=float, nargs="+", default=[0.8, 0.9])
    args = parser.parse_args()

    data = pd.read_csv(args.predictions)
    required = {"run", "case", "y_true", *ARMS.values()}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    labels = data.groupby("case").y_true.first()
    if len(labels) != 75 or int(labels.sum()) != 62:
        raise ValueError("Expected 75 cases comprising 62 abnormal and 13 normal")
    cases = labels.index.astype(str).tolist()
    case_to_i = {case: i for i, case in enumerate(cases)}
    y_global = labels.to_numpy(dtype=int)
    positive = np.flatnonzero(y_global == 1)
    negative = np.flatnonzero(y_global == 0)
    rng = np.random.default_rng(args.seed)
    weights = np.zeros((args.bootstrap, len(cases)), dtype=float)
    weights[:, positive] = rng.multinomial(
        len(positive), np.full(len(positive), 1 / len(positive)), size=args.bootstrap
    )
    weights[:, negative] = rng.multinomial(
        len(negative), np.full(len(negative), 1 / len(negative)), size=args.bootstrap
    )

    result: dict[str, object] = {
        "estimand": "mean split sensitivity at a declared ROC specificity",
        "specificity_levels": args.specificity,
        "case_cluster_bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
        "arms": {},
        "contrasts": {},
    }
    distributions: dict[float, dict[str, np.ndarray]] = {}
    for specificity in args.specificity:
        point_by_arm: dict[str, list[float]] = {name: [] for name in ARMS}
        boot_by_arm: dict[str, list[np.ndarray]] = {name: [] for name in ARMS}
        for _, frame in data.groupby("run", sort=True):
            y = frame.y_true.to_numpy(dtype=int)
            indices = np.asarray([case_to_i[str(case)] for case in frame.case])
            local_weights = weights[:, indices]
            for name, column in ARMS.items():
                score = frame[column].to_numpy(dtype=float)
                point_by_arm[name].append(
                    float(sensitivity_at_specificity(y, score, specificity)[0])
                )
                boot_by_arm[name].append(
                    sensitivity_at_specificity(y, score, specificity, local_weights)
                )
        distributions[specificity] = {}
        for name in ARMS:
            matrix = np.column_stack(boot_by_arm[name])
            dist = np.nanmean(matrix, axis=1)
            distributions[specificity][name] = dist
            result["arms"].setdefault(name, {})[f"specificity_{specificity:.2f}"] = {
                "mean_split_sensitivity": float(np.mean(point_by_arm[name])),
                "case_cluster_ci95": [float(v) for v in np.nanpercentile(dist, [2.5, 97.5])],
            }
        base = distributions[specificity]["No gaze"]
        for name in list(ARMS)[1:]:
            delta = distributions[specificity][name] - base
            result["contrasts"].setdefault(f"{name} minus No gaze", {})[
                f"specificity_{specificity:.2f}"
            ] = {
                "point_difference": float(
                    result["arms"][name][f"specificity_{specificity:.2f}"]["mean_split_sensitivity"]
                    - result["arms"]["No gaze"][f"specificity_{specificity:.2f}"]["mean_split_sensitivity"]
                ),
                "case_cluster_ci95": [
                    float(v) for v in np.nanpercentile(delta, [2.5, 97.5])
                ],
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
