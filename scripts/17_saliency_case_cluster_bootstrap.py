#!/usr/bin/env python3
"""Case- or patient-cluster bootstrap for saliency-transfer predictions.

The input is a data-derived local table and is not distributed with the code
release. It contains one row per case, training target, evaluation target and
metric. ``mean`` averages the stored execution-level predictions for that
case, and ``appearances`` records how many predictions contributed. Repeated
held-out predictions are first averaged within case. With ``--patient-map``,
case-level contrasts are averaged within patient and patients are sampled with
equal weight. The local source table must already have applied the manuscript's
all-records-held-out retention rule; no study data are distributed here.
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
    patient_map: pd.DataFrame | None = None,
    n_bootstrap: int = 20_000,
    seed: int = 20260830,
) -> pd.DataFrame:
    cases = validate(frame)
    case_to_cluster = {case: str(case) for case in cases}
    cluster_unit = "case"
    if patient_map is not None:
        required = {"case", "patient"}
        missing = required.difference(patient_map.columns)
        if missing:
            raise ValueError(f"Patient map is missing columns: {sorted(missing)}")
        if patient_map["case"].duplicated().any():
            raise ValueError("Patient map contains duplicate case rows")
        local_map = {
            str(case): str(patient)
            for case, patient in patient_map[["case", "patient"]].itertuples(index=False)
        }
        absent = sorted(set(map(str, cases)).difference(local_map), key=_case_sort_key)
        if absent:
            raise ValueError(f"Patient map does not cover cases: {absent}")
        case_to_cluster = {case: local_map[str(case)] for case in cases}
        cluster_unit = "patient"

    clusters = sorted(set(case_to_cluster.values()), key=_case_sort_key)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(clusters), size=(n_bootstrap, len(clusters)))
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
            case_difference = pd.Series(difference, index=[str(case) for case in cases])
            case_appearance = pd.Series(case_weights, index=[str(case) for case in cases])
            cluster_difference = np.asarray(
                [
                    case_difference[
                        [str(case) for case in cases if case_to_cluster[case] == cluster]
                    ].mean()
                    for cluster in clusters
                ],
                dtype=float,
            )
            cluster_appearance = np.asarray(
                [
                    case_appearance[
                        [str(case) for case in cases if case_to_cluster[case] == cluster]
                    ].sum()
                    for cluster in clusters
                ],
                dtype=float,
            )
            draws = cluster_difference[sampled]
            draw_weights = cluster_appearance[sampled]
            equal_cluster_draws = draws.mean(axis=1)
            weighted_draws = np.sum(draws * draw_weights, axis=1) / np.sum(draw_weights, axis=1)

            rows.append(
                {
                    "metric": metric,
                    "contrast": contrast,
                    "evaluation_target": target,
                    "cluster_unit": cluster_unit,
                    "n_clusters": len(clusters),
                    "n_case_records": len(cases),
                    "primary_point": float(cluster_difference.mean()),
                    "primary_ci_low": float(np.percentile(equal_cluster_draws, 2.5)),
                    "primary_ci_high": float(np.percentile(equal_cluster_draws, 97.5)),
                    "weighted_point": float(np.average(cluster_difference, weights=cluster_appearance)),
                    "weighted_ci_low": float(np.percentile(weighted_draws, 2.5)),
                    "weighted_ci_high": float(np.percentile(weighted_draws, 97.5)),
                    "equal_cluster_point": float(cluster_difference.mean()),
                    "equal_cluster_ci_low": float(np.percentile(equal_cluster_draws, 2.5)),
                    "equal_cluster_ci_high": float(np.percentile(equal_cluster_draws, 97.5)),
                    "positive_clusters": int((cluster_difference > 0).sum()),
                    "zero_clusters": int((cluster_difference == 0).sum()),
                    "negative_clusters": int((cluster_difference < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Local per-case saliency CSV")
    parser.add_argument(
        "--patient-map",
        type=Path,
        help="Optional local CSV with case and patient columns for patient-cluster inference",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output contrast CSV")
    parser.add_argument("--metadata", type=Path, help="Optional JSON audit record")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    frame = pd.read_csv(args.source)
    patient_map = pd.read_csv(args.patient_map, dtype=str) if args.patient_map else None
    result = analyse(
        frame,
        patient_map=patient_map,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    if args.metadata:
        payload = {
            "source": str(args.source),
            "cluster_unit": str(result["cluster_unit"].iloc[0]),
            "clusters": int(result["n_clusters"].iloc[0]),
            "case_records": int(result["n_case_records"].iloc[0]),
            "bootstrap_resamples": args.bootstrap,
            "bootstrap_seed": args.seed,
            "primary_estimand": "equal-cluster mean with cluster bootstrap",
            "sensitivity_estimand": "appearance-weighted mean with cluster bootstrap",
        }
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
