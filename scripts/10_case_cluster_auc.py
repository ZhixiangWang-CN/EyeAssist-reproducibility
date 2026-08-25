#!/usr/bin/env python3
"""Case-clustered AUROC reanalysis using only the supplied results workbook.

The reported point estimand remains the mean AUROC across the 50 stored paired
test splits. Uncertainty is recomputed by resampling the 75 unique cases as
clusters, with all appearances of a sampled case and all four paired arms kept
together. Within each bootstrap sample, cluster multiplicities are used as
weights inside each original split and the valid split AUROCs are averaged.

This script never averages probabilities across fitted models, because doing so
would create a different implicit-ensemble estimand.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
SHEET = "per_case"
LABEL = "y_true(1=异常)"
ARMS = {
    "image_only": "image_only_prob",
    "generalist_gaze": "generalist_gaze_prob",
    "cold_read_gaze": "cold_read_gaze_prob",
    "informed_gaze": "informed_gaze_prob",
}
SEED = 20260824
N_BOOT = 20_000


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def weighted_auc_batch(wp: np.ndarray, wn: np.ndarray, cmp: np.ndarray) -> np.ndarray:
    numerator = np.einsum("bi,ij,bj->b", wp, cmp, wn)
    denominator = wp.sum(axis=1) * wn.sum(axis=1)
    out = np.full(len(denominator), np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def interval(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]


def validate(raw: pd.DataFrame) -> dict[str, object]:
    required = {"run", "case", LABEL, *ARMS.values()}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    if raw[list(required)].isna().any().any():
        raise RuntimeError("Missing values in analysis columns")
    if raw.duplicated(["run", "case"]).any():
        raise RuntimeError("Duplicate run-case rows")
    if raw.groupby("case")[LABEL].nunique().max() != 1:
        raise RuntimeError("Inconsistent label within case cluster")
    by_run = raw.groupby("run").agg(rows=("case", "size"), pos=(LABEL, "sum"))
    if not ((by_run.rows == 15) & (by_run.pos == 12)).all():
        raise RuntimeError("Expected 15 test cases (12 abnormal, 3 normal) per run")
    labels = raw.groupby("case")[LABEL].first()
    if len(raw) != 750 or raw.run.nunique() != 50 or len(labels) != 75:
        raise RuntimeError("Expected 750 rows, 50 runs and 75 unique cases")
    return {
        "rows": int(len(raw)),
        "runs": int(raw.run.nunique()),
        "unique_cases": int(len(labels)),
        "abnormal_cases": int(labels.sum()),
        "normal_cases": int((1 - labels).sum()),
        "appearances_per_case": {
            "min": int(raw.groupby("case").size().min()),
            "median": float(raw.groupby("case").size().median()),
            "max": int(raw.groupby("case").size().max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Case-level results workbook")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "outputs/classification_case_cluster_auc.json",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    raw = pd.read_excel(source, sheet_name=SHEET)
    validation = validate(raw)
    stored_runs = pd.read_excel(source, sheet_name="runs")

    case_labels = raw.groupby("case")[LABEL].first()
    cases = sorted(case_labels.index, key=lambda value: int(value))
    case_to_i = {case: i for i, case in enumerate(cases)}
    labels = np.array([int(case_labels.loc[case]) for case in cases])
    pos_cases = np.flatnonzero(labels == 1)
    neg_cases = np.flatnonzero(labels == 0)

    # Original descriptive estimand: mean AUROC across the 50 paired splits.
    per_run: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for _, frame in raw.groupby("run", sort=True):
        y = frame[LABEL].to_numpy(dtype=int)
        for arm, column in ARMS.items():
            per_run[arm].append(auc(y, frame[column].to_numpy(dtype=float)))
    point = {arm: float(np.mean(values)) for arm, values in per_run.items()}
    run_reconciliation: dict[str, object] = {}
    for arm in ARMS:
        stored = (
            stored_runs.loc[stored_runs.arm == arm, ["run", "auc"]]
            .sort_values("run")
            .auc.to_numpy(dtype=float)
        )
        recomputed = np.asarray(per_run[arm], dtype=float)
        delta = recomputed - stored
        run_reconciliation[arm] = {
            "stored_runs_mean_auc": float(stored.mean()),
            "recomputed_from_per_case_mean_auc": float(recomputed.mean()),
            "max_abs_run_difference": float(np.max(np.abs(delta))),
            "runs_differing_over_5e-5": int(np.sum(np.abs(delta) > 5e-5)),
            "mismatched_runs": [
                {
                    "run": int(index + 1),
                    "stored": float(stored[index]),
                    "recomputed": float(recomputed[index]),
                    "difference": float(delta[index]),
                }
                for index in np.flatnonzero(np.abs(delta) > 5e-5)
            ],
        }

    # Standard label-stratified nonparametric cluster bootstrap.
    rng = np.random.default_rng(SEED)
    weights = np.zeros((N_BOOT, len(cases)), dtype=float)
    weights[:, pos_cases] = rng.multinomial(
        len(pos_cases), np.full(len(pos_cases), 1 / len(pos_cases)), size=N_BOOT
    )
    weights[:, neg_cases] = rng.multinomial(
        len(neg_cases), np.full(len(neg_cases), 1 / len(neg_cases)), size=N_BOOT
    )

    arm_run_boot: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    run_valid: list[np.ndarray] = []
    for _, frame in raw.groupby("run", sort=True):
        y = frame[LABEL].to_numpy(dtype=int)
        row_case = np.array([case_to_i[case] for case in frame.case])
        pos_local = np.flatnonzero(y == 1)
        neg_local = np.flatnonzero(y == 0)
        wp = weights[:, row_case[pos_local]]
        wn = weights[:, row_case[neg_local]]
        run_valid.append((wp.sum(axis=1) > 0) & (wn.sum(axis=1) > 0))
        for arm, column in ARMS.items():
            score = frame[column].to_numpy(dtype=float)
            ps = score[pos_local]
            ns = score[neg_local]
            cmp = (ps[:, None] > ns[None, :]).astype(float)
            cmp += 0.5 * (ps[:, None] == ns[None, :])
            arm_run_boot[arm].append(weighted_auc_batch(wp, wn, cmp))

    boot = {
        arm: np.nanmean(np.column_stack(values), axis=1)
        for arm, values in arm_run_boot.items()
    }
    valid_count = np.column_stack(run_valid).sum(axis=1)

    arms_out = {
        arm: {
            "mean_split_auc": point[arm],
            "case_cluster_bootstrap_ci95": interval(boot[arm]),
        }
        for arm in ARMS
    }

    pairs = [
        ("generalist_gaze", "image_only"),
        ("cold_read_gaze", "image_only"),
        ("informed_gaze", "image_only"),
        ("cold_read_gaze", "generalist_gaze"),
        ("informed_gaze", "generalist_gaze"),
        ("informed_gaze", "cold_read_gaze"),
    ]
    contrasts: dict[str, object] = {}
    for left, right in pairs:
        dist = boot[left] - boot[right]
        split_diff = np.asarray(per_run[left]) - np.asarray(per_run[right])
        contrasts[f"{left}_minus_{right}"] = {
            "point_difference": float(point[left] - point[right]),
            "case_cluster_bootstrap_ci95": interval(dist),
            "bootstrap_probability_le_zero": float(np.mean(dist <= 0)),
            "split_wins_ties_losses": [
                int(np.sum(split_diff > 0)),
                int(np.sum(split_diff == 0)),
                int(np.sum(split_diff < 0)),
            ],
        }

    result = {
        "source": source.name,
        "source_sheet": SHEET,
        "validation": validation,
        "run_reconciliation": run_reconciliation,
        "estimand": "mean test AUROC across the 50 stored paired split/model runs",
        "bootstrap": {
            "unit": "unique case cluster",
            "pairing": "all appearances and all four arms retained together",
            "stratification": "62 abnormal and 13 normal cases sampled with replacement within label",
            "resamples": N_BOOT,
            "seed": SEED,
            "valid_split_count_per_resample": {
                "min": int(valid_count.min()),
                "median": float(np.median(valid_count)),
                "mean": float(valid_count.mean()),
                "max": int(valid_count.max()),
            },
        },
        "arms": arms_out,
        "paired_contrasts": contrasts,
        "excluded_analysis": "No averaging or pooling of probabilities across fitted models; cross-run scores need not share a calibration scale and such pooling would change the estimand.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
