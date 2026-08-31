#!/usr/bin/env python3
"""Patient-independent repeated-split AUROC inference.

The local prediction table contains one row per test case and stored split,
with all model arms kept on the same row. A separate local case-to-patient map
defines independence. A prediction is retained only when every record from the
same patient appears in that split's test set. The retained predictions are
then analysed with a label-stratified patient-cluster bootstrap.

No study data, predictions, weights or numerical outputs are distributed with
this source-code release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LABEL = "y_true(1=异常)"
ARMS = {
    "image_only": "image_only_prob",
    "generalist_gaze": "generalist_gaze_prob",
    "first_session_gaze": "cold_read_gaze_prob",
    "second_session_gaze": "informed_gaze_prob",
}


def auc(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    weight = np.ones(len(y), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    comparison = (score[pos, None] > score[None, neg]).astype(float)
    comparison += 0.5 * (score[pos, None] == score[None, neg])
    wp = weight[pos]
    wn = weight[neg]
    denominator = wp.sum() * wn.sum()
    if denominator == 0:
        return float("nan")
    return float(np.sum(comparison * wp[:, None] * wn[None, :]) / denominator)


def _read_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet or 0, dtype={"case": str})
    return pd.read_csv(path, dtype={"case": str})


def prepare_patient_independent_rows(
    raw: pd.DataFrame,
    patient_map: pd.DataFrame,
    *,
    label_column: str,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    required = {"run", "case", label_column, *ARMS.values()}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Prediction table has missing analysis values")
    if raw.duplicated(["run", "case"]).any():
        raise ValueError("Prediction table contains duplicate run-case rows")

    map_required = {"case", "patient"}
    map_missing = map_required.difference(patient_map.columns)
    if map_missing:
        raise ValueError(f"Patient map is missing columns: {sorted(map_missing)}")
    if patient_map["case"].duplicated().any():
        raise ValueError("Patient map contains duplicate case rows")

    frame = raw.copy()
    frame["case"] = frame["case"].astype(str)
    mapping = patient_map[["case", "patient"]].astype(str)
    frame = frame.merge(mapping, on="case", how="left", validate="many_to_one")
    if frame["patient"].isna().any():
        absent = sorted(frame.loc[frame.patient.isna(), "case"].unique())
        raise ValueError(f"Patient map does not cover cases: {absent}")

    cohort_members = mapping.groupby("patient")["case"].apply(set).to_dict()
    clean_frames: list[pd.DataFrame] = []
    audit: list[dict[str, int]] = []
    for run, run_frame in frame.groupby("run", sort=True):
        test_cases = set(run_frame["case"])
        keep = run_frame.apply(
            lambda row: cohort_members[row["patient"]].issubset(test_cases), axis=1
        )
        retained = run_frame.loc[keep].copy()
        clean_frames.append(retained)
        audit.append(
            {
                "run": int(run),
                "original_test_records": int(len(run_frame)),
                "retained_test_records": int(len(retained)),
                "retained_test_patients": int(retained["patient"].nunique()),
            }
        )

    clean = pd.concat(clean_frames, ignore_index=True)
    if clean.groupby("patient")[label_column].nunique().max() != 1:
        raise ValueError("A patient has inconsistent classification labels")
    return clean, audit


def analyse(
    raw: pd.DataFrame,
    patient_map: pd.DataFrame,
    *,
    label_column: str = DEFAULT_LABEL,
    n_bootstrap: int = 20_000,
    seed: int = 20260830,
) -> dict[str, object]:
    clean, run_audit = prepare_patient_independent_rows(
        raw, patient_map, label_column=label_column
    )

    per_run: dict[str, list[float]] = {arm: [] for arm in ARMS}
    grouped_runs = list(clean.groupby("run", sort=True))
    for _, frame in grouped_runs:
        y = frame[label_column].to_numpy(dtype=int)
        for arm, column in ARMS.items():
            per_run[arm].append(auc(y, frame[column].to_numpy(dtype=float)))
    point = {arm: float(np.nanmean(values)) for arm, values in per_run.items()}

    patient_labels = clean.groupby("patient")[label_column].first()
    patients = sorted(patient_labels.index.astype(str))
    positive = [patient for patient in patients if int(patient_labels.loc[patient]) == 1]
    negative = [patient for patient in patients if int(patient_labels.loc[patient]) == 0]
    if not positive or not negative:
        raise ValueError("Patient-cluster AUROC requires both outcome classes")
    patient_index = {patient: index for index, patient in enumerate(patients)}

    rng = np.random.default_rng(seed)
    weights = np.zeros((n_bootstrap, len(patients)), dtype=float)
    positive_index = [patient_index[patient] for patient in positive]
    negative_index = [patient_index[patient] for patient in negative]
    weights[:, positive_index] = rng.multinomial(
        len(positive), np.full(len(positive), 1 / len(positive)), size=n_bootstrap
    )
    weights[:, negative_index] = rng.multinomial(
        len(negative), np.full(len(negative), 1 / len(negative)), size=n_bootstrap
    )

    bootstrap = {arm: np.full(n_bootstrap, np.nan) for arm in ARMS}
    for draw in range(n_bootstrap):
        for arm, column in ARMS.items():
            split_values = []
            for _, frame in grouped_runs:
                row_weights = np.asarray(
                    [weights[draw, patient_index[str(patient)]] for patient in frame["patient"]],
                    dtype=float,
                )
                split_values.append(
                    auc(
                        frame[label_column].to_numpy(dtype=int),
                        frame[column].to_numpy(dtype=float),
                        row_weights,
                    )
                )
            bootstrap[arm][draw] = np.nanmean(split_values)

    pairs = [
        ("generalist_gaze", "image_only"),
        ("first_session_gaze", "image_only"),
        ("second_session_gaze", "image_only"),
        ("second_session_gaze", "generalist_gaze"),
        ("second_session_gaze", "first_session_gaze"),
    ]
    contrasts: dict[str, object] = {}
    for left, right in pairs:
        draws = bootstrap[left] - bootstrap[right]
        contrasts[f"{left}_minus_{right}"] = {
            "point_difference": float(point[left] - point[right]),
            "patient_cluster_ci95": [
                float(value) for value in np.nanpercentile(draws, [2.5, 97.5])
            ],
            "bootstrap_probability_le_zero": float(np.nanmean(draws <= 0)),
        }

    return {
        "retention_rule": (
            "retain a test prediction only when every image-case record from that patient "
            "is present in the same test split"
        ),
        "original_rows": int(len(raw)),
        "retained_rows": int(len(clean)),
        "represented_case_records": int(clean["case"].nunique()),
        "represented_patients": int(clean["patient"].nunique()),
        "runs": int(clean["run"].nunique()),
        "test_records_per_run": {
            "min": int(min(row["retained_test_records"] for row in run_audit)),
            "median": float(np.median([row["retained_test_records"] for row in run_audit])),
            "max": int(max(row["retained_test_records"] for row in run_audit)),
        },
        "bootstrap": {
            "unit": "patient",
            "stratification": "outcome label",
            "resamples": int(n_bootstrap),
            "seed": int(seed),
        },
        "arms": {
            arm: {
                "mean_split_auc": point[arm],
                "patient_cluster_ci95": [
                    float(value)
                    for value in np.nanpercentile(bootstrap[arm], [2.5, 97.5])
                ],
                "valid_runs": int(np.isfinite(per_run[arm]).sum()),
            }
            for arm in ARMS
        },
        "paired_contrasts": contrasts,
        "run_audit": run_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Local paired predictions")
    parser.add_argument(
        "--patient-map",
        type=Path,
        required=True,
        help="Local CSV with case and patient columns",
    )
    parser.add_argument("--source-sheet", default="per_case")
    parser.add_argument("--label-column", default=DEFAULT_LABEL)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, required=True, help="Local JSON output")
    args = parser.parse_args()

    raw = _read_table(args.source, args.source_sheet)
    patient_map = pd.read_csv(args.patient_map, dtype=str)
    result = analyse(
        raw,
        patient_map,
        label_column=args.label_column,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
