#!/usr/bin/env python3
"""Independent all-reader EyeAssist-PE slice-stable sensitivity analysis.

Expected layout below ``--gaze-root``::

    Session 1/R1/Trial.csv ... Session 1/R7/Trial.csv
    Session 2/R1/Trial.csv ... Session 2/R7/Trial.csv

``Session1``/``Session2`` (without a space) are also accepted.  The script
does not modify source data or author code.  A slice-stable run is a maximal
ordered sequence with an unchanged synchronized ``Slice Number``.  Run length
is expressed in CSV rows because the packaged Figure 4 streams do not provide
sub-second timestamps suitable for reconstructing the manuscript's I-VT
durations.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260824
READERS = [f"R{i}" for i in range(1, 8)]
SESSIONS = (1, 2)
PRIMARY_MIN_RUN_ROWS = 3
SENSITIVITY_MIN_RUN_ROWS = (2, 3, 5)

PAIR_METRICS = (
    "n_rows",
    "slice_transitions",
    "unique_slices",
    "slice_range",
    "adjacent_same_slice_fraction",
    "stable_rows",
    "stable_row_fraction",
    "stable_run_count",
    "mean_stable_run_rows",
    "median_stable_run_rows",
    "median_stable_dispersion_px",
    "median_stable_endpoint_amplitude_px",
    "median_stable_path_length_px",
)

FEATURE_SETS = {
    "count_plus_geometry": (
        "stable_rows",
        "stable_row_fraction",
        "stable_run_count",
        "mean_stable_run_rows",
        "median_stable_run_rows",
        "median_stable_dispersion_px",
        "median_stable_endpoint_amplitude_px",
        "median_stable_path_length_px",
    ),
    "composition_plus_geometry_only": (
        "stable_row_fraction",
        "mean_stable_run_rows",
        "median_stable_run_rows",
        "median_stable_dispersion_px",
        "median_stable_endpoint_amplitude_px",
        "median_stable_path_length_px",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaze-root",
        required=True,
        type=Path,
        help="EyeAssist-PE Gaze directory containing both session folders",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for JSON and CSV audit files",
    )
    parser.add_argument(
        "--min-record-rows",
        type=int,
        default=100,
        help="QC threshold below which a recording is flagged short/truncated (default: 100)",
    )
    parser.add_argument(
        "--min-run-rows",
        type=int,
        default=PRIMARY_MIN_RUN_ROWS,
        help="Primary minimum number of consecutive equal-slice rows (default: 3)",
    )
    return parser.parse_args()


def resolve_trial(gaze_root: Path, session: int, reader: str) -> Path | None:
    candidates = [
        gaze_root / f"Session {session}" / reader / "Trial.csv",
        gaze_root / f"Session{session}" / reader / "Trial.csv",
        gaze_root / f"Session_{session}" / reader / "Trial.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_float(value: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("blank numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite numeric value")
    return result


def parse_int(value: str) -> int:
    number = parse_float(value)
    integer = int(number)
    if number != integer:
        raise ValueError(f"non-integer slice index: {number}")
    return integer


def read_trial(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped = 0
    total = 0
    required = {
        "Time",
        "Gaze_Location_X",
        "Gaze_Location_Y",
        "Slice Number",
        "File Name",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required columns {missing}")
        for source in reader:
            total += 1
            try:
                case = source["File Name"].strip()
                if not case:
                    raise ValueError("blank File Name")
                duration_raw = source.get("Duration", "")
                duration = parse_float(duration_raw) if duration_raw.strip() else None
                grouped[case].append(
                    {
                        "time": source["Time"].strip(),
                        "x": parse_float(source["Gaze_Location_X"]),
                        "y": parse_float(source["Gaze_Location_Y"]),
                        "slice": parse_int(source["Slice Number"]),
                        "duration": duration,
                    }
                )
            except (KeyError, TypeError, ValueError):
                dropped += 1
    return dict(grouped), {
        "path": str(path),
        "input_rows": total,
        "valid_rows": total - dropped,
        "dropped_rows": dropped,
        "case_count": len(grouped),
        "case_names": sorted(grouped),
    }


def stable_runs(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    runs: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(1, len(rows)):
        if rows[index]["slice"] != rows[index - 1]["slice"]:
            runs.append(rows[start:index])
            start = index
    runs.append(rows[start:])
    return runs


def median_or_nan(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(np.median(materialized)) if materialized else float("nan")


def mean_or_nan(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(np.mean(materialized)) if materialized else float("nan")


def run_geometry(run: list[dict[str, Any]]) -> tuple[float, float, float]:
    xy = np.asarray([(row["x"], row["y"]) for row in run], dtype=float)
    centre = xy.mean(axis=0)
    dispersion = float(np.sqrt(np.mean(np.sum((xy - centre) ** 2, axis=1))))
    endpoint_amplitude = float(np.linalg.norm(xy[-1] - xy[0]))
    path_length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    return dispersion, endpoint_amplitude, path_length


def summarize_recording(rows: list[dict[str, Any]], min_run_rows: int) -> dict[str, Any]:
    runs = stable_runs(rows)
    selected = [run for run in runs if len(run) >= min_run_rows]
    geometries = [run_geometry(run) for run in selected]
    n_rows = len(rows)
    transitions = sum(
        rows[index]["slice"] != rows[index - 1]["slice"]
        for index in range(1, n_rows)
    )
    selected_rows = sum(len(run) for run in selected)
    lengths = [float(len(run)) for run in selected]
    timestamps = [str(row["time"]) for row in rows]
    durations = [float(row["duration"]) for row in rows if row["duration"] is not None]
    return {
        "n_rows": n_rows,
        "slice_transitions": int(transitions),
        "unique_slices": len({row["slice"] for row in rows}),
        "slice_range": int(
            max(row["slice"] for row in rows) - min(row["slice"] for row in rows)
        ),
        "adjacent_same_slice_fraction": (
            float((n_rows - 1 - transitions) / (n_rows - 1))
            if n_rows > 1
            else float("nan")
        ),
        "stable_rows": selected_rows,
        "stable_row_fraction": float(selected_rows / n_rows) if n_rows else float("nan"),
        "stable_run_count": len(selected),
        "mean_stable_run_rows": mean_or_nan(lengths),
        "median_stable_run_rows": median_or_nan(lengths),
        "median_stable_dispersion_px": median_or_nan(g[0] for g in geometries),
        "median_stable_endpoint_amplitude_px": median_or_nan(g[1] for g in geometries),
        "median_stable_path_length_px": median_or_nan(g[2] for g in geometries),
        "x_min": min(float(row["x"]) for row in rows),
        "x_max": max(float(row["x"]) for row in rows),
        "y_min": min(float(row["y"]) for row in rows),
        "y_max": max(float(row["y"]) for row in rows),
        "duration_min": min(durations) if durations else None,
        "duration_max": max(durations) if durations else None,
        "timestamp_has_subsecond": any(
            "." in timestamp or len(timestamp) > 8 for timestamp in timestamps
        ),
    }


def exact_sign_flip_p(reader_differences: list[float]) -> float | None:
    values = np.asarray([value for value in reader_differences if math.isfinite(value)])
    if len(values) == 0:
        return None
    observed = abs(float(np.mean(values)))
    permuted = np.fromiter(
        (
            abs(float(np.mean(values * np.asarray(signs, dtype=float))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ),
        dtype=float,
        count=2 ** len(values),
    )
    return float(np.mean(permuted >= observed - 1e-15))


def pair_summary(pair_rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    valid = [row for row in pair_rows if math.isfinite(float(row[f"{metric}_diff"]))]
    differences = np.asarray([row[f"{metric}_diff"] for row in valid], dtype=float)
    s1 = np.asarray([row[f"{metric}_s1"] for row in valid], dtype=float)
    s2 = np.asarray([row[f"{metric}_s2"] for row in valid], dtype=float)
    by_reader: dict[str, dict[str, Any]] = {}
    reader_medians: list[float] = []
    for reader in READERS:
        values = np.asarray(
            [row[f"{metric}_diff"] for row in valid if row["reader"] == reader],
            dtype=float,
        )
        if len(values) == 0:
            by_reader[reader] = {"n_pairs": 0, "median_difference": None}
        else:
            median = float(np.median(values))
            by_reader[reader] = {
                "n_pairs": len(values),
                "median_difference": median,
                "mean_difference": float(np.mean(values)),
            }
            reader_medians.append(median)
    return {
        "n_reader_case_pairs": len(valid),
        "session1_median": float(np.median(s1)) if len(s1) else None,
        "session2_median": float(np.median(s2)) if len(s2) else None,
        "pooled_pair_median_difference_s2_minus_s1": (
            float(np.median(differences)) if len(differences) else None
        ),
        "pooled_pair_mean_difference_s2_minus_s1": (
            float(np.mean(differences)) if len(differences) else None
        ),
        "positive_pairs": int(np.sum(differences > 0)),
        "negative_pairs": int(np.sum(differences < 0)),
        "zero_pairs": int(np.sum(differences == 0)),
        "reader_median_differences": by_reader,
        "mean_of_reader_medians": (
            float(np.mean(reader_medians)) if reader_medians else None
        ),
        "two_sided_exact_reader_sign_flip_p": exact_sign_flip_p(reader_medians),
    }


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = np.asarray([fold["accuracy"] for fold in folds], dtype=float)
    aucs = np.asarray(
        [fold["auroc"] for fold in folds if fold["auroc"] is not None], dtype=float
    )
    return {
        "n_folds": len(folds),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_sd": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "accuracy_min": float(np.min(accuracies)),
        "accuracy_max": float(np.max(accuracies)),
        "auroc_mean": float(np.mean(aucs)) if len(aucs) else None,
        "auroc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "auroc_min": float(np.min(aucs)) if len(aucs) else None,
        "auroc_max": float(np.max(aucs)) if len(aucs) else None,
    }


def run_attribution(
    pair_rows: list[dict[str, Any]], feature_names: tuple[str, ...], feature_set: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x: list[list[float]] = []
    y: list[int] = []
    case_groups: list[str] = []
    reader_groups: list[str] = []
    for pair in pair_rows:
        for session in SESSIONS:
            values = [float(pair[f"{feature}_s{session}"]) for feature in feature_names]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(
                    f"Non-finite attribution feature for {pair['reader']} / {pair['case']}"
                )
            x.append(values)
            y.append(session - 1)
            case_groups.append(pair["case"])
            reader_groups.append(pair["reader"])
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=int)

    outputs: dict[str, Any] = {}
    all_folds: list[dict[str, Any]] = []
    for cv_name, groups in (
        ("LOCO", np.asarray(case_groups, dtype=object)),
        ("LORO", np.asarray(reader_groups, dtype=object)),
    ):
        splitter = LeaveOneGroupOut()
        probability = np.empty(len(y_array), dtype=float)
        prediction = np.empty(len(y_array), dtype=int)
        folds: list[dict[str, Any]] = []
        for fold_index, (train, test) in enumerate(
            splitter.split(x_array, y_array, groups), start=1
        ):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    random_state=SEED,
                    max_iter=10_000,
                ),
            )
            model.fit(x_array[train], y_array[train])
            probability[test] = model.predict_proba(x_array[test])[:, 1]
            prediction[test] = model.predict(x_array[test])
            held_out = sorted({str(value) for value in groups[test]})
            fold_auc = (
                float(roc_auc_score(y_array[test], probability[test]))
                if len(np.unique(y_array[test])) == 2
                else None
            )
            fold = {
                "feature_set": feature_set,
                "cv": cv_name,
                "fold": fold_index,
                "held_out_group": "|".join(held_out),
                "n_train": len(train),
                "n_test": len(test),
                "accuracy": float(accuracy_score(y_array[test], prediction[test])),
                "auroc": fold_auc,
            }
            folds.append(fold)
            all_folds.append(fold)
        outputs[cv_name] = {
            "grouping": "case" if cv_name == "LOCO" else "reader",
            "n_recordings": len(y_array),
            "n_reader_case_pairs": len(pair_rows),
            "pooled_accuracy": float(accuracy_score(y_array, prediction)),
            "pooled_auroc": float(roc_auc_score(y_array, probability)),
            "fold_summary": summarize_folds(folds),
            "folds": folds,
        }
    return outputs, all_folds


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.min_record_rows < 1 or args.min_run_rows < 2:
        raise ValueError("--min-record-rows must be >=1 and --min-run-rows must be >=2")
    args.out.mkdir(parents=True, exist_ok=True)

    raw: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    file_qc: list[dict[str, Any]] = []
    missing_files: list[dict[str, Any]] = []
    for reader in READERS:
        for session in SESSIONS:
            path = resolve_trial(args.gaze_root, session, reader)
            if path is None:
                missing_files.append({"reader": reader, "session": session})
                continue
            grouped, qc = read_trial(path)
            raw[(reader, session)] = grouped
            file_qc.append({"reader": reader, "session": session, **qc})

    if not raw:
        raise FileNotFoundError(f"No Trial.csv files found below {args.gaze_root}")

    thresholds = sorted(set(SENSITIVITY_MIN_RUN_ROWS + (args.min_run_rows,)))
    threshold_records: dict[
        int, dict[tuple[str, int, str], dict[str, Any]]
    ] = {threshold: {} for threshold in thresholds}
    per_record_rows: list[dict[str, Any]] = []
    short_records: list[dict[str, Any]] = []
    case_count_mismatches: list[dict[str, Any]] = []
    for qc in file_qc:
        if qc["case_count"] != 40:
            case_count_mismatches.append(
                {
                    "reader": qc["reader"],
                    "session": qc["session"],
                    "case_count": qc["case_count"],
                }
            )
    for (reader, session), by_case in raw.items():
        for case, rows in by_case.items():
            for threshold in thresholds:
                threshold_records[threshold][(reader, session, case)] = summarize_recording(
                    rows, threshold
                )
            metrics = threshold_records[args.min_run_rows][(reader, session, case)]
            short = metrics["n_rows"] < args.min_record_rows
            record = {
                "reader": reader,
                "session": session,
                "case": case,
                "source_file": str(resolve_trial(args.gaze_root, session, reader)),
                "short_or_truncated": short,
                "min_record_rows": args.min_record_rows,
                "min_run_rows": args.min_run_rows,
                **metrics,
            }
            per_record_rows.append(record)
            if short:
                short_records.append(
                    {
                        "reader": reader,
                        "session": session,
                        "case": case,
                        "n_rows": metrics["n_rows"],
                    }
                )

    def build_pairs(
        records: dict[tuple[str, int, str], dict[str, Any]], threshold: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        complete: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for reader in READERS:
            if (reader, 1) not in raw or (reader, 2) not in raw:
                continue
            case_union = sorted(set(raw[(reader, 1)]) | set(raw[(reader, 2)]))
            for case in case_union:
                key1, key2 = (reader, 1, case), (reader, 2, case)
                reasons: list[str] = []
                if key1 not in records:
                    reasons.append("missing_session1")
                if key2 not in records:
                    reasons.append("missing_session2")
                if not reasons:
                    m1, m2 = records[key1], records[key2]
                    if m1["n_rows"] < args.min_record_rows:
                        reasons.append("session1_short")
                    if m2["n_rows"] < args.min_record_rows:
                        reasons.append("session2_short")
                    if m1["stable_run_count"] == 0:
                        reasons.append("session1_no_stable_run")
                    if m2["stable_run_count"] == 0:
                        reasons.append("session2_no_stable_run")
                    for metric in PAIR_METRICS:
                        if not math.isfinite(float(m1[metric])) or not math.isfinite(
                            float(m2[metric])
                        ):
                            reasons.append(f"nonfinite_{metric}")
                if reasons:
                    exclusions.append(
                        {
                            "reader": reader,
                            "case": case,
                            "threshold": threshold,
                            "reasons": reasons,
                            "session1_rows": records.get(key1, {}).get("n_rows"),
                            "session2_rows": records.get(key2, {}).get("n_rows"),
                        }
                    )
                    continue
                pair: dict[str, Any] = {
                    "reader": reader,
                    "case": case,
                    "min_run_rows": threshold,
                }
                for metric in PAIR_METRICS:
                    value1 = float(records[key1][metric])
                    value2 = float(records[key2][metric])
                    pair[f"{metric}_s1"] = value1
                    pair[f"{metric}_s2"] = value2
                    pair[f"{metric}_diff"] = value2 - value1
                complete.append(pair)
        return complete, exclusions

    pairs, pair_exclusions = build_pairs(
        threshold_records[args.min_run_rows], args.min_run_rows
    )
    threshold_pairs: dict[int, list[dict[str, Any]]] = {}
    threshold_exclusions: dict[int, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        threshold_pairs[threshold], threshold_exclusions[threshold] = build_pairs(
            threshold_records[threshold], threshold
        )

    paired_summaries = {
        metric: pair_summary(pairs, metric) for metric in PAIR_METRICS
    }
    threshold_sensitivity = {
        str(threshold): {
            "n_complete_pairs": len(threshold_pairs[threshold]),
            "n_excluded_pairs": len(threshold_exclusions[threshold]),
            "metrics": {
                metric: pair_summary(threshold_pairs[threshold], metric)
                for metric in (
                    "stable_row_fraction",
                    "stable_run_count",
                    "median_stable_run_rows",
                    "median_stable_dispersion_px",
                    "median_stable_endpoint_amplitude_px",
                    "median_stable_path_length_px",
                )
            },
        }
        for threshold in thresholds
    }

    attribution: dict[str, Any] = {}
    attribution_fold_rows: list[dict[str, Any]] = []
    for feature_set, feature_names in FEATURE_SETS.items():
        results, folds = run_attribution(pairs, feature_names, feature_set)
        attribution[feature_set] = {
            "features": list(feature_names),
            **results,
        }
        attribution_fold_rows.extend(folds)

    x_min = min(row["x_min"] for row in per_record_rows)
    x_max = max(row["x_max"] for row in per_record_rows)
    y_min = min(row["y_min"] for row in per_record_rows)
    y_max = max(row["y_max"] for row in per_record_rows)
    duration_min = min(
        row["duration_min"]
        for row in per_record_rows
        if row["duration_min"] is not None
    )
    duration_max = max(
        row["duration_max"]
        for row in per_record_rows
        if row["duration_max"] is not None
    )

    status = "complete" if not missing_files and not case_count_mismatches else "incomplete_input"
    summary = {
        "status": status,
        "analysis": "Independent all-reader EyeAssist-PE slice-stable sensitivity",
        "seed": SEED,
        "inputs": {
            "gaze_root": str(args.gaze_root),
            "expected_files": 14,
            "found_files": len(file_qc),
            "missing_files": missing_files,
            "file_qc": file_qc,
            "case_count_mismatches": case_count_mismatches,
        },
        "qc": {
            "min_record_rows": args.min_record_rows,
            "primary_min_run_rows": args.min_run_rows,
            "short_or_truncated_record_count": len(short_records),
            "short_or_truncated_records": short_records,
            "complete_reader_case_pairs": len(pairs),
            "excluded_reader_case_pairs": len(pair_exclusions),
            "pair_exclusions": pair_exclusions,
            "coordinate_range": {"x": [x_min, x_max], "y": [y_min, y_max]},
            "duration_range": [duration_min, duration_max],
            "any_subsecond_timestamp": any(
                row["timestamp_has_subsecond"] for row in per_record_rows
            ),
        },
        "definition": {
            "slice_stable_run": "maximal consecutive CSV rows with identical Slice Number",
            "run_length_unit": "rows, not milliseconds",
            "pairing": "reader and case; both sessions required and QC-passing",
        },
        "paired_summaries": paired_summaries,
        "minimum_run_threshold_sensitivity": threshold_sensitivity,
        "session_attribution": attribution,
        "interpretation_boundary": (
            "This slice-stable analysis excludes immediate slice transitions but does not "
            "reconstruct the original I-VT event durations without precise native timestamps."
        ),
    }
    summary = json_safe(summary)

    stem = "pe_allreader_slice_stable"
    json_path = args.out / f"{stem}_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(args.out / f"{stem}_per_record.csv", per_record_rows)
    write_csv(args.out / f"{stem}_paired_differences.csv", pairs)
    write_csv(args.out / f"{stem}_attribution_folds.csv", attribution_fold_rows)
    write_csv(args.out / f"{stem}_pair_exclusions.csv", pair_exclusions)

    console = {
        "status": status,
        "found_files": len(file_qc),
        "short_or_truncated_records": len(short_records),
        "complete_reader_case_pairs": len(pairs),
        "key_paired_results": {
            metric: {
                "pooled_pair_median_difference_s2_minus_s1": paired_summaries[metric][
                    "pooled_pair_median_difference_s2_minus_s1"
                ],
                "mean_of_reader_medians": paired_summaries[metric][
                    "mean_of_reader_medians"
                ],
                "reader_sign_flip_p": paired_summaries[metric][
                    "two_sided_exact_reader_sign_flip_p"
                ],
            }
            for metric in (
                "slice_transitions",
                "stable_row_fraction",
                "median_stable_dispersion_px",
                "median_stable_endpoint_amplitude_px",
                "median_stable_path_length_px",
            )
        },
        "session_attribution": {
            feature_set: {
                cv: {
                    "pooled_accuracy": result[cv]["pooled_accuracy"],
                    "pooled_auroc": result[cv]["pooled_auroc"],
                    "n_folds": result[cv]["fold_summary"]["n_folds"],
                }
                for cv in ("LOCO", "LORO")
            }
            for feature_set, result in attribution.items()
        },
        "summary_json": str(json_path),
    }
    print(json.dumps(json_safe(console), indent=2, allow_nan=False))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
