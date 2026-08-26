#!/usr/bin/env python3
"""Read-only structural audit of the public GazeVaLM scanpath release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_fixed_pool import TASKS, find_task_roots, stimulus_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = find_task_roots(args.data_root)
    report = {"task_roots": {k: str(v) for k, v in roots.items()}, "tasks": {}}
    stimulus_sets = {}
    all_reader_sets = {}
    for task in TASKS:
        csvs = sorted(roots[task].glob("*/scanpaths.csv"))
        stimulus_sets[task] = {p.parent.name for p in csvs}
        rows = []
        reader_sets = {}
        for path in csvs:
            frame = pd.read_csv(path)
            required = {"participant_id", "x", "y", "duration_ms"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{path}: missing {missing}")
            ids = sorted(frame["participant_id"].dropna().astype(str).str.strip().unique())
            reader_sets[path.parent.name] = set(ids)
            numeric = frame[["x", "y", "duration_ms"]].apply(pd.to_numeric, errors="coerce")
            rows.append(
                {
                    "stimulus": path.parent.name,
                    "authenticity": stimulus_metadata(path.parent.name)[0],
                    "study_id": stimulus_metadata(path.parent.name)[1],
                    "n_rows": int(len(frame)),
                    "n_readers": len(ids),
                    "readers": ids,
                    "x_min": float(numeric["x"].min()),
                    "x_max": float(numeric["x"].max()),
                    "y_min": float(numeric["y"].min()),
                    "y_max": float(numeric["y"].max()),
                    "duration_min": float(numeric["duration_ms"].min()),
                    "duration_max": float(numeric["duration_ms"].max()),
                    "nonfinite_numeric": int((~np.isfinite(numeric)).any(axis=1).sum()),
                    "nonpositive_duration": int((numeric["duration_ms"] <= 0).sum()),
                    "duplicate_rows": int(frame.duplicated().sum()),
                    "columns": list(frame.columns),
                }
            )
        all_reader_sets[task] = reader_sets
        union = sorted(set().union(*reader_sets.values())) if reader_sets else []
        intersection = sorted(set.intersection(*reader_sets.values())) if reader_sets else []
        report["tasks"][task] = {
            "n_stimuli": len(csvs),
            "n_real": sum(x["authenticity"] == "real" for x in rows),
            "n_fake": sum(x["authenticity"] == "fake" for x in rows),
            "total_rows": sum(x["n_rows"] for x in rows),
            "reader_union": union,
            "reader_intersection": intersection,
            "n_reader_union": len(union),
            "n_reader_intersection": len(intersection),
            "n_stimuli_with_nonfinite_numeric": sum(x["nonfinite_numeric"] > 0 for x in rows),
            "n_stimuli_with_nonpositive_duration": sum(x["nonpositive_duration"] > 0 for x in rows),
            "n_stimuli_with_duplicate_rows": sum(x["duplicate_rows"] > 0 for x in rows),
            "coordinate_range": {
                "x_min": min(x["x_min"] for x in rows),
                "x_max": max(x["x_max"] for x in rows),
                "y_min": min(x["y_min"] for x in rows),
                "y_max": max(x["y_max"] for x in rows),
            },
            "columns": sorted(set().union(*(set(x["columns"]) for x in rows))),
            "records": rows,
        }

    common = sorted(stimulus_sets["Task1"] & stimulus_sets["Task2"])
    task_crossing = []
    for stimulus in common:
        a = all_reader_sets["Task1"][stimulus]
        b = all_reader_sets["Task2"][stimulus]
        task_crossing.append(
            {
                "stimulus": stimulus,
                "n_Task1": len(a),
                "n_Task2": len(b),
                "n_intersection": len(a & b),
                "same_reader_set": a == b,
            }
        )
    study_to_auth = {}
    for stimulus in common:
        auth, study = stimulus_metadata(stimulus)
        study_to_auth.setdefault(study, set()).add(auth)
    report["cross_task"] = {
        "n_common_stimuli": len(common),
        "Task1_only": sorted(stimulus_sets["Task1"] - stimulus_sets["Task2"]),
        "Task2_only": sorted(stimulus_sets["Task2"] - stimulus_sets["Task1"]),
        "n_same_reader_set": sum(x["same_reader_set"] for x in task_crossing),
        "minimum_reader_intersection": min(x["n_intersection"] for x in task_crossing),
        "n_source_study_ids": len(study_to_auth),
        "n_complete_real_fake_pairs": sum(v == {"real", "fake"} for v in study_to_auth.values()),
        "records": task_crossing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    compact = {"task_roots": report["task_roots"], "tasks": {}, "cross_task": {k: v for k, v in report["cross_task"].items() if k != "records"}}
    for task in TASKS:
        compact["tasks"][task] = {k: v for k, v in report["tasks"][task].items() if k != "records"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
