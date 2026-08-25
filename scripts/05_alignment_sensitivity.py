#!/usr/bin/env python3
"""Global versus leave-one-case-out session-alignment sensitivity.

The analysis uses the packaged EyeAssist-Neo session fixation CSVs. It reports
raw and residual centre-of-mass relocation and reconstructs session-axis
finite-panel log-score contrasts under both offset estimators. Both estimators
use the same case-specific dimensions from the image manifest, matching the
primary Fig. 5 density-score pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eyeassist.gaze import (
    center_of_mass,
    density_map,
    equal_reader_pool,
    fixation_log_score,
    leave_one_case_out_offsets,
    reader_offset,
    translate_fixations,
)
from eyeassist.io import read_fixations, read_manifest


COLUMNS = {
    "case": "Source_File",
    "x": "CURRENT_FIX_X",
    "y": "CURRENT_FIX_Y",
    "duration": "CURRENT_FIX_DURATION",
    "fixation_index": "CURRENT_FIX_INDEX",
    "reader": "RECORDING_SESSION_LABEL",
}


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "iqr": [float(x) for x in np.percentile(values, [25, 75])],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-1", type=Path, required=True)
    parser.add_argument("--session-2", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/alignment_sensitivity.json"))
    args = parser.parse_args()

    manifest = read_manifest(args.manifest).set_index("case_id")

    first = {}
    second = {}
    for reader in range(1, 4):
        reader_id = f"reader{reader}"
        first[reader_id] = read_fixations(
            args.session_1 / f"reader{reader}_fixations.csv", COLUMNS, reader_id=reader_id
        )
        second[reader_id] = read_fixations(
            args.session_2 / f"reader{reader}_fixations.csv", COLUMNS, reader_id=reader_id
        )

    cases = sorted(set.intersection(*[set(table.case_id) for table in [*first.values(), *second.values()]]))
    missing_dimensions = sorted(set(cases) - set(manifest.index.astype(str)))
    if missing_dimensions:
        raise ValueError(f"Manifest lacks dimensions for cases: {missing_dimensions[:5]}")
    global_offset = {reader: reader_offset(first[reader], second[reader]) for reader in first}
    crossfit_offset = {
        reader: leave_one_case_out_offsets(first[reader], second[reader]) for reader in first
    }

    relocation = {"raw": [], "global": [], "leave_one_case_out": []}
    relocation_by_reader = {
        key: {reader: [] for reader in first} for key in relocation
    }
    per_case_scores = {"global": [], "leave_one_case_out": []}

    for case in cases:
        case_meta = manifest.loc[str(case)]
        image_shape = (int(case_meta.height), int(case_meta.width))
        aligned = {"global": {}, "leave_one_case_out": {}}
        for reader in first:
            first_case = first[reader][first[reader].case_id == case]
            second_case = second[reader][second[reader].case_id == case]
            raw = np.linalg.norm(center_of_mass(second_case) - center_of_mass(first_case))
            global_second = translate_fixations(second_case, global_offset[reader])
            crossfit_second = translate_fixations(second_case, crossfit_offset[reader][case])
            global_value = np.linalg.norm(center_of_mass(global_second) - center_of_mass(first_case))
            crossfit_value = np.linalg.norm(center_of_mass(crossfit_second) - center_of_mass(first_case))
            for key, value in [("raw", raw), ("global", global_value), ("leave_one_case_out", crossfit_value)]:
                relocation[key].append(value)
                relocation_by_reader[key][reader].append(value)
            aligned["global"][("session_1", reader)] = first_case
            aligned["global"][("session_2", reader)] = global_second
            aligned["leave_one_case_out"][("session_1", reader)] = first_case
            aligned["leave_one_case_out"][("session_2", reader)] = crossfit_second

        for estimator in ["global", "leave_one_case_out"]:
            tables = aligned[estimator]
            maps = {
                key: density_map(
                    table,
                    image_shape,
                    downsample_factor=4,
                    sigma_pixels=40,
                    smoothing_mass=0.01,
                    weighting="duration",
                    coordinate_policy="clip",
                )
                for key, table in tables.items()
            }
            contrasts = []
            for state in ["session_1", "session_2"]:
                opposite = "session_2" if state == "session_1" else "session_1"
                for target_reader in first:
                    target = tables[(state, target_reader)]
                    others = [reader for reader in first if reader != target_reader]
                    matched = equal_reader_pool(maps, [(state, reader) for reader in others])
                    off_state = equal_reader_pool(maps, [(opposite, reader) for reader in others])
                    half_scores = []
                    for same_reader in others:
                        for off_reader in others:
                            if same_reader == off_reader:
                                continue
                            half = equal_reader_pool(
                                maps, [(state, same_reader), (opposite, off_reader)]
                            )
                            half_scores.append(fixation_log_score(target, half))
                    matched_score = fixation_log_score(target, matched)
                    all_records = equal_reader_pool(
                        maps,
                        [(session, reader) for session in [state, opposite] for reader in others],
                    )
                    contrasts.append(
                        {
                            "matched_minus_half_mixed": matched_score - float(np.mean(half_scores)),
                            "matched_minus_opposite": matched_score - fixation_log_score(target, off_state),
                            "all_records_minus_matched": fixation_log_score(target, all_records) - matched_score,
                        }
                    )
            per_case_scores[estimator].append(
                {key: float(np.mean([row[key] for row in contrasts])) for key in contrasts[0]}
            )

    payload = {
        "n_cases": len(cases),
        "analysis_role": "case_specific_alignment_sensitivity",
        "canvas_source": "manifest_width_height",
        "canvas_width_range": [int(manifest.loc[cases, "width"].min()), int(manifest.loc[cases, "width"].max())],
        "canvas_height_range": [int(manifest.loc[cases, "height"].min()), int(manifest.loc[cases, "height"].max())],
        "global_offsets": {reader: [float(x) for x in value] for reader, value in global_offset.items()},
        "relocation": {
            key: {
                **summarize(values),
                "per_reader_medians": {
                    reader: float(np.median(reader_values))
                    for reader, reader_values in relocation_by_reader[key].items()
                },
            }
            for key, values in relocation.items()
        },
        "density_score": {},
    }
    for estimator, rows in per_case_scores.items():
        frame = pd.DataFrame(rows)
        payload["density_score"][estimator] = {
            column: {
                "mean": float(frame[column].mean()),
                "positive_cases": int((frame[column] > 0).sum()),
            }
            for column in frame.columns
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
