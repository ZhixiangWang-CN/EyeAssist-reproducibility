#!/usr/bin/env python3
"""Uniform-floor sensitivity for the six EyeAssist finite-panel contrasts."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from eyeassist.gaze import density_map, reader_offset, translate_fixations
from eyeassist.io import read_fixations, read_manifest
from eyeassist.statistics import percentile_bootstrap_mean


COLUMNS = {
    "case": "Source_File",
    "x": "CURRENT_FIX_X",
    "y": "CURRENT_FIX_Y",
    "duration": "CURRENT_FIX_DURATION",
    "fixation_index": "CURRENT_FIX_INDEX",
    "reader": "RECORDING_SESSION_LABEL",
}


def score_from_values(values, members):
    probability = np.mean([values[member] for member in members], axis=0)
    return float(np.mean(np.log2(np.clip(probability, 1e-300, None))))


def summarize(rows):
    keys = rows[0].keys()
    output = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        estimate, low, high = percentile_bootstrap_mean(values, n_resamples=2000, seed=20260824)
        output[key] = {
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "positive_cases": int(np.sum(values > 0)),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expertise-root", type=Path, required=True)
    parser.add_argument("--session-1", type=Path, required=True)
    parser.add_argument("--session-2", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest).set_index("case_id")
    groups = {
        "subspecialist": [f"expert{i}" for i in range(1, 6)],
        "general_radiologist": [f"generalist{i}" for i in range(1, 6)],
    }
    group_tables = {}
    for group_folder, prefix in [("Expert", "expert"), ("General", "generalist")]:
        for number in range(1, 6):
            reader = f"{prefix}{number}"
            group_tables[reader] = read_fixations(args.expertise_root / group_folder / f"{reader}_fixations.csv", COLUMNS, reader_id=reader)

    first, second = {}, {}
    for number in range(1, 4):
        reader = f"reader{number}"
        first[reader] = read_fixations(args.session_1 / f"{reader}_fixations.csv", COLUMNS, reader_id=reader)
        second[reader] = read_fixations(args.session_2 / f"{reader}_fixations.csv", COLUMNS, reader_id=reader)
    session_offsets = {reader: reader_offset(first[reader], second[reader]) for reader in first}
    second_aligned = {reader: translate_fixations(second[reader], session_offsets[reader]) for reader in second}

    cases = sorted(set.intersection(*[set(table.case_id) for table in [*group_tables.values(), *first.values(), *second.values()]]))
    output = {
        "n_cases": len(cases),
        "sigma_native_px": 40,
        "alpha_values": [0.001, 0.01, 0.05, 0.1],
        "results": {},
    }

    for alpha in output["alpha_values"]:
        reader_group_rows, session_rows = [], []
        for case in cases:
            meta = manifest.loc[str(case)]
            shape = (int(meta.height), int(meta.width))

            group_maps = {
                reader: density_map(table[table.case_id == case], shape, downsample_factor=4,
                                    sigma_pixels=40, smoothing_mass=alpha, weighting="duration",
                                    coordinate_policy="clip")
                for reader, table in group_tables.items()
            }
            group_contrasts = []
            for target, table in group_tables.items():
                event = table[table.case_id == case]
                xi = np.clip((event.x.to_numpy(float) / 4).astype(int), 0, next(iter(group_maps.values())).shape[1] - 1)
                yi = np.clip((event.y.to_numpy(float) / 4).astype(int), 0, next(iter(group_maps.values())).shape[0] - 1)
                values = {reader: member_map[yi, xi] for reader, member_map in group_maps.items()}
                target_group = groups["subspecialist"] if target.startswith("expert") else groups["general_radiologist"]
                opposite_group = groups["general_radiologist"] if target.startswith("expert") else groups["subspecialist"]
                matched_members = tuple(reader for reader in target_group if reader != target)
                matched = score_from_values(values, matched_members)
                half = np.mean([
                    score_from_values(values, same_pair + opposite_pair)
                    for same_pair in combinations(matched_members, 2)
                    for opposite_pair in combinations(opposite_group, 2)
                ])
                opposite = np.mean([score_from_values(values, subset) for subset in combinations(opposite_group, 4)])
                all_records = score_from_values(values, matched_members + tuple(opposite_group))
                group_contrasts.append({
                    "matched_minus_half_mixed": matched - half,
                    "matched_minus_opposite": matched - opposite,
                    "all_records_minus_matched": all_records - matched,
                })
            reader_group_rows.append({key: float(np.mean([row[key] for row in group_contrasts])) for key in group_contrasts[0]})

            session_tables = {
                ("session_1", reader): first[reader][first[reader].case_id == case] for reader in first
            }
            session_tables.update({
                ("session_2", reader): second_aligned[reader][second_aligned[reader].case_id == case] for reader in second_aligned
            })
            session_maps = {
                key: density_map(table, shape, downsample_factor=4, sigma_pixels=40,
                                 smoothing_mass=alpha, weighting="duration", coordinate_policy="clip")
                for key, table in session_tables.items()
            }
            session_contrasts = []
            for (state, target), event in session_tables.items():
                opposite_state = "session_2" if state == "session_1" else "session_1"
                xi = np.clip((event.x.to_numpy(float) / 4).astype(int), 0, next(iter(session_maps.values())).shape[1] - 1)
                yi = np.clip((event.y.to_numpy(float) / 4).astype(int), 0, next(iter(session_maps.values())).shape[0] - 1)
                values = {key: member_map[yi, xi] for key, member_map in session_maps.items()}
                others = tuple(reader for reader in first if reader != target)
                matched_members = tuple((state, reader) for reader in others)
                opposite_members = tuple((opposite_state, reader) for reader in others)
                matched = score_from_values(values, matched_members)
                half = np.mean([
                    score_from_values(values, ((state, same_reader), (opposite_state, off_reader)))
                    for same_reader in others for off_reader in others if same_reader != off_reader
                ])
                opposite = score_from_values(values, opposite_members)
                all_records = score_from_values(values, matched_members + opposite_members)
                session_contrasts.append({
                    "matched_minus_half_mixed": matched - half,
                    "matched_minus_opposite": matched - opposite,
                    "all_records_minus_matched": all_records - matched,
                })
            session_rows.append({key: float(np.mean([row[key] for row in session_contrasts])) for key in session_contrasts[0]})

        output["results"][str(alpha)] = {
            "reader_group": summarize(reader_group_rows),
            "session_condition": summarize(session_rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
