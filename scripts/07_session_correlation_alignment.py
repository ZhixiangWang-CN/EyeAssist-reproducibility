#!/usr/bin/env python3
"""Symmetric alignment sensitivity for EyeAssist-Neo inter-reader map correlation.

The primary manuscript comparison translated session-2 coordinates by each
reader's mean paired session displacement.  This sensitivity analysis reports
the session contrast (i) without alignment, (ii) under that primary rule and
(iii) after applying the same within-session reader-centering rule to both
sessions.  The case remains the paired inference unit throughout.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from eyeassist.gaze import center_of_mass, density_map, pearson_cc, reader_offset, translate_fixations
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


def reader_mean_centroid(table):
    centroids = [center_of_mass(group) for _, group in table.groupby("case_id")]
    return np.nanmean(np.asarray(centroids, dtype=float), axis=0)


def within_session_centering_offsets(tables):
    centroids = {reader: reader_mean_centroid(table) for reader, table in tables.items()}
    grand = np.mean(np.stack(list(centroids.values())), axis=0)
    offsets = {reader: centroid - grand for reader, centroid in centroids.items()}
    return offsets, centroids, grand


def reader_centroid_spread(centroids):
    values = np.stack(list(centroids.values()))
    pairwise = [float(np.linalg.norm(values[i] - values[j])) for i, j in combinations(range(len(values)), 2)]
    return {
        "x_range_px": float(np.ptp(values[:, 0])),
        "y_range_px": float(np.ptp(values[:, 1])),
        "mean_pairwise_distance_px": float(np.mean(pairwise)),
        "pairwise_distances_px": pairwise,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-1", type=Path, required=True)
    parser.add_argument("--session-2", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest).set_index("case_id")
    first, second = {}, {}
    for number in range(1, 4):
        reader = f"reader{number}"
        first[reader] = read_fixations(args.session_1 / f"{reader}_fixations.csv", COLUMNS, reader_id=reader)
        second[reader] = read_fixations(args.session_2 / f"{reader}_fixations.csv", COLUMNS, reader_id=reader)

    cases = sorted(set.intersection(*[set(table.case_id) for table in [*first.values(), *second.values()]]))
    primary_offsets = {reader: reader_offset(first[reader], second[reader]) for reader in first}
    centre_first, centroids_first, grand_first = within_session_centering_offsets(first)
    centre_second, centroids_second, grand_second = within_session_centering_offsets(second)

    conditions = {
        "unaligned": {
            "session_1": first,
            "session_2": second,
        },
        "primary_session2_only": {
            "session_1": first,
            "session_2": {reader: translate_fixations(second[reader], primary_offsets[reader]) for reader in second},
        },
        "symmetric_within_session_centering": {
            "session_1": {reader: translate_fixations(first[reader], centre_first[reader]) for reader in first},
            "session_2": {reader: translate_fixations(second[reader], centre_second[reader]) for reader in second},
        },
    }

    results = {}
    for name, sessions in conditions.items():
        case_rows = []
        for case in cases:
            meta = manifest.loc[str(case)]
            shape = (int(meta.height), int(meta.width))
            means = {}
            for session_name, tables in sessions.items():
                maps = {}
                for reader, table in tables.items():
                    case_table = table[table.case_id == case]
                    maps[reader] = density_map(
                        case_table,
                        shape,
                        downsample_factor=4,
                        sigma_pixels=40,
                        smoothing_mass=0.01,
                        weighting="duration",
                        coordinate_policy="clip",
                    )
                means[session_name] = float(np.mean([
                    pearson_cc(maps[a], maps[b]) for a, b in combinations(sorted(maps), 2)
                ]))
            case_rows.append({
                "case_id": str(case),
                "session_1": means["session_1"],
                "session_2": means["session_2"],
                "difference": means["session_2"] - means["session_1"],
            })
        delta = np.asarray([row["difference"] for row in case_rows])
        estimate, low, high = percentile_bootstrap_mean(delta, n_resamples=200_000, seed=20260822)
        results[name] = {
            "session_1_mean": float(np.mean([row["session_1"] for row in case_rows])),
            "session_2_mean": float(np.mean([row["session_2"] for row in case_rows])),
            "difference": estimate,
            "percentile_95_interval": [low, high],
            "positive_cases": int(np.sum(delta > 0)),
            "case_rows": case_rows,
        }

    payload = {
        "n_cases": len(cases),
        "n_readers": len(first),
        "map_rule": "duration weighted; sigma 40 native px; fourfold downsampling; 1% uniform floor",
        "primary_session2_offsets_px": {reader: [float(x) for x in value] for reader, value in primary_offsets.items()},
        "raw_reader_centroids_px": {
            "session_1": {reader: [float(x) for x in value] for reader, value in centroids_first.items()},
            "session_2": {reader: [float(x) for x in value] for reader, value in centroids_second.items()},
        },
        "raw_reader_centroid_spread": {
            "session_1": reader_centroid_spread(centroids_first),
            "session_2": reader_centroid_spread(centroids_second),
        },
        "within_session_grand_centroids_px": {
            "session_1": [float(x) for x in grand_first],
            "session_2": [float(x) for x in grand_second],
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"n_cases": payload["n_cases"], "spread": payload["raw_reader_centroid_spread"], "results": {k: {kk: vv for kk, vv in v.items() if kk != "case_rows"} for k, v in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
