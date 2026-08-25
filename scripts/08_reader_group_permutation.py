#!/usr/bin/env python3
"""Exact reader-label permutation for the EyeAssist-Neo reader-group axis.

The script holds the ten observed reader identities and 75 cases fixed, then
enumerates all 252 allocations of five readers to the target-labelled group.
For each allocation it recomputes (1) the directional pooled-NSS residual at
held-out target-group fixations and (2) the symmetric fixed-pool
matched-minus-opposite log-score contrast.  It also reports the five
anonymized reader contributions under the observed allocation.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from eyeassist.gaze import density_map
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


def empirical_two_sided(null, observed):
    null = np.asarray(null, dtype=float)
    return float(np.mean(np.abs(null) >= abs(observed) - 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expertise-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reader_files = []
    for group, prefix in [("Expert", "expert"), ("General", "generalist")]:
        for number in range(1, 6):
            reader_files.append((f"{prefix}{number}", args.expertise_root / group / f"{prefix}{number}_fixations.csv"))
    readers = [name for name, _ in reader_files]
    observed_target = tuple(range(5))
    manifest = read_manifest(args.manifest).set_index("case_id")
    tables = {name: read_fixations(path, COLUMNS, reader_id=name) for name, path in reader_files}
    cases = sorted(set.intersection(*[set(table.case_id) for table in tables.values()]))

    # Case-level score caches keyed by (target reader index, source-reader tuple).
    nss_cache, log_cache = {}, {}
    for case in cases:
        meta = manifest.loc[str(case)]
        shape = (int(meta.height), int(meta.width))
        maps = []
        target_events = []
        for reader in readers:
            event = tables[reader][tables[reader].case_id == case]
            target_events.append(event)
            maps.append(density_map(event, shape, downsample_factor=4, sigma_pixels=40,
                                    smoothing_mass=0.01, weighting="duration", coordinate_policy="clip"))
        stack = np.stack(maps)
        flattened = stack.reshape(len(readers), -1)
        gram = flattened @ flattened.T
        n_cells = flattened.shape[1]
        uniform_mean = 1.0 / n_cells
        for target, event in enumerate(target_events):
            xi = np.clip((event.x.to_numpy(float) / 4).astype(int), 0, stack.shape[2] - 1)
            yi = np.clip((event.y.to_numpy(float) / 4).astype(int), 0, stack.shape[1] - 1)
            sampled = stack[:, yi, xi]
            others = [idx for idx in range(len(readers)) if idx != target]
            for size in (4, 5):
                for subset in combinations(others, size):
                    subset_arr = np.asarray(subset, dtype=int)
                    probability = sampled[subset_arr].mean(axis=0)
                    mean_square = float(gram[np.ix_(subset_arr, subset_arr)].sum() / (size * size * n_cells))
                    sd = float(np.sqrt(max(mean_square - uniform_mean * uniform_mean, 1e-18)))
                    nss_cache[(case, target, subset)] = float(np.mean((probability - uniform_mean) / sd))
                    log_cache[(case, target, subset)] = float(np.mean(np.log2(np.clip(probability, 1e-300, None))))

    allocations = list(combinations(range(10), 5))
    allocation_rows = []
    reader_observed_rows = {idx: [] for idx in observed_target}
    for target_group in allocations:
        target_group = tuple(target_group)
        other_group = tuple(idx for idx in range(10) if idx not in target_group)
        directional_case, symmetric_case = [], []
        target_group_bits_case, other_group_bits_case = [], []
        for case in cases:
            directional_targets, symmetric_targets = [], []
            target_group_bits, other_group_bits = [], []
            for target in range(10):
                group = target_group if target in target_group else other_group
                opposite = other_group if target in target_group else target_group
                same = tuple(idx for idx in group if idx != target)
                opposite_subsets = list(combinations(opposite, 4))
                matched_log = log_cache[(case, target, same)]
                opposite_log = float(np.mean([log_cache[(case, target, subset)] for subset in opposite_subsets]))
                bit_contrast = matched_log - opposite_log
                symmetric_targets.append(bit_contrast)
                if target in target_group:
                    target_group_bits.append(bit_contrast)
                else:
                    other_group_bits.append(bit_contrast)
                if target in target_group:
                    directional = nss_cache[(case, target, same)] - nss_cache[(case, target, opposite)]
                    directional_targets.append(directional)
                    if target_group == observed_target:
                        reader_observed_rows[target].append(directional)
            directional_case.append(float(np.mean(directional_targets)))
            symmetric_case.append(float(np.mean(symmetric_targets)))
            target_group_bits_case.append(float(np.mean(target_group_bits)))
            other_group_bits_case.append(float(np.mean(other_group_bits)))
        allocation_rows.append({
            "target_group_indices": list(target_group),
            "directional_pooled_nss_residual": float(np.mean(directional_case)),
            "symmetric_matched_minus_opposite_bits": float(np.mean(symmetric_case)),
            "directional_case_values": directional_case,
            "symmetric_case_values": symmetric_case,
            "target_group_bits_case_values": target_group_bits_case,
            "other_group_bits_case_values": other_group_bits_case,
        })

    observed = next(row for row in allocation_rows if tuple(row["target_group_indices"]) == observed_target)
    reverse = next(row for row in allocation_rows if tuple(row["target_group_indices"]) == tuple(range(5, 10)))
    directional_null = [row["directional_pooled_nss_residual"] for row in allocation_rows]
    symmetric_null = [row["symmetric_matched_minus_opposite_bits"] for row in allocation_rows]
    payload = {
        "n_cases": len(cases),
        "n_readers": len(readers),
        "n_allocations": len(allocations),
        "observed_target_group": [readers[idx] for idx in observed_target],
        "observed": {
            "directional_pooled_nss_residual": observed["directional_pooled_nss_residual"],
            "directional_exact_two_sided_p": empirical_two_sided(directional_null, observed["directional_pooled_nss_residual"]),
            "symmetric_matched_minus_opposite_bits": observed["symmetric_matched_minus_opposite_bits"],
            "symmetric_exact_two_sided_p": empirical_two_sided(symmetric_null, observed["symmetric_matched_minus_opposite_bits"]),
        },
        "direction_specific_case_bootstrap": {
            "subspecialist_target_nss_residual": dict(zip(
                ["estimate", "ci_low", "ci_high"],
                percentile_bootstrap_mean(np.asarray(observed["directional_case_values"]), n_resamples=5000, seed=20260824),
            )),
            "general_radiologist_target_nss_residual": dict(zip(
                ["estimate", "ci_low", "ci_high"],
                percentile_bootstrap_mean(np.asarray(reverse["directional_case_values"]), n_resamples=5000, seed=20260824),
            )),
            "subspecialist_target_matched_minus_opposite_bits": dict(zip(
                ["estimate", "ci_low", "ci_high"],
                percentile_bootstrap_mean(np.asarray(observed["target_group_bits_case_values"]), n_resamples=5000, seed=20260824),
            )),
            "general_radiologist_target_matched_minus_opposite_bits": dict(zip(
                ["estimate", "ci_low", "ci_high"],
                percentile_bootstrap_mean(np.asarray(observed["other_group_bits_case_values"]), n_resamples=5000, seed=20260824),
            )),
            "symmetric_matched_minus_opposite_bits": dict(zip(
                ["estimate", "ci_low", "ci_high"],
                percentile_bootstrap_mean(np.asarray(observed["symmetric_case_values"]), n_resamples=5000, seed=20260824),
            )),
        },
        "observed_target_reader_contributions": {
            readers[idx]: {
                "mean_directional_nss_residual": float(np.mean(values)),
                "positive_cases": int(np.sum(np.asarray(values) > 0)),
            }
            for idx, values in reader_observed_rows.items()
        },
        "null_summary": {
            "directional_pooled_nss_residual": [float(x) for x in np.percentile(directional_null, [0, 2.5, 50, 97.5, 100])],
            "symmetric_matched_minus_opposite_bits": [float(x) for x in np.percentile(symmetric_null, [0, 2.5, 50, 97.5, 100])],
        },
        "allocation_rows": allocation_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: value for key, value in payload.items() if key != "allocation_rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
