#!/usr/bin/env python3
"""All-three-of-five sensitivity analysis for the subspecialist reader panel.

Every three-reader subset of the five-member subspecialist panel is compared
with the five general radiologists using equal two-reader references.  The
script also performs an exact three-versus-five label permutation within each
retained eight-reader panel.
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


def exact_two_sided(null: list[float], observed: float) -> float:
    values = np.asarray(null, dtype=float)
    return float(np.mean(np.abs(values) >= abs(observed) - 1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expertise-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    reader_files: list[tuple[str, Path]] = []
    for group, prefix in (("Expert", "expert"), ("General", "generalist")):
        for number in range(1, 6):
            reader_files.append(
                (f"{prefix}{number}", args.expertise_root / group / f"{prefix}{number}_fixations.csv")
            )
    readers = [name for name, _ in reader_files]
    manifest = read_manifest(args.manifest).set_index("case_id")
    tables = {
        name: read_fixations(path, COLUMNS, reader_id=name) for name, path in reader_files
    }
    cases = sorted(set.intersection(*[set(table.case_id) for table in tables.values()]))

    nss_cache: dict[tuple[str, int, tuple[int, int]], float] = {}
    log_cache: dict[tuple[str, int, tuple[int, int]], float] = {}
    for case in cases:
        meta = manifest.loc[str(case)]
        shape = (int(meta.height), int(meta.width))
        maps = []
        targets = []
        for reader in readers:
            event = tables[reader][tables[reader].case_id == case]
            targets.append(event)
            maps.append(
                density_map(
                    event,
                    shape,
                    downsample_factor=4,
                    sigma_pixels=40,
                    smoothing_mass=0.01,
                    weighting="duration",
                    coordinate_policy="clip",
                )
            )
        stack = np.stack(maps)
        flat = stack.reshape(len(readers), -1)
        gram = flat @ flat.T
        n_cells = flat.shape[1]
        uniform_mean = 1.0 / n_cells
        for target, event in enumerate(targets):
            xi = np.clip((event.x.to_numpy(float) / 4).astype(int), 0, stack.shape[2] - 1)
            yi = np.clip((event.y.to_numpy(float) / 4).astype(int), 0, stack.shape[1] - 1)
            sampled = stack[:, yi, xi]
            for subset in combinations([i for i in range(10) if i != target], 2):
                index = np.asarray(subset, dtype=int)
                probability = sampled[index].mean(axis=0)
                mean_square = float(gram[np.ix_(index, index)].sum() / (4 * n_cells))
                sd = float(np.sqrt(max(mean_square - uniform_mean**2, 1e-18)))
                nss_cache[(case, target, subset)] = float(
                    np.mean((probability - uniform_mean) / sd)
                )
                log_cache[(case, target, subset)] = float(
                    np.mean(np.log2(np.clip(probability, 1e-300, None)))
                )

    def allocation_statistics(panel: tuple[int, ...], target_group: tuple[int, ...]):
        target_set = set(target_group)
        other_group = tuple(i for i in panel if i not in target_set)
        directional_cases: list[float] = []
        symmetric_cases: list[float] = []
        target_bits_cases: list[float] = []
        other_bits_cases: list[float] = []
        reader_directional: dict[int, list[float]] = {i: [] for i in target_group}
        for case in cases:
            directional_targets = []
            all_bits = []
            target_bits = []
            other_bits = []
            for target in panel:
                if target in target_set:
                    same_group = tuple(i for i in target_group if i != target)
                    same_subsets = [same_group]
                    opposite_subsets = list(combinations(other_group, 2))
                else:
                    same_group = tuple(i for i in other_group if i != target)
                    same_subsets = list(combinations(same_group, 2))
                    opposite_subsets = list(combinations(target_group, 2))
                matched_log = float(
                    np.mean([log_cache[(case, target, tuple(s))] for s in same_subsets])
                )
                opposite_log = float(
                    np.mean([log_cache[(case, target, tuple(s))] for s in opposite_subsets])
                )
                contrast = matched_log - opposite_log
                all_bits.append(contrast)
                if target in target_set:
                    target_bits.append(contrast)
                    matched_nss = nss_cache[(case, target, tuple(same_subsets[0]))]
                    opposite_nss = float(
                        np.mean([nss_cache[(case, target, tuple(s))] for s in opposite_subsets])
                    )
                    residual = matched_nss - opposite_nss
                    directional_targets.append(residual)
                    reader_directional[target].append(residual)
                else:
                    other_bits.append(contrast)
            directional_cases.append(float(np.mean(directional_targets)))
            symmetric_cases.append(float(np.mean(all_bits)))
            target_bits_cases.append(float(np.mean(target_bits)))
            other_bits_cases.append(float(np.mean(other_bits)))
        return {
            "directional_cases": directional_cases,
            "symmetric_cases": symmetric_cases,
            "target_bits_cases": target_bits_cases,
            "other_bits_cases": other_bits_cases,
            "reader_directional": reader_directional,
        }

    rows = []
    for candidate in combinations(range(5), 3):
        panel = tuple(candidate) + tuple(range(5, 10))
        observed = allocation_statistics(panel, tuple(candidate))
        null = []
        for allocation in combinations(panel, 3):
            stats = allocation_statistics(panel, tuple(allocation))
            null.append(float(np.mean(stats["symmetric_cases"])))
        symmetric = percentile_bootstrap_mean(
            np.asarray(observed["symmetric_cases"]),
            n_resamples=args.bootstrap,
            seed=args.seed,
        )
        directional = percentile_bootstrap_mean(
            np.asarray(observed["directional_cases"]),
            n_resamples=args.bootstrap,
            seed=args.seed,
        )
        target_bits = percentile_bootstrap_mean(
            np.asarray(observed["target_bits_cases"]),
            n_resamples=args.bootstrap,
            seed=args.seed,
        )
        other_bits = percentile_bootstrap_mean(
            np.asarray(observed["other_bits_cases"]),
            n_resamples=args.bootstrap,
            seed=args.seed,
        )
        rows.append(
            {
                "retained_subspecialist_readers": [readers[i] for i in candidate],
                "excluded_subspecialist_readers": [
                    readers[i] for i in range(5) if i not in candidate
                ],
                "directional_pooled_nss_residual": dict(
                    zip(("estimate", "ci_low", "ci_high"), map(float, directional))
                ),
                "symmetric_matched_minus_opposite_bits": dict(
                    zip(("estimate", "ci_low", "ci_high"), map(float, symmetric))
                ),
                "subspecialist_target_bits": dict(
                    zip(("estimate", "ci_low", "ci_high"), map(float, target_bits))
                ),
                "general_radiologist_target_bits": dict(
                    zip(("estimate", "ci_low", "ci_high"), map(float, other_bits))
                ),
                "exact_three_vs_five_p": exact_two_sided(null, float(symmetric[0])),
                "reader_directional_contributions": {
                    readers[i]: float(np.mean(values))
                    for i, values in observed["reader_directional"].items()
                },
            }
        )

    payload = {
        "analysis": "all three-of-five subspecialist subsets versus five general radiologists",
        "n_cases": len(cases),
        "reference_size": 2,
        "n_candidate_triplets": len(rows),
        "bootstrap_resamples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "subgroup_results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    compact = [
        {
            "triplet": row["retained_subspecialist_readers"],
            "symmetric_bits": row["symmetric_matched_minus_opposite_bits"]["estimate"],
            "ci": [
                row["symmetric_matched_minus_opposite_bits"]["ci_low"],
                row["symmetric_matched_minus_opposite_bits"]["ci_high"],
            ],
            "exact_p": row["exact_three_vs_five_p"],
        }
        for row in rows
    ]
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
