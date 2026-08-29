#!/usr/bin/env python3
"""Reconstruct EyeAssist saliency-transfer robustness analyses.

The script reads locally supplied execution shards, fixation CSVs and a
case-dimension manifest. Generated summaries are written to ``--output-dir``.

Analyses
--------
1. Reconstruct the complete 4 x 4 NSS/CC/KL/sAUC matrices from 17 random
   partitions plus one coverage-completion partition. Repeated predictions are
   averaged within case before the 75-case matrix is summarized.
2. Compute paired partition-bootstrap intervals for every diagonal-versus-
   off-diagonal contrast at a fixed evaluation target.
3. Reconstruct duration-weighted 256 x 256 target densities from the raw
   fixations and summarize Shannon entropy and effective support under two
   pooling rules (event-duration pooling and equal-reader pooling), with and
   without a 1% uniform floor.
4. Quantify how the published global session alignment changes the entropy of
   the second-session pooled target.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/saliency_robustness")
DATA = Path(".")
MANIFEST = Path("manifest.csv")
COMPLETION_DATA = Path(".")

REFS = ["expert_consensus", "generalist_consensus", "pre_report", "post_report"]
LABELS = {
    "expert_consensus": "Subspecialist group",
    "generalist_consensus": "General-radiologist group",
    "pre_report": "First session",
    "post_report": "Second session",
}
METRICS = ["NSS", "CC", "KLDiv", "sAUC"]
LOWER_IS_BETTER = {"NSS": False, "CC": False, "KLDiv": True, "sAUC": False}
AXIS_CONTRASTS = {
    "session_second_target": ("pre_report", "post_report"),
    "session_first_target": ("post_report", "pre_report"),
    "reader_group_subspecialty_target": ("generalist_consensus", "expert_consensus"),
    "reader_group_general_radiologist_target": ("expert_consensus", "generalist_consensus"),
}

FIXATION_SOURCES = {
    "expert_consensus": Path("."),
    "generalist_consensus": Path("."),
    "pre_report": Path("."),
    "post_report": Path("."),
}

GLOBAL_OFFSETS = {
    "reader1": (-131.18481373034274, -238.73715739294357),
    "reader2": (-135.34648019666372, -29.847081398478853),
    "reader3": (-54.3691178821593, -72.21575248244243),
}


def bootstrap_mean(values: np.ndarray, seed: int = 20260824, n: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    # Chunk to avoid allocating a large array for future larger inputs.
    draws = []
    remaining = n
    while remaining:
        size = min(2_000, remaining)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        draws.append(values[idx].mean(axis=1))
        remaining -= size
    lo, hi = np.percentile(np.concatenate(draws), [2.5, 97.5])
    return float(lo), float(hi)


def mean_metric(run: dict, trained: str, target: str, metric: str) -> float:
    values = np.asarray(run["matrix"][trained][target][metric], float)
    return float(np.nanmean(values))


def transfer_audit() -> dict:
    shards = [json.loads((DATA / f"expA_shard{i}.json").read_text()) for i in (1, 2, 3)]
    if any(len(shard["runs"]) != 17 for shard in shards):
        raise ValueError("Expected 17 partitions in every execution shard")

    completion_runs = [
        json.loads((COMPLETION_DATA / f"execution{i}.json").read_text())
        for i in (1, 2, 3)
    ]
    if any(len(run["test_cases"]) != 19 for run in completion_runs):
        raise ValueError("Coverage-completion executions must contain 19 test cases")
    if len({tuple(run["test_cases"]) for run in completion_runs}) != 1:
        raise ValueError("Coverage-completion executions do not share the test set")

    partition_groups = [
        [shard["runs"][partition] for shard in shards]
        for partition in range(17)
    ]
    partition_groups.append(completion_runs)

    partition_rows = []
    contrast_rows = []
    case_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    for partition, runs in enumerate(partition_groups):
        split_seeds = {
            int(run.get("seed", run.get("config", {}).get("split_seed"))) for run in runs
        }
        if len(split_seeds) != 1:
            raise ValueError(f"Seed mismatch in partition {partition + 1}")
        split_seed = next(iter(split_seeds))
        if len({tuple(run["test_cases"]) for run in runs}) != 1:
            raise ValueError(f"Test-case mismatch in partition {partition + 1}")

        for trained in REFS:
            for target in REFS:
                for metric in METRICS:
                    execution_means = [mean_metric(run, trained, target, metric) for run in runs]
                    partition_rows.append({
                        "partition": partition + 1,
                        "seed": split_seed,
                        "trained_on": trained,
                        "scored_against": target,
                        "metric": metric,
                        "value": float(np.mean(execution_means)),
                    })
                for run in runs:
                    cases = run["test_cases"]
                    for metric in METRICS:
                        vals = run["matrix"][trained][target][metric]
                        if len(vals) != len(cases):
                            raise ValueError("Per-case metric length does not match test cases")
                        for case, value in zip(cases, vals):
                            if np.isfinite(float(value)):
                                case_values[(str(case), trained, target, metric)].append(float(value))

    part = pd.DataFrame(partition_rows)
    part.to_csv(OUT / "2026-08-24_saliency_per_partition_matrix.csv", index=False)

    case_rows = [
        {
            "case": key[0],
            "trained_on": key[1],
            "scored_against": key[2],
            "metric": key[3],
            "mean": float(np.mean(values)),
            "appearances": len(values),
        }
        for key, values in sorted(case_values.items())
    ]
    case_frame = pd.DataFrame(case_rows)
    if case_frame["case"].nunique() != 75:
        raise ValueError(
            f"Expected held-out predictions for 75 cases, found {case_frame['case'].nunique()}"
        )
    case_frame.to_csv(OUT / "2026-08-24_saliency_per_case_matrix.csv", index=False)

    matrices = {}
    column_best = {}
    spans = {}
    for metric in METRICS:
        subset = part[part.metric == metric]
        case_subset = case_frame[case_frame.metric == metric]
        matrix = []
        for trained in REFS:
            matrix.append([
                float(
                    case_subset[
                        (case_subset.trained_on == trained)
                        & (case_subset.scored_against == target)
                    ]["mean"].mean()
                )
                for target in REFS
            ])
        matrices[metric] = matrix
        array = np.asarray(matrix)
        if LOWER_IS_BETTER[metric]:
            best = np.argmin(array, axis=0)
        else:
            best = np.argmax(array, axis=0)
        column_best[metric] = [REFS[int(x)] for x in best]
        spans[metric] = [float(array[:, j].max() - array[:, j].min()) for j in range(4)]

        # Every diagonal-versus-off-diagonal comparison at a fixed target.
        for target in REFS:
            matched = subset[(subset.trained_on == target) & (subset.scored_against == target)].sort_values("partition")
            for alternative in REFS:
                if alternative == target:
                    continue
                other = subset[(subset.trained_on == alternative) & (subset.scored_against == target)].sort_values("partition")
                if not np.array_equal(matched.partition.to_numpy(), other.partition.to_numpy()):
                    raise ValueError("Partition mismatch in paired contrast")
                if LOWER_IS_BETTER[metric]:
                    delta = other.value.to_numpy() - matched.value.to_numpy()
                else:
                    delta = matched.value.to_numpy() - other.value.to_numpy()
                lo, hi = bootstrap_mean(delta)
                contrast_rows.append({
                    "metric": metric,
                    "target": target,
                    "matched_train": target,
                    "alternative_train": alternative,
                    "advantage_positive_favours_matched": float(delta.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "positive_partitions": int((delta > 0).sum()),
                    "n_partitions": len(delta),
                })

    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(OUT / "2026-08-24_saliency_all_paired_contrasts.csv", index=False)

    axis_rows = []
    for metric in METRICS:
        for name, (alternative, target) in AXIS_CONTRASTS.items():
            row = contrasts[
                (contrasts.metric == metric)
                & (contrasts.target == target)
                & (contrasts.alternative_train == alternative)
            ].iloc[0]
            axis_rows.append({"contrast": name, **row.to_dict()})
    pd.DataFrame(axis_rows).to_csv(OUT / "2026-08-24_saliency_axis_contrasts.csv", index=False)

    return {
        "n_partitions": len(partition_groups),
        "executions_per_partition": 3,
        "n_fitted_models": len(partition_groups) * 3 * 4,
        "n_held_out_cases": int(case_frame["case"].nunique()),
        "matrices": matrices,
        "column_best_training_source": column_best,
        "within_column_spans": spans,
        "axis_contrasts": axis_rows,
    }


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    radius = max(1, int(truncate * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def convolve_axis_same(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    n = array.shape[axis]
    full = n + len(kernel) - 1
    fft_n = 1 << (full - 1).bit_length()
    transformed = np.fft.rfft(array, n=fft_n, axis=axis)
    shape = [1] * array.ndim
    shape[axis] = len(kernel)
    kernel_fft = np.fft.rfft(kernel.reshape(shape), n=fft_n, axis=axis)
    result = np.fft.irfft(transformed * kernel_fft, n=fft_n, axis=axis)
    start = (len(kernel) - 1) // 2
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, start + n)
    return result[tuple(index)]


def smooth_density(histogram: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    smoothed = convolve_axis_same(histogram, gaussian_kernel(sigma_x), axis=1)
    smoothed = convolve_axis_same(smoothed, gaussian_kernel(sigma_y), axis=0)
    smoothed = np.clip(smoothed, 0, None)
    total = smoothed.sum()
    if total <= 0:
        return np.full_like(smoothed, 1.0 / smoothed.size)
    return smoothed / total


def read_fixations() -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for reference, directory in FIXATION_SOURCES.items():
        readers = {}
        for path in sorted(directory.glob("*_fixations.csv")):
            frame = pd.read_csv(path, usecols=[
                "Source_File", "CURRENT_FIX_X", "CURRENT_FIX_Y", "CURRENT_FIX_DURATION"
            ])
            frame.columns = ["case", "x", "y", "duration"]
            frame["case"] = frame.case.astype(str).str.replace("60.jpeg", "60.jpg", regex=False)
            frame = frame.apply({"case": str, "x": pd.to_numeric, "y": pd.to_numeric, "duration": pd.to_numeric})
            frame = frame.dropna(subset=["x", "y", "duration"])
            readers[path.stem.replace("_fixations", "")] = frame
        result[reference] = readers
    return result


def histogram_for_reader(
    frame: pd.DataFrame,
    case: str,
    width: int,
    height: int,
    *,
    offset: tuple[float, float] | None = None,
    size: int = 256,
) -> np.ndarray:
    sub = frame[frame.case == case]
    x = sub.x.to_numpy(float)
    y = sub.y.to_numpy(float)
    duration = sub.duration.to_numpy(float)
    if offset is not None:
        x = x - offset[0]
        y = y - offset[1]
    xi = np.clip((x / width * size).astype(int), 0, size - 1)
    yi = np.clip((y / height * size).astype(int), 0, size - 1)
    hist = np.zeros((size, size), dtype=float)
    np.add.at(hist, (yi, xi), duration)
    return hist


def density_summaries(probability: np.ndarray) -> dict[str, float]:
    p = probability.ravel()
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    return {
        "entropy_bits": entropy,
        "normalized_entropy": entropy / 16.0,
        "effective_support_shannon_cells": float(2.0 ** entropy),
        "effective_support_simpson_cells": float(1.0 / np.sum(p * p)),
    }


def target_entropy_audit() -> dict:
    manifest = pd.read_csv(MANIFEST).set_index("case_id")
    fixations = read_fixations()
    common = sorted(set(manifest.index.astype(str)).intersection(*[
        set.intersection(*[set(frame.case) for frame in readers.values()])
        for readers in fixations.values()
    ]))
    if len(common) != 75:
        raise ValueError(f"Expected 75 common cases, found {len(common)}")

    rows = []
    for case in common:
        width = int(manifest.loc[case, "width"])
        height = int(manifest.loc[case, "height"])
        sigma_x = 40.0 / width * 256
        sigma_y = 40.0 / height * 256
        for reference, readers in fixations.items():
            variants = [("unaligned", None)]
            if reference == "post_report":
                variants.append(("global_aligned", GLOBAL_OFFSETS))
            for alignment, offsets in variants:
                member_histograms = []
                for reader, frame in readers.items():
                    offset = offsets.get(reader) if offsets is not None else None
                    member_histograms.append(histogram_for_reader(frame, case, width, height, offset=offset))

                # Total-duration pooling matches concatenating all events. Equal-reader
                # pooling normalizes each reader before averaging, bracketing the only
                # consequential ambiguity in the released target-construction record.
                event_hist = np.sum(member_histograms, axis=0)
                equal_hist = np.mean([
                    h / h.sum() if h.sum() > 0 else np.full_like(h, 1 / h.size)
                    for h in member_histograms
                ], axis=0)
                for pooling, histogram in [("event_duration", event_hist), ("equal_reader", equal_hist)]:
                    density = smooth_density(histogram, sigma_x, sigma_y)
                    for floor in (0.0, 0.01):
                        probability = (1 - floor) * density + floor / density.size
                        metrics = density_summaries(probability / probability.sum())
                        rows.append({
                            "case": case,
                            "reference": reference,
                            "alignment": alignment,
                            "pooling": pooling,
                            "uniform_floor": floor,
                            "n_readers": len(readers),
                            **metrics,
                        })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "2026-08-24_saliency_target_entropy_per_case.csv", index=False)

    summary_rows = []
    contrast_rows = []
    for (pooling, floor, alignment), subset in frame.groupby(["pooling", "uniform_floor", "alignment"]):
        for reference, group in subset.groupby("reference"):
            for metric in ["entropy_bits", "effective_support_shannon_cells", "effective_support_simpson_cells"]:
                values = group[metric].to_numpy(float)
                lo, hi = bootstrap_mean(values)
                summary_rows.append({
                    "pooling": pooling,
                    "uniform_floor": floor,
                    "alignment": alignment,
                    "reference": reference,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                })

    # Paired raw contrasts: positive entropy delta means the named second member
    # is sharper/lower entropy. Log support ratio makes multiplicative changes clear.
    for pooling in frame.pooling.unique():
        for floor in frame.uniform_floor.unique():
            raw = frame[(frame.pooling == pooling) & (frame.uniform_floor == floor) & (frame.alignment == "unaligned")]
            pivot_h = raw.pivot(index="case", columns="reference", values="entropy_bits")
            pivot_s = raw.pivot(index="case", columns="reference", values="effective_support_shannon_cells")
            for name, broad, sharp in [
                ("first_minus_second_session", "pre_report", "post_report"),
                ("subspecialist_minus_general_radiologist", "expert_consensus", "generalist_consensus"),
            ]:
                delta = pivot_h[broad] - pivot_h[sharp]
                log_ratio = np.log(pivot_s[sharp] / pivot_s[broad])
                dlo, dhi = bootstrap_mean(delta.to_numpy())
                rlo, rhi = bootstrap_mean(log_ratio.to_numpy())
                contrast_rows.append({
                    "pooling": pooling,
                    "uniform_floor": floor,
                    "contrast": name,
                    "entropy_difference_bits_positive_means_second_is_sharper": float(delta.mean()),
                    "entropy_ci_low": dlo,
                    "entropy_ci_high": dhi,
                    "geometric_effective_support_ratio_second_over_first": float(np.exp(log_ratio.mean())),
                    "support_ratio_ci_low": float(np.exp(rlo)),
                    "support_ratio_ci_high": float(np.exp(rhi)),
                    "cases_second_sharper": int((delta > 0).sum()),
                    "n_cases": len(delta),
                })

            post_raw = raw[raw.reference == "post_report"].set_index("case")
            post_aligned = frame[
                (frame.pooling == pooling)
                & (frame.uniform_floor == floor)
                & (frame.alignment == "global_aligned")
                & (frame.reference == "post_report")
            ].set_index("case")
            delta = post_raw.entropy_bits - post_aligned.entropy_bits
            dlo, dhi = bootstrap_mean(delta.to_numpy())
            ratio = np.log(
                post_aligned.effective_support_shannon_cells
                / post_raw.effective_support_shannon_cells
            )
            rlo, rhi = bootstrap_mean(ratio.to_numpy())
            contrast_rows.append({
                "pooling": pooling,
                "uniform_floor": floor,
                "contrast": "post_unaligned_minus_post_global_aligned",
                "entropy_difference_bits_positive_means_second_is_sharper": float(delta.mean()),
                "entropy_ci_low": dlo,
                "entropy_ci_high": dhi,
                "geometric_effective_support_ratio_second_over_first": float(np.exp(ratio.mean())),
                "support_ratio_ci_low": float(np.exp(rlo)),
                "support_ratio_ci_high": float(np.exp(rhi)),
                "cases_second_sharper": int((delta > 0).sum()),
                "n_cases": len(delta),
            })

    pd.DataFrame(summary_rows).to_csv(OUT / "2026-08-24_saliency_target_entropy_summary.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT / "2026-08-24_saliency_target_entropy_contrasts.csv", index=False)
    return {
        "n_cases": len(common),
        "grid": [256, 256],
        "kernel": "40 native pixels, represented anisotropically on the 256x256 grid",
        "pooling_rules": ["event_duration", "equal_reader"],
        "uniform_floors": [0.0, 0.01],
        "summary": summary_rows,
        "contrasts": contrast_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-shards-dir", type=Path, required=True)
    parser.add_argument("--coverage-completion-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subspecialist-fixations", type=Path, required=True)
    parser.add_argument("--general-radiologist-fixations", type=Path, required=True)
    parser.add_argument("--session-1-fixations", type=Path, required=True)
    parser.add_argument("--session-2-fixations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUT, DATA, COMPLETION_DATA, MANIFEST, FIXATION_SOURCES
    OUT = args.output_dir
    DATA = args.execution_shards_dir
    COMPLETION_DATA = args.coverage_completion_dir
    MANIFEST = args.manifest
    FIXATION_SOURCES = {
        "expert_consensus": args.subspecialist_fixations,
        "generalist_consensus": args.general_radiologist_fixations,
        "pre_report": args.session_1_fixations,
        "post_report": args.session_2_fixations,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    transfer = transfer_audit()
    entropy = target_entropy_audit()
    output = {
        "scope": "locally supplied execution records and controlled fixation inputs",
        "transfer": transfer,
        "target_entropy": entropy,
    }
    (OUT / "2026-08-24_saliency_robustness.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "matrices": transfer["matrices"],
        "column_best": transfer["column_best_training_source"],
        "entropy_contrasts": entropy["contrasts"],
    }, indent=2))


if __name__ == "__main__":
    main()
