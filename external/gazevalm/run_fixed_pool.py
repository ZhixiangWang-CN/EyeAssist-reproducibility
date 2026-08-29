#!/usr/bin/env python3
"""Fixed-pool task analysis on the public GazeVaLM release.

The primary estimand is the held-out log-score advantage of a task-matched
reference over equal-size half-mixed and opposite-task references. Source
reader identities are paired across all configurations and the target reader
is excluded from both tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter


TASKS = ("Task1", "Task2")


@dataclass(frozen=True)
class Config:
    pool_size: int = 4
    grid_size: int = 270
    native_size: int = 1080
    sigma_native_px: float = 25.0
    alpha: float = 0.01
    pool_mode: str = "exact"
    sampled_draws: int = 512
    pool_seed: int = 20260824
    bootstrap_replicates: int = 5000
    bootstrap_seed: int = 20260824
    cache_version: str = "gazevalm-fixed-pool-v1"

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text())
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**raw)

    def validate(self) -> None:
        if self.pool_size <= 0 or self.pool_size % 2:
            raise ValueError("pool_size must be a positive even integer")
        if self.grid_size <= 1 or self.native_size <= 1:
            raise ValueError("grid_size and native_size must exceed one")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must lie strictly between zero and one")
        if self.sigma_native_px <= 0:
            raise ValueError("sigma_native_px must be positive")
        if self.pool_mode not in {"exact", "sampled"}:
            raise ValueError("pool_mode must be 'exact' or 'sampled'")
        if self.sampled_draws <= 0:
            raise ValueError("sampled_draws must be positive")


def stable_seed(base: int, *parts: object) -> int:
    text = "|".join([str(base), *(str(p) for p in parts)])
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def find_task_roots(data_root: Path) -> Dict[str, Path]:
    roots: Dict[str, Path] = {}
    for task in TASKS:
        candidates = [
            p
            for p in data_root.rglob(task)
            if p.is_dir() and "__MACOSX" not in p.parts
        ]
        candidates = [
            p
            for p in candidates
            if any(q.is_dir() and q.name.startswith(("real_", "fake_")) for q in p.iterdir())
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {task} directory below {data_root}; found {candidates}"
            )
        roots[task] = candidates[0]
    return roots


def stimulus_metadata(name: str) -> Tuple[str, str]:
    if name.startswith("real_"):
        authenticity = "real"
    elif name.startswith("fake_"):
        authenticity = "fake"
    else:
        raise ValueError(f"Unrecognized stimulus name: {name}")
    study_id = name.split("_", 1)[1]
    return authenticity, study_id


def official_participant_set(task_roots: Mapping[str, Path]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Read the 16-reader expert cohort declared by the public release tables."""
    by_task: Dict[str, List[str]] = {}
    for task in TASKS:
        path = task_roots[task] / "expert_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing cohort-defining file: {path}")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        participant_columns = [str(x).strip() for x in columns if str(x).strip() != "File Name"]
        if not participant_columns:
            raise ValueError(f"No participant columns found in {path}")
        by_task[task] = sorted(set(participant_columns))
    common = sorted(set(by_task["Task1"]) & set(by_task["Task2"]))
    if not common:
        raise ValueError("Task1 and Task2 expert tables have no participant IDs in common")
    return common, by_task


def clean_events(frame: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = {"participant_id", "x", "y", "duration_ms"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"scanpaths.csv missing columns: {sorted(missing)}")
    work = frame.copy()
    for col in ("x", "y", "duration_ms"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    participant_present = work["participant_id"].notna() & (
        work["participant_id"].astype(str).str.strip() != ""
    )
    finite = participant_present & np.isfinite(work[["x", "y", "duration_ms"]]).all(axis=1)
    positive_duration = work["duration_ms"] > 0
    in_bounds = (
        (work["x"] >= 0)
        & (work["x"] < config.native_size)
        & (work["y"] >= 0)
        & (work["y"] < config.native_size)
    )
    keep = finite & positive_duration & in_bounds
    counts = {
        "rows_total": int(len(work)),
        "rows_nonfinite": int((~finite).sum()),
        "rows_nonpositive_duration": int((finite & ~positive_duration).sum()),
        "rows_out_of_bounds": int((finite & positive_duration & ~in_bounds).sum()),
        "rows_valid": int(keep.sum()),
    }
    work = work.loc[keep].copy()
    # Preserve the public release's participant labels verbatim. They may be
    # integers or strings; coercing them to numbers can silently erase valid IDs.
    work["participant_id"] = work["participant_id"].astype(str).str.strip()
    return work, counts


def event_indices(events: pd.DataFrame, config: Config) -> Tuple[np.ndarray, np.ndarray]:
    scale = config.grid_size / config.native_size
    x = np.floor(events["x"].to_numpy(float) * scale).astype(int)
    y = np.floor(events["y"].to_numpy(float) * scale).astype(int)
    return np.clip(y, 0, config.grid_size - 1), np.clip(x, 0, config.grid_size - 1)


def density_map_from_events(events: pd.DataFrame, config: Config) -> np.ndarray:
    if events.empty:
        raise ValueError("Cannot build a map from an empty event table")
    y, x = event_indices(events, config)
    weights = events["duration_ms"].to_numpy(float)
    hist = np.zeros((config.grid_size, config.grid_size), dtype=np.float64)
    np.add.at(hist, (y, x), weights)
    sigma_grid = config.sigma_native_px * config.grid_size / config.native_size
    smoothed = gaussian_filter(hist, sigma=sigma_grid, mode="constant")
    total = float(smoothed.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid density-map mass")
    smoothed /= total
    uniform = 1.0 / smoothed.size
    smoothed = (1.0 - config.alpha) * smoothed + config.alpha * uniform
    smoothed /= smoothed.sum()
    if not np.isfinite(smoothed).all() or np.any(smoothed <= 0):
        raise AssertionError("Density map must be finite and strictly positive")
    return smoothed.astype(np.float32)


def source_sets_exact(n_sources: int, pool_size: int) -> np.ndarray:
    if n_sources < pool_size:
        return np.empty((0, pool_size), dtype=np.int16)
    return np.asarray(list(itertools.combinations(range(n_sources), pool_size)), dtype=np.int16)


def balanced_masks(pool_size: int) -> np.ndarray:
    masks = []
    for chosen in itertools.combinations(range(pool_size), pool_size // 2):
        mask = np.zeros(pool_size, dtype=bool)
        mask[list(chosen)] = True
        masks.append(mask)
    return np.stack(masks)


def sampled_source_sets(
    n_sources: int, pool_size: int, n_draws: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    sets = np.empty((n_draws, pool_size), dtype=np.int16)
    masks = np.empty((n_draws, pool_size), dtype=bool)
    for i in range(n_draws):
        sets[i] = np.sort(rng.choice(n_sources, size=pool_size, replace=False))
        chosen = rng.choice(pool_size, size=pool_size // 2, replace=False)
        masks[i] = False
        masks[i, chosen] = True
    return sets, masks


def relative_log_score(values: np.ndarray, uniform: float) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must be compositions x target-events")
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise AssertionError("Pool probabilities must be finite and positive")
    return np.log2(values / uniform).mean(axis=1)


def score_target_record(
    target_points: Tuple[np.ndarray, np.ndarray],
    matched_maps: np.ndarray,
    opposite_maps: np.ndarray,
    config: Config,
    rng: np.random.Generator,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Score one target record against paired source-reader configurations."""
    config.validate()
    if matched_maps.shape != opposite_maps.shape:
        raise AssertionError("Matched and opposite source arrays must have identical identities")
    if matched_maps.ndim != 3:
        raise ValueError("source maps must have shape readers x height x width")
    n_sources = matched_maps.shape[0]
    if n_sources < config.pool_size:
        raise ValueError("Insufficient paired source readers")
    y, x = target_points
    a = matched_maps[:, y, x].astype(np.float64)
    b = opposite_maps[:, y, x].astype(np.float64)
    uniform = 1.0 / (config.grid_size * config.grid_size)

    matched_scores: List[np.ndarray] = []
    opposite_scores: List[np.ndarray] = []
    mixed_scores: List[np.ndarray] = []

    if config.pool_mode == "sampled":
        sets, masks = sampled_source_sets(
            n_sources, config.pool_size, config.sampled_draws, rng
        )
        for start in range(0, len(sets), batch_size):
            idx = sets[start : start + batch_size]
            mask = masks[start : start + batch_size, :, None]
            av = a[idx]
            bv = b[idx]
            matched_scores.append(relative_log_score(av.mean(axis=1), uniform))
            opposite_scores.append(relative_log_score(bv.mean(axis=1), uniform))
            mixed_scores.append(relative_log_score(np.where(mask, av, bv).mean(axis=1), uniform))
        n_source_sets = len(sets)
        n_mixed_assignments = len(sets)
    else:
        sets = source_sets_exact(n_sources, config.pool_size)
        masks = balanced_masks(config.pool_size)
        for start in range(0, len(sets), batch_size):
            idx = sets[start : start + batch_size]
            av = a[idx]
            bv = b[idx]
            matched_scores.append(relative_log_score(av.mean(axis=1), uniform))
            opposite_scores.append(relative_log_score(bv.mean(axis=1), uniform))
            for mask in masks:
                mixed = np.where(mask[None, :, None], av, bv).mean(axis=1)
                mixed_scores.append(relative_log_score(mixed, uniform))
        n_source_sets = len(sets)
        n_mixed_assignments = len(sets) * len(masks)

    matched = float(np.concatenate(matched_scores).mean())
    opposite = float(np.concatenate(opposite_scores).mean())
    mixed = float(np.concatenate(mixed_scores).mean())
    return {
        "matched_bits": matched,
        "opposite_bits": opposite,
        "half_mixed_bits": mixed,
        "matched_minus_opposite": matched - opposite,
        "matched_minus_half_mixed": matched - mixed,
        "n_paired_source_readers": int(n_sources),
        "n_source_sets": int(n_source_sets),
        "n_mixed_assignments": int(n_mixed_assignments),
        "n_target_fixations": int(len(y)),
    }


def cache_key(scanpath: Path, config: Config, allowed_participants: Sequence[str]) -> str:
    stat = scanpath.stat()
    payload = {
        "path": str(scanpath.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "grid_size": config.grid_size,
        "native_size": config.native_size,
        "sigma_native_px": config.sigma_native_px,
        "alpha": config.alpha,
        "cache_version": config.cache_version,
        "allowed_participants": sorted(allowed_participants),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def load_stimulus_records(
    scanpath: Path,
    task: str,
    stimulus: str,
    config: Config,
    cache_dir: Path,
    allowed_participants: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, object]]:
    key = cache_key(scanpath, config, allowed_participants)
    cache_path = cache_dir / task / f"{stimulus}.{key}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(scanpath)
    clean, counts = clean_events(raw, config)
    allowed = set(str(x) for x in allowed_participants)
    pre_cohort_rows = len(clean)
    clean = clean.loc[clean["participant_id"].isin(allowed)].copy()
    counts["rows_valid_precohort"] = int(pre_cohort_rows)
    counts["rows_excluded_noncohort"] = int(pre_cohort_rows - len(clean))
    counts["rows_valid"] = int(len(clean))
    participant_ids = sorted(clean["participant_id"].unique().tolist())
    maps: Dict[str, np.ndarray] = {}
    points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["participant_ids"].astype(str).tolist()
        if cached_ids == participant_ids:
            stack = cached["maps"]
            for i, pid in enumerate(participant_ids):
                maps[pid] = stack[i]

    if len(maps) != len(participant_ids):
        maps = {}
        stack = []
        for pid in participant_ids:
            events = clean.loc[clean["participant_id"] == pid]
            member_map = density_map_from_events(events, config)
            maps[pid] = member_map
            stack.append(member_map)
        np.savez_compressed(
            cache_path,
            participant_ids=np.asarray(participant_ids, dtype=str),
            maps=np.stack(stack) if stack else np.empty((0, config.grid_size, config.grid_size)),
        )

    for pid in participant_ids:
        events = clean.loc[clean["participant_id"] == pid]
        points[pid] = event_indices(events, config)

    audit = {
        "task": task,
        "stimulus": stimulus,
        "scanpath": str(scanpath),
        "participants": ";".join(str(x) for x in participant_ids),
        "n_participants": len(participant_ids),
        **counts,
    }
    return maps, points, audit


def percentile_bootstrap_mean(
    values: np.ndarray, n_replicates: int, seed: int
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_replicates, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(
    target_df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = ["matched_bits", "opposite_bits", "half_mixed_bits", "matched_minus_opposite", "matched_minus_half_mixed"]
    task_level = target_df.groupby(
        ["stimulus", "authenticity", "study_id", "target_task"], as_index=False
    )[metrics].mean()
    symmetric = task_level.groupby(
        ["stimulus", "authenticity", "study_id"], as_index=False
    )[metrics].mean()
    symmetric["target_task"] = "symmetric"
    stimulus_df = pd.concat([task_level, symmetric], ignore_index=True)

    cluster_df = symmetric.groupby("study_id", as_index=False)[metrics].mean()
    summary_rows: List[Dict[str, object]] = []

    def add_group(label: str, frame: pd.DataFrame, cluster_col: str) -> None:
        clustered = frame.groupby(cluster_col, as_index=False)[metrics].mean()
        for metric in metrics:
            estimate, low, high = percentile_bootstrap_mean(
                clustered[metric].to_numpy(),
                config.bootstrap_replicates,
                stable_seed(config.bootstrap_seed, label, metric),
            )
            summary_rows.append(
                {
                    "analysis": label,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_clusters": int(len(clustered)),
                    "n_positive_clusters": int((clustered[metric] > 0).sum()),
                    "fraction_positive_clusters": float((clustered[metric] > 0).mean()),
                    "cluster_unit": cluster_col,
                }
            )

    add_group("overall_symmetric", symmetric, "study_id")
    for task in TASKS:
        add_group(
            f"target_{task}", task_level.loc[task_level["target_task"] == task], "study_id"
        )
    task1_cluster = task_level.loc[task_level["target_task"] == "Task1"].groupby(
        "study_id", as_index=False
    )[metrics].mean()
    task2_cluster = task_level.loc[task_level["target_task"] == "Task2"].groupby(
        "study_id", as_index=False
    )[metrics].mean()
    task_contrast = task1_cluster.merge(
        task2_cluster, on="study_id", suffixes=("_Task1", "_Task2"), validate="one_to_one"
    )
    for metric in metrics:
        task_contrast[metric] = (
            task_contrast[f"{metric}_Task1"] - task_contrast[f"{metric}_Task2"]
        )
    add_group("Task1_minus_Task2", task_contrast, "study_id")
    for authenticity in ("real", "fake"):
        add_group(
            authenticity,
            symmetric.loc[symmetric["authenticity"] == authenticity],
            "study_id",
        )

    reader_rows = []
    for pid, frame in target_df.groupby("target_reader"):
        row = {"target_reader": pid, "n_target_records": len(frame)}
        row.update({metric: frame[metric].mean() for metric in metrics})
        reader_rows.append(row)
    reader_df = pd.DataFrame(reader_rows)

    loro_rows = []
    for held_out in sorted(target_df["target_reader"].unique()):
        reduced = target_df.loc[target_df["target_reader"] != held_out]
        reduced_task = reduced.groupby(
            ["stimulus", "authenticity", "study_id", "target_task"], as_index=False
        )[metrics].mean()
        reduced_sym = reduced_task.groupby(
            ["stimulus", "authenticity", "study_id"], as_index=False
        )[metrics].mean()
        row = {"held_out_target_reader": held_out, "n_target_records": len(reduced)}
        row.update({metric: reduced_sym[metric].mean() for metric in metrics})
        loro_rows.append(row)
    loro_df = pd.DataFrame(loro_rows)
    return stimulus_df, cluster_df, pd.DataFrame(summary_rows), reader_df.merge(loro_df, how="outer", left_on="target_reader", right_on="held_out_target_reader")


def tree_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def run(data_root: Path, output_dir: Path, cache_dir: Path, config: Config) -> None:
    started = time.time()
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    task_roots = find_task_roots(data_root)
    official_participants, official_participants_by_task = official_participant_set(task_roots)
    stimuli_by_task = {
        task: {
            p.name: p / "scanpaths.csv"
            for p in root.iterdir()
            if p.is_dir() and p.name.startswith(("real_", "fake_")) and (p / "scanpaths.csv").exists()
        }
        for task, root in task_roots.items()
    }
    common_stimuli = sorted(set(stimuli_by_task["Task1"]) & set(stimuli_by_task["Task2"]))
    exclusions: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    target_rows: List[Dict[str, object]] = []
    scanpath_files: List[Path] = []

    task_participants = {task: set() for task in TASKS}
    print(
        f"Found {len(common_stimuli)} common stimuli; "
        f"pool_mode={config.pool_mode}, pool_size={config.pool_size}",
        flush=True,
    )
    for stimulus_index, stimulus in enumerate(common_stimuli, start=1):
        authenticity, study_id = stimulus_metadata(stimulus)
        records = {}
        for task in TASKS:
            scanpath = stimuli_by_task[task][stimulus]
            scanpath_files.append(scanpath)
            maps, points, audit = load_stimulus_records(
                scanpath,
                task,
                stimulus,
                config,
                cache_dir,
                official_participants,
            )
            records[task] = (maps, points)
            task_participants[task].update(maps)
            audit_rows.append(audit)

        eligible_target_ids = sorted(
            set(records["Task1"][0]) & set(records["Task2"][0])
        )
        for missing_target in sorted(set(official_participants) - set(eligible_target_ids)):
            for target_task in TASKS:
                exclusions.append(
                    {
                        "stimulus": stimulus,
                        "target_task": target_task,
                        "target_reader": missing_target,
                        "reason": "unpaired_target_across_tasks",
                        "n_available": len(eligible_target_ids),
                    }
                )

        for target_task in TASKS:
            opposite_task = "Task2" if target_task == "Task1" else "Task1"
            target_maps, target_points = records[target_task]
            opposite_maps, _ = records[opposite_task]
            for target_reader in eligible_target_ids:
                candidate_ids = sorted(
                    set(eligible_target_ids) - {target_reader}
                )
                if len(candidate_ids) < config.pool_size:
                    exclusions.append(
                        {
                            "stimulus": stimulus,
                            "target_task": target_task,
                            "target_reader": target_reader,
                            "reason": "insufficient_paired_source_readers",
                            "n_available": len(candidate_ids),
                        }
                    )
                    continue
                points = target_points[target_reader]
                if len(points[0]) == 0:
                    exclusions.append(
                        {
                            "stimulus": stimulus,
                            "target_task": target_task,
                            "target_reader": target_reader,
                            "reason": "empty_target_fixations",
                            "n_available": len(candidate_ids),
                        }
                    )
                    continue
                matched_stack = np.stack([target_maps[x] for x in candidate_ids])
                opposite_stack = np.stack([opposite_maps[x] for x in candidate_ids])
                rng = np.random.default_rng(
                    stable_seed(config.pool_seed, stimulus, target_task, target_reader)
                )
                score = score_target_record(
                    points, matched_stack, opposite_stack, config, rng
                )
                target_rows.append(
                    {
                        "stimulus": stimulus,
                        "authenticity": authenticity,
                        "study_id": study_id,
                        "target_task": target_task,
                        "target_reader": target_reader,
                        **score,
                    }
                )

        print(
            f"[{stimulus_index}/{len(common_stimuli)}] {stimulus}: "
            f"cumulative targets={len(target_rows)}, exclusions={len(exclusions)}",
            flush=True,
        )
        if stimulus_index % 5 == 0:
            pd.DataFrame(target_rows).to_csv(
                output_dir / "target_record_scores.partial.csv", index=False
            )

    for task in TASKS:
        for missing_stimulus in sorted(set(stimuli_by_task[task]) - set(common_stimuli)):
            exclusions.append(
                {
                    "stimulus": missing_stimulus,
                    "target_task": task,
                    "target_reader": "",
                    "reason": "stimulus_missing_from_other_task",
                    "n_available": "",
                }
            )

    audit_df = pd.DataFrame(audit_rows)
    target_df = pd.DataFrame(target_rows)
    if target_df.empty:
        raise RuntimeError("No eligible target records were scored")
    stimulus_df, cluster_df, summary_df, reader_sensitivity_df = summarize(target_df, config)

    audit_df.to_csv(output_dir / "audit_records.csv", index=False)
    target_df.to_csv(output_dir / "target_record_scores.csv", index=False)
    partial_path = output_dir / "target_record_scores.partial.csv"
    if partial_path.exists():
        partial_path.unlink()
    stimulus_df.to_csv(output_dir / "stimulus_level_scores.csv", index=False)
    cluster_df.to_csv(output_dir / "source_study_cluster_scores.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    reader_sensitivity_df.to_csv(output_dir / "reader_sensitivity.csv", index=False)
    pd.DataFrame(exclusions).to_csv(output_dir / "exclusions.csv", index=False)

    task1_ids = sorted(task_participants["Task1"])
    task2_ids = sorted(task_participants["Task2"])
    audit = {
        "task_roots": {k: str(v) for k, v in task_roots.items()},
        "n_stimuli": {k: len(v) for k, v in stimuli_by_task.items()},
        "n_common_stimuli": len(common_stimuli),
        "n_source_study_pairs": len({stimulus_metadata(x)[1] for x in common_stimuli}),
        "participants_Task1": task1_ids,
        "participants_Task2": task2_ids,
        "participant_intersection": sorted(set(task1_ids) & set(task2_ids)),
        "same_participant_set": task1_ids == task2_ids,
        "official_participants_by_task": official_participants_by_task,
        "official_participant_intersection": official_participants,
        "n_target_records_scored": len(target_df),
        "n_exclusions": len(exclusions),
        "rows_total": int(audit_df["rows_total"].sum()),
        "rows_valid": int(audit_df["rows_valid"].sum()),
        "rows_nonfinite": int(audit_df["rows_nonfinite"].sum()),
        "rows_nonpositive_duration": int(audit_df["rows_nonpositive_duration"].sum()),
        "rows_out_of_bounds": int(audit_df["rows_out_of_bounds"].sum()),
        "rows_excluded_noncohort": int(audit_df["rows_excluded_noncohort"].sum()),
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    (output_dir / "summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2)
    )
    manifest = {
        "config": asdict(config),
        "data_root": str(data_root.resolve()),
        "task_roots": {k: str(v.resolve()) for k, v in task_roots.items()},
        "scanpath_tree_digest": tree_digest(scanpath_files),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "started_unix": started,
        "finished_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    primary = summary_df.loc[
        (summary_df["analysis"] == "overall_symmetric")
        & summary_df["metric"].isin(
            ["matched_minus_opposite", "matched_minus_half_mixed"]
        )
    ]
    lines = [
        "GazeVaLM fixed-pool task analysis",
        f"Eligible target records: {len(target_df)}",
        f"Common stimuli: {len(common_stimuli)}; source-study clusters: {audit['n_source_study_pairs']}",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"{row.metric}: {row.estimate:+.4f} bits/fixation "
            f"(95% source-study-cluster bootstrap interval {row.ci_low:+.4f} to {row.ci_high:+.4f})"
        )
    (output_dir / "manuscript_report.txt").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.data_root, args.output_dir, args.cache_dir, Config.from_json(args.config))


if __name__ == "__main__":
    main()
