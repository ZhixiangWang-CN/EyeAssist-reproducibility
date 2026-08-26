#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_fixed_pool import (
    Config,
    balanced_masks,
    density_map_from_events,
    percentile_bootstrap_mean,
    run,
    score_target_record,
    source_sets_exact,
    stable_seed,
    stimulus_metadata,
)


class FixedPoolTests(unittest.TestCase):
    def test_density_is_positive_and_normalized(self) -> None:
        cfg = Config(grid_size=16, native_size=64, sigma_native_px=3)
        events = pd.DataFrame(
            {"participant_id": ["R1", "R1"], "x": [10, 44], "y": [12, 40], "duration_ms": [100, 200]}
        )
        density = density_map_from_events(events, cfg)
        self.assertEqual(density.shape, (16, 16))
        self.assertTrue(np.all(density > 0))
        self.assertAlmostEqual(float(density.sum()), 1.0, places=6)

    def test_exact_compositions(self) -> None:
        sets = source_sets_exact(5, 4)
        masks = balanced_masks(4)
        self.assertEqual(sets.shape, (5, 4))
        self.assertEqual(masks.shape, (6, 4))
        self.assertTrue(np.all(masks.sum(axis=1) == 2))

    def test_matched_score_wins_constructed_example(self) -> None:
        cfg = Config(
            pool_size=2,
            grid_size=4,
            native_size=4,
            sigma_native_px=1,
            pool_mode="exact",
            bootstrap_replicates=20,
        )
        matched = np.full((3, 4, 4), 0.01, dtype=float)
        opposite = np.full((3, 4, 4), 0.01, dtype=float)
        matched[:, 1, 1] = 0.30
        opposite[:, 1, 1] = 0.02
        result = score_target_record(
            (np.asarray([1, 1]), np.asarray([1, 1])),
            matched,
            opposite,
            cfg,
            np.random.default_rng(1),
        )
        self.assertGreater(result["matched_minus_opposite"], 0)
        self.assertGreater(result["matched_minus_half_mixed"], 0)

    def test_seed_and_bootstrap_are_deterministic(self) -> None:
        self.assertEqual(stable_seed(7, "x", 2), stable_seed(7, "x", 2))
        a = percentile_bootstrap_mean(np.arange(5), 100, 23)
        b = percentile_bootstrap_mean(np.arange(5), 100, 23)
        self.assertEqual(a, b)

    def test_stimulus_pair_key(self) -> None:
        self.assertEqual(stimulus_metadata("real_013"), ("real", "013"))
        self.assertEqual(stimulus_metadata("fake_013"), ("fake", "013"))

    def test_small_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            for task, center in (("Task1", 180), ("Task2", 820)):
                (data / task).mkdir(parents=True)
                pd.DataFrame(
                    [{"File Name": "placeholder", **{f"R{x:02d}": 1 for x in range(1, 7)}}]
                ).to_csv(data / task / "expert_results.csv", index=False)
                for stimulus in ("real_001", "fake_001"):
                    folder = data / task / stimulus
                    folder.mkdir(parents=True)
                    rows = []
                    for reader in range(1, 7):
                        if task == "Task2" and stimulus == "fake_001" and reader == 6:
                            continue
                        for jitter in (-8, 0, 8):
                            rows.append(
                                {
                                    "participant_id": f"R{reader:02d}",
                                    "x": center + jitter,
                                    "y": center - jitter,
                                    "duration_ms": 100 + reader,
                                }
                            )
                    pd.DataFrame(rows).to_csv(folder / "scanpaths.csv", index=False)
            output = root / "output"
            cfg = Config(
                pool_size=4,
                grid_size=32,
                native_size=1080,
                sigma_native_px=25,
                pool_mode="sampled",
                sampled_draws=8,
                bootstrap_replicates=20,
            )
            run(data, output, root / "cache", cfg)
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(audit["n_common_stimuli"], 2)
            self.assertEqual(audit["n_source_study_pairs"], 1)
            self.assertEqual(audit["n_target_records_scored"], 22)
            self.assertEqual(audit["n_exclusions"], 2)
            scores = pd.read_csv(output / "target_record_scores.csv")
            self.assertTrue((scores["matched_minus_opposite"] > 0).all())
            exclusions = pd.read_csv(output / "exclusions.csv")
            self.assertEqual(set(exclusions["reason"]), {"unpaired_target_across_tasks"})

    def test_macos_metadata_tree_is_ignored(self) -> None:
        from run_fixed_pool import find_task_roots

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for prefix in (Path("Output"), Path("__MACOSX/Output")):
                for task in ("Task1", "Task2"):
                    (root / prefix / task / "real_001").mkdir(parents=True)
            roots = find_task_roots(root)
            self.assertNotIn("__MACOSX", roots["Task1"].parts)
            self.assertNotIn("__MACOSX", roots["Task2"].parts)


if __name__ == "__main__":
    unittest.main()
