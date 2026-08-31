import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/17_saliency_case_cluster_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("saliency_case_cluster", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SaliencyCaseClusterTests(unittest.TestCase):
    def test_equal_case_cluster_bootstrap_is_reproducible(self):
        rows = []
        targets = {value for contrast in MODULE.CONTRASTS.values() for value in contrast}
        for case, appearances in [("a", 3), ("b", 6), ("c", 3)]:
            for metric in MODULE.METRICS:
                for trained_on in targets:
                    for scored_against in targets:
                        value = 0.2 if trained_on == scored_against else 0.1
                        if metric == "KLDiv":
                            value = 0.1 if trained_on == scored_against else 0.2
                        rows.append(
                            {
                                "case": case,
                                "trained_on": trained_on,
                                "scored_against": scored_against,
                                "metric": metric,
                                "mean": value,
                                "appearances": appearances,
                            }
                        )
        frame = pd.DataFrame(rows)
        first = MODULE.analyse(frame, n_bootstrap=100, seed=7)
        second = MODULE.analyse(frame, n_bootstrap=100, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(np.allclose(first["primary_point"], 0.1))
        self.assertTrue(np.allclose(first["equal_cluster_point"], 0.1))
        self.assertTrue(np.allclose(first["weighted_point"], 0.1))

    def test_patient_map_averages_repeated_records_before_resampling(self):
        rows = []
        targets = {value for contrast in MODULE.CONTRASTS.values() for value in contrast}
        effects = {"a": 0.3, "b": 0.1, "c": 0.2}
        for case, effect in effects.items():
            for metric in MODULE.METRICS:
                for trained_on in targets:
                    for scored_against in targets:
                        matched = trained_on == scored_against
                        if metric == "KLDiv":
                            value = 0.4 - effect if matched else 0.4
                        else:
                            value = effect if matched else 0.0
                        rows.append(
                            {
                                "case": case,
                                "trained_on": trained_on,
                                "scored_against": scored_against,
                                "metric": metric,
                                "mean": value,
                                "appearances": 1,
                            }
                        )
        patient_map = pd.DataFrame(
            {"case": ["a", "b", "c"], "patient": ["p1", "p1", "p2"]}
        )
        result = MODULE.analyse(
            pd.DataFrame(rows), patient_map=patient_map, n_bootstrap=100, seed=7
        )
        self.assertTrue((result["cluster_unit"] == "patient").all())
        self.assertTrue((result["n_clusters"] == 2).all())
        # p1 contributes mean(0.3, 0.1)=0.2 and p2 contributes 0.2.
        self.assertTrue(np.allclose(result["primary_point"], 0.2))


if __name__ == "__main__":
    unittest.main()
