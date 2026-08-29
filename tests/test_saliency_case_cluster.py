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
    def test_weighted_case_cluster_bootstrap_is_reproducible(self):
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
        self.assertTrue(np.allclose(first["weighted_point"], 0.1))


if __name__ == "__main__":
    unittest.main()
