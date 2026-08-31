import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/10_case_cluster_auc.py"
SPEC = importlib.util.spec_from_file_location("patient_independent_auc", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PatientIndependentClassificationTests(unittest.TestCase):
    def test_split_patient_overlap_is_removed_and_bootstrap_is_reproducible(self):
        labels = {"a": 1, "b": 1, "c": 1, "d": 0, "e": 0}
        rows = []
        for run, cases in [(1, list(labels)), (2, ["a", "c", "d", "e"])]:
            for index, case in enumerate(cases):
                label = labels[case]
                base = 0.8 if label else 0.2
                rows.append(
                    {
                        "run": run,
                        "case": case,
                        MODULE.DEFAULT_LABEL: label,
                        "image_only_prob": base,
                        "generalist_gaze_prob": base + 0.01,
                        "cold_read_gaze_prob": base + 0.02,
                        "informed_gaze_prob": base + 0.03,
                    }
                )
        patient_map = pd.DataFrame(
            {
                "case": ["a", "b", "c", "d", "e"],
                "patient": ["p1", "p1", "p2", "p3", "p4"],
            }
        )
        first = MODULE.analyse(
            pd.DataFrame(rows), patient_map, n_bootstrap=100, seed=11
        )
        second = MODULE.analyse(
            pd.DataFrame(rows), patient_map, n_bootstrap=100, seed=11
        )
        self.assertEqual(first, second)
        self.assertEqual(first["original_rows"], 9)
        self.assertEqual(first["retained_rows"], 8)
        self.assertEqual(first["represented_patients"], 4)
        self.assertEqual(first["test_records_per_run"], {"min": 3, "median": 4.0, "max": 5})


if __name__ == "__main__":
    unittest.main()
