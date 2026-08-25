import pandas as pd
import unittest

from eyeassist.splits import assert_group_disjoint, repeated_group_stratified_splits


class SplitTests(unittest.TestCase):
    def test_repeated_splits_are_group_disjoint(self):
        manifest = pd.DataFrame(
            {
                "case_id": [f"c{i}" for i in range(20)],
                "reader_group": [f"g{i // 2}" for i in range(20)],
                "label": [(i // 2) % 2 for i in range(20)],
            }
        )
        splits = repeated_group_stratified_splits(
            manifest, repeats=4, test_groups=4, seed=7, group_column="reader_group"
        )
        assert_group_disjoint(splits, group_column="reader_group")
        self.assertTrue(splits.groupby("split_id").size().eq(20).all())

    def test_inconsistent_group_labels_are_rejected(self):
        manifest = pd.DataFrame(
            {"case_id": ["a", "b"], "reader_group": ["g", "g"], "label": [0, 1]}
        )
        with self.assertRaisesRegex(ValueError, "one label per group"):
            repeated_group_stratified_splits(
                manifest, repeats=1, test_groups=1, seed=0, group_column="reader_group"
            )
