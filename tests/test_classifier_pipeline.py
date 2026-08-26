import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eyeassist.classifier_pipeline import (
    checkpoint_is_better,
    classifier_metrics,
    load_case_and_split_tables,
)


class ClassifierPipelineTests(unittest.TestCase):
    def test_last_epoch_checkpoint_rule(self) -> None:
        self.assertEqual(
            checkpoint_is_better(
                rule="last_epoch",
                epoch=2,
                max_epochs=3,
                validation_metrics=None,
                best_value=None,
            ),
            (False, None),
        )
        self.assertEqual(
            checkpoint_is_better(
                rule="last_epoch",
                epoch=3,
                max_epochs=3,
                validation_metrics=None,
                best_value=None,
            ),
            (True, None),
        )

    def test_validation_checkpoint_never_uses_test_metric(self) -> None:
        selected, value = checkpoint_is_better(
            rule="best_val_auroc",
            epoch=2,
            max_epochs=5,
            validation_metrics={"loss": 0.4, "auroc": 0.8},
            best_value=0.7,
        )
        self.assertTrue(selected)
        self.assertEqual(value, 0.8)

    def test_metrics(self) -> None:
        result = classifier_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.6, 0.9]))
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)

    def test_undefined_predictive_value_is_null(self) -> None:
        result = classifier_metrics(np.asarray([0, 1]), np.asarray([0.1, 0.2]))
        self.assertIsNone(result["ppv"])

    def test_split_loader_preserves_case_disjointness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.png", "b.png"):
                (root / name).touch()
            pd.DataFrame(
                [
                    {"case_id": "a", "image_path": "a.png", "label": 0},
                    {"case_id": "b", "image_path": "b.png", "label": 1},
                ]
            ).to_csv(root / "manifest.csv", index=False)
            pd.DataFrame(
                [
                    {"split_id": 0, "case_id": "a", "partition": "train"},
                    {"split_id": 0, "case_id": "b", "partition": "test"},
                ]
            ).to_csv(root / "splits.csv", index=False)
            table = load_case_and_split_tables(
                root / "manifest.csv", root / "splits.csv", split_id=0, arm="image_only"
            )
            self.assertEqual(set(table.partition), {"train", "test"})


if __name__ == "__main__":
    unittest.main()
