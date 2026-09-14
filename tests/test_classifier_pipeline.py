import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eyeassist.classifier_pipeline import (
    FINAL_CLASSIFIER_EPOCH,
    classifier_metrics,
    is_final_classifier_epoch,
    load_case_and_split_tables,
    validate_final_classifier_checkpoint,
)


class ClassifierPipelineTests(unittest.TestCase):
    def test_final_epoch_selection(self) -> None:
        self.assertFalse(is_final_classifier_epoch(FINAL_CLASSIFIER_EPOCH - 1))
        self.assertTrue(is_final_classifier_epoch(FINAL_CLASSIFIER_EPOCH))

    def test_final_epoch_bounds(self) -> None:
        with self.assertRaises(ValueError):
            is_final_classifier_epoch(FINAL_CLASSIFIER_EPOCH + 1)

    def test_final_epoch_checkpoint(self) -> None:
        checkpoint = {
            "epoch": FINAL_CLASSIFIER_EPOCH,
            "checkpoint_rule": "final_epoch",
            "run_config": {
                "epochs": FINAL_CLASSIFIER_EPOCH,
                "checkpoint_rule": "final_epoch",
            },
        }
        validate_final_classifier_checkpoint(checkpoint, "selected.pt")
        with self.assertRaises(ValueError):
            validate_final_classifier_checkpoint(checkpoint, "last.pt")
        checkpoint["epoch"] = FINAL_CLASSIFIER_EPOCH - 1
        with self.assertRaises(ValueError):
            validate_final_classifier_checkpoint(checkpoint, "selected.pt")

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
