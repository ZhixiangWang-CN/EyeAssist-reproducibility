import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GAZEVALM = ROOT / "external" / "gazevalm"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ReleaseHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(GAZEVALM))
        cls.interaction = load_module(
            "gazevalm_task_interaction", GAZEVALM / "summarize_task_interaction.py"
        )
        cls.concentration = load_module(
            "gazevalm_task_concentration", GAZEVALM / "summarize_task_concentration.py"
        )
        cls.fixed_specificity = load_module(
            "classification_fixed_specificity",
            ROOT / "scripts" / "16_classification_fixed_specificity.py",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if sys.path and sys.path[0] == str(GAZEVALM):
            sys.path.pop(0)

    def test_gazevalm_interaction_ci_is_deterministic(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0])
        indices = np.asarray([[0, 1, 2], [0, 0, 0], [2, 2, 2]])
        self.assertEqual(self.interaction.ci(values, indices), [1.05, 2.95])

    def test_gazevalm_concentration_ci_is_deterministic(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0])
        indices = np.asarray([[0, 1, 2], [0, 0, 0], [2, 2, 2]])
        self.assertEqual(self.concentration.percentile_ci(values, indices), [1.05, 2.95])

    def test_fixed_specificity_helper_recovers_separable_ranking(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])
        observed = self.fixed_specificity.sensitivity_at_specificity(
            labels, scores, specificity=0.5
        )
        np.testing.assert_allclose(observed, np.asarray([1.0]))


if __name__ == "__main__":
    unittest.main()
