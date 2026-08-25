import numpy as np
import pandas as pd
import unittest

from eyeassist.gaze import (
    density_map,
    fixation_log_score,
    kl_divergence,
    leave_one_case_out_offsets,
    nss,
    pearson_cc,
    similarity,
)


class GazeTests(unittest.TestCase):
    def test_density_is_finite_unit_mass(self):
        fixations = np.array([[20, 20, 100], [22, 19, 200], [80, 80, 50]], dtype=float)
        density = density_map(fixations, (100, 100), sigma_pixels=8)
        self.assertTrue(np.isfinite(density).all())
        self.assertTrue(np.isclose(density.sum(), 1.0))
        self.assertTrue((density > 0).all())

    def test_matching_density_has_higher_log_score(self):
        target = np.array([[20, 20, 100], [21, 19, 80], [22, 20, 120]], dtype=float)
        match = density_map(target, (100, 100), sigma_pixels=6)
        opposite = density_map(
            np.array([[80, 80, 100], [78, 82, 100]]), (100, 100), sigma_pixels=6
        )
        self.assertGreater(fixation_log_score(target, match), fixation_log_score(target, opposite))

    def test_metrics_have_expected_identity_values(self):
        array = np.array([[0.1, 0.2], [0.3, 0.4]])
        mask = np.array([[0, 0], [0, 1]])
        self.assertTrue(np.isclose(pearson_cc(array, array), 1.0))
        self.assertTrue(np.isclose(similarity(array, array), 1.0))
        self.assertTrue(np.isclose(kl_divergence(array, array), 0.0))
        self.assertGreater(nss(array, mask), 0)

    def test_leave_one_case_out_alignment_does_not_use_held_out_case(self):
        first = pd.DataFrame(
            {"case_id": ["a", "b", "c"], "x": [0, 0, 0], "y": [0, 0, 0], "duration": [1, 1, 1]}
        )
        second = pd.DataFrame(
            {"case_id": ["a", "b", "c"], "x": [10, 10, 100], "y": [5, 5, 50], "duration": [1, 1, 1]}
        )
        offsets = leave_one_case_out_offsets(first, second)
        self.assertTrue(np.allclose(offsets["c"], [10, 5]))
        self.assertFalse(np.allclose(offsets["a"], [10, 5]))
