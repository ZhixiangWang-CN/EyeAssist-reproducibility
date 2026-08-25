import numpy as np
import unittest

from eyeassist.statistics import (
    exact_group_label_permutation,
    exact_sign_permutation,
    paired_bootstrap_difference,
)


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_permutation_detects_consistent_direction(self):
        mean, p = exact_sign_permutation(np.ones(7))
        self.assertEqual(mean, 1.0)
        self.assertTrue(np.isclose(p, 2 / 128))

    def test_paired_bootstrap_is_reproducible(self):
        first = np.array([2.0, 3.0, 4.0])
        second = np.array([1.0, 1.0, 1.0])
        one = paired_bootstrap_difference(first, second, n_resamples=1000, seed=3)
        two = paired_bootstrap_difference(first, second, n_resamples=1000, seed=3)
        self.assertEqual(one, two)

    def test_exact_group_permutation_counts_all_allocations(self):
        difference, p_value, exceed, total = exact_group_label_permutation(
            np.array([3.0, 4.0]), np.array([0.0, 1.0])
        )
        self.assertEqual(difference, 3.0)
        self.assertEqual(total, 6)
        self.assertEqual(exceed, 2)
        self.assertTrue(np.isclose(p_value, 1 / 3))
