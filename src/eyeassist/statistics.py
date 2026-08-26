"""Inference utilities with explicit sampling units."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np


def percentile_bootstrap_mean(
    values: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 20260822,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(values.mean()), float(low), float(high)


def exact_sign_permutation(values: np.ndarray) -> tuple[float, float]:
    """Two-sided exact sign-randomization test over paired unit differences."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite values")
    observed = abs(values.mean())
    null = np.array(
        [np.mean(values * np.asarray(signs)) for signs in product([-1.0, 1.0], repeat=len(values))]
    )
    p_value = np.mean(np.abs(null) >= observed - 1e-15)
    return float(values.mean()), float(p_value)


def paired_bootstrap_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 20260822,
) -> tuple[float, float, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("Paired arrays must have identical shapes")
    return percentile_bootstrap_mean(first - second, n_resamples=n_resamples, seed=seed)


def exact_group_label_permutation(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float, int, int]:
    """Two-sided exact permutation of fixed-size group labels."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    values = np.concatenate([first, second])
    n_first = len(first)
    observed = float(first.mean() - second.mean())
    exceed = 0
    total = 0
    indices = range(len(values))
    for selected_tuple in combinations(indices, n_first):
        selected = set(selected_tuple)
        perm_first = np.array([values[i] for i in indices if i in selected])
        perm_second = np.array([values[i] for i in indices if i not in selected])
        difference = perm_first.mean() - perm_second.mean()
        exceed += int(abs(difference) >= abs(observed) - 1e-15)
        total += 1
    return observed, exceed / total, exceed, total
