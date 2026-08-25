"""Reader-level behavioral summaries with explicit spatial grids."""

from __future__ import annotations

import numpy as np
import pandas as pd


def grid_entropy_coverage(
    fixations: pd.DataFrame,
    *,
    width: float,
    height: float,
    grid: int = 10,
) -> tuple[float, int]:
    """Shannon entropy and number of occupied cells on a clipped grid."""
    if width <= 0 or height <= 0 or grid <= 0:
        raise ValueError("width, height and grid must be positive")
    x = np.clip((fixations["x"].to_numpy(float) / width * grid).astype(int), 0, grid - 1)
    y = np.clip((fixations["y"].to_numpy(float) / height * grid).astype(int), 0, grid - 1)
    counts = np.bincount(y * grid + x, minlength=grid * grid)
    occupied = counts[counts > 0]
    if len(occupied) == 0:
        return float("nan"), 0
    probability = occupied / occupied.sum()
    entropy = -np.sum(probability * np.log2(probability))
    return float(entropy), int(len(occupied))


def mean_case_entropy_coverage(
    fixations: pd.DataFrame,
    *,
    width: float,
    height: float,
    grid: int = 10,
) -> tuple[float, float]:
    values = [
        grid_entropy_coverage(table, width=width, height=height, grid=grid)
        for _, table in fixations.groupby("case_id")
    ]
    return float(np.nanmean([value[0] for value in values])), float(
        np.nanmean([value[1] for value in values])
    )
