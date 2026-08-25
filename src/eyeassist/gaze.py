"""Fixation-density construction, alignment and saliency metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


EPS = 1e-12


def _coordinates(
    fixations: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(fixations, pd.DataFrame):
        return (
            fixations["x"].to_numpy(float),
            fixations["y"].to_numpy(float),
            fixations["duration"].to_numpy(float),
        )
    array = np.asarray(fixations, dtype=float)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("Fixations must have shape (n,2) or (n,3)")
    duration = array[:, 2] if array.shape[1] == 3 else np.ones(len(array))
    return array[:, 0], array[:, 1], duration


def density_map(
    fixations: pd.DataFrame | np.ndarray,
    image_shape: tuple[int, int],
    *,
    downsample_factor: int = 4,
    sigma_pixels: float = 40.0,
    smoothing_mass: float = 0.01,
    weighting: str = "duration",
    coordinate_policy: str = "clip",
) -> np.ndarray:
    """Build a unit-mass fixation density map.

    The implementation follows the packaged EyeAssist analysis: coordinates are
    histogrammed on a fourfold-downsampled array, smoothed with a Gaussian kernel,
    normalized, and mixed with a small uniform component to keep log scores finite.
    """

    height, width = map(int, image_shape)
    if height <= 0 or width <= 0 or downsample_factor <= 0:
        raise ValueError("Image dimensions and downsample_factor must be positive")
    if not 0 <= smoothing_mass < 1:
        raise ValueError("smoothing_mass must be in [0,1)")

    x, y, duration = _coordinates(fixations)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(duration)
    x, y, duration = x[finite], y[finite], duration[finite]
    grid_h = height // downsample_factor + 1
    grid_w = width // downsample_factor + 1

    if coordinate_policy == "discard":
        keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y, duration = x[keep], y[keep], duration[keep]
    elif coordinate_policy != "clip":
        raise ValueError("coordinate_policy must be 'clip' or 'discard'")

    weights = duration if weighting == "duration" else np.ones(len(x))
    if weighting not in {"duration", "event"}:
        raise ValueError("weighting must be 'duration' or 'event'")
    density = np.zeros((grid_h, grid_w), dtype=float)
    if len(x):
        xi = np.clip((x / downsample_factor).astype(int), 0, grid_w - 1)
        yi = np.clip((y / downsample_factor).astype(int), 0, grid_h - 1)
        np.add.at(density, (yi, xi), weights)
        density = gaussian_filter(density, sigma_pixels / downsample_factor)

    total = density.sum()
    if total <= 0:
        return np.full_like(density, 1.0 / density.size)
    density /= total
    density = (1.0 - smoothing_mass) * density + smoothing_mass / density.size
    return density / density.sum()


def equal_reader_pool(maps: Mapping[str, np.ndarray], readers: list[str]) -> np.ndarray:
    if not readers:
        raise ValueError("A pool must contain at least one reader")
    selected = [np.asarray(maps[reader], dtype=float) for reader in readers]
    if len({array.shape for array in selected}) != 1:
        raise ValueError("All member maps must have the same shape")
    normalized = [array / max(array.sum(), EPS) for array in selected]
    pool = np.mean(normalized, axis=0)
    return pool / max(pool.sum(), EPS)


def fixation_log_score(
    fixations: pd.DataFrame | np.ndarray,
    probability: np.ndarray,
    *,
    downsample_factor: int = 4,
    base: float = 2.0,
    duration_weighted: bool = False,
) -> float:
    x, y, duration = _coordinates(fixations)
    height, width = probability.shape
    xi = np.clip((x / downsample_factor).astype(int), 0, width - 1)
    yi = np.clip((y / downsample_factor).astype(int), 0, height - 1)
    values = np.log(np.clip(probability[yi, xi], EPS, None)) / np.log(base)
    if duration_weighted:
        return float(np.average(values, weights=duration))
    return float(values.mean())


def center_of_mass(fixations: pd.DataFrame | np.ndarray, duration_weighted: bool = True) -> np.ndarray:
    x, y, duration = _coordinates(fixations)
    if len(x) == 0:
        return np.array([np.nan, np.nan])
    weights = duration if duration_weighted else np.ones(len(x))
    return np.array([np.average(x, weights=weights), np.average(y, weights=weights)])


def reader_offset(session_1: pd.DataFrame, session_2: pd.DataFrame) -> np.ndarray:
    """Mean case-paired translation from session 1 to session 2."""
    common = sorted(set(session_1["case_id"]) & set(session_2["case_id"]))
    if not common:
        raise ValueError("The two sessions have no common cases")
    shifts = []
    for case in common:
        first = session_1[session_1["case_id"] == case]
        second = session_2[session_2["case_id"] == case]
        shifts.append(center_of_mass(second) - center_of_mass(first))
    return np.nanmean(shifts, axis=0)


def leave_one_case_out_offsets(
    session_1: pd.DataFrame, session_2: pd.DataFrame
) -> dict[str, np.ndarray]:
    common = sorted(set(session_1["case_id"]) & set(session_2["case_id"]))
    if len(common) < 2:
        raise ValueError("At least two paired cases are required for cross-fitted alignment")
    offsets: dict[str, np.ndarray] = {}
    for held_out in common:
        first = session_1[session_1["case_id"] != held_out]
        second = session_2[session_2["case_id"] != held_out]
        offsets[held_out] = reader_offset(first, second)
    return offsets


def translate_fixations(fixations: pd.DataFrame, offset: np.ndarray) -> pd.DataFrame:
    result = fixations.copy()
    result["x"] = result["x"] - float(offset[0])
    result["y"] = result["y"] - float(offset[1])
    return result


def _unit_mass(array: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(array, dtype=float), 0, None)
    return array / max(array.sum(), EPS)


def nss(prediction: np.ndarray, fixation_mask: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    fixation_mask = np.asarray(fixation_mask, dtype=float)
    if prediction.shape != fixation_mask.shape:
        raise ValueError("prediction and fixation_mask must have identical shapes")
    sd = prediction.std()
    if sd <= EPS or fixation_mask.sum() <= 0:
        return float("nan")
    z = (prediction - prediction.mean()) / sd
    return float((z * fixation_mask).sum() / fixation_mask.sum())


def pearson_cc(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    if first.std() <= EPS or second.std() <= EPS:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.minimum(_unit_mass(first), _unit_mass(second)).sum())


def kl_divergence(target: np.ndarray, model: np.ndarray) -> float:
    target = _unit_mass(target)
    model = _unit_mass(model)
    return float(np.sum(target * (np.log(target + EPS) - np.log(model + EPS))))
