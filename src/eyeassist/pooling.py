"""Held-out finite-panel reference-pool comparisons."""

from __future__ import annotations

from itertools import combinations, product
from typing import Callable, Mapping

import numpy as np

from .gaze import equal_reader_pool


def _mean_over_pools(
    pools: list[list[str]],
    maps: Mapping[str, np.ndarray],
    score: Callable[[np.ndarray], float],
) -> float:
    if not pools:
        raise ValueError("No admissible pools were generated")
    return float(np.mean([score(equal_reader_pool(maps, pool)) for pool in pools]))


def held_out_configuration_scores(
    *,
    maps: Mapping[str, np.ndarray],
    target_reader: str,
    state_by_reader: Mapping[str, str],
    identity_by_reader: Mapping[str, str] | None = None,
    score: Callable[[np.ndarray], float],
    matched_size: int,
    half_mixed_target: int,
    half_mixed_off_state: int,
    opposite_size: int,
) -> dict[str, float]:
    """Score matched, half-mixed, opposite and all-record pools.

    The target reader identity is excluded from every pool, including records
    from the same reader in another session. Half-mixed pools also require
    distinct contributing identities. All member maps are normalized and
    averaged with equal reader/record weight.
    """

    if target_reader not in state_by_reader:
        raise KeyError(f"Unknown target reader: {target_reader}")
    if identity_by_reader is None:
        identity_by_reader = {reader: reader for reader in maps}
    missing_identities = set(maps) - set(identity_by_reader)
    if missing_identities:
        raise KeyError(f"Missing reader identities: {sorted(missing_identities)}")
    target_state = state_by_reader[target_reader]
    target_identity = identity_by_reader[target_reader]
    available = [
        reader for reader in maps if identity_by_reader[reader] != target_identity
    ]
    same = [reader for reader in available if state_by_reader[reader] == target_state]
    opposite = [reader for reader in available if state_by_reader[reader] != target_state]

    matched_pools = [list(pool) for pool in combinations(same, matched_size)]
    opposite_pools = [list(pool) for pool in combinations(opposite, opposite_size)]
    half_pools = [
        list(left) + list(right)
        for left, right in product(
            combinations(same, half_mixed_target),
            combinations(opposite, half_mixed_off_state),
        )
        if len({identity_by_reader[reader] for reader in [*left, *right]})
        == len(left) + len(right)
    ]

    scores = {
        "matched": _mean_over_pools(matched_pools, maps, score),
        "half_mixed": _mean_over_pools(half_pools, maps, score),
        "opposite": _mean_over_pools(opposite_pools, maps, score),
        "all_records": score(equal_reader_pool(maps, available)),
    }
    scores.update(
        {
            "matched_minus_half_mixed": scores["matched"] - scores["half_mixed"],
            "matched_minus_opposite": scores["matched"] - scores["opposite"],
            "matched_minus_all_records": scores["matched"] - scores["all_records"],
        }
    )
    return scores
