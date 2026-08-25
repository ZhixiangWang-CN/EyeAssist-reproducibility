"""Group-disjoint, label-stratified repeated partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _group_table(manifest: pd.DataFrame, group_column: str, label_column: str) -> pd.DataFrame:
    required = {"case_id", group_column, label_column}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing split columns: {missing}")
    consistency = manifest.groupby(group_column)[label_column].nunique(dropna=False)
    inconsistent = consistency[consistency != 1]
    if len(inconsistent):
        raise ValueError(
            "Group-level stratification requires one label per group; inconsistent IDs: "
            + ", ".join(map(str, inconsistent.index[:5]))
        )
    return manifest[[group_column, label_column]].drop_duplicates(group_column).reset_index(drop=True)


def repeated_group_stratified_splits(
    manifest: pd.DataFrame,
    *,
    repeats: int,
    test_groups: int | float,
    seed: int,
    group_column: str = "case_id",
    label_column: str = "label",
) -> pd.DataFrame:
    groups = _group_table(manifest, group_column, label_column)
    splitter = StratifiedShuffleSplit(
        n_splits=repeats, test_size=test_groups, random_state=seed
    )
    rows: list[dict[str, object]] = []
    dummy = np.zeros(len(groups))
    for split_id, (train_idx, test_idx) in enumerate(
        splitter.split(dummy, groups[label_column].to_numpy())
    ):
        train_ids = set(groups.iloc[train_idx][group_column])
        test_ids = set(groups.iloc[test_idx][group_column])
        if train_ids & test_ids:
            raise AssertionError("Group leakage detected")
        for row in manifest.itertuples(index=False):
            group = getattr(row, group_column)
            partition = "train" if group in train_ids else "test"
            rows.append(
                {
                    "split_id": split_id,
                    "case_id": row.case_id,
                    group_column: group,
                    label_column: getattr(row, label_column),
                    "partition": partition,
                }
            )
    result = pd.DataFrame(rows)
    assert_group_disjoint(result, group_column=group_column)
    return result


def assert_group_disjoint(splits: pd.DataFrame, group_column: str = "case_id") -> None:
    counts = splits.groupby(["split_id", group_column])["partition"].nunique()
    leaking = counts[counts > 1]
    if len(leaking):
        raise ValueError(f"Group leakage in {len(leaking)} split-group combinations")
