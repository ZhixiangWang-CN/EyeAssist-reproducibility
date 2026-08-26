"""Data, metrics and checkpoint utilities for the ResNet-50 comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

ARMS = {
    "image_only": None,
    "generalist_gaze": "gaze_generalist_path",
    "cold_read_gaze": "gaze_cold_read_path",
    "informed_gaze": "gaze_informed_path",
}


def resolve_local_path(value: object, *, manifest_path: Path, data_root: Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if data_root is not None:
        candidates.append(data_root / path)
    candidates.extend([manifest_path.parent / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_case_and_split_tables(
    manifest_path: Path,
    split_path: Path,
    *,
    split_id: int,
    arm: str,
    data_root: Path | None = None,
) -> pd.DataFrame:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str})
    splits = pd.read_csv(split_path, dtype={"case_id": str})
    required_manifest = {"case_id", "image_path", "label"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    required_splits = {"split_id", "case_id", "partition"}
    missing = required_splits - set(splits.columns)
    if missing:
        raise ValueError(f"Split table is missing columns: {sorted(missing)}")
    if manifest.case_id.duplicated().any():
        raise ValueError("Manifest must contain one row per case_id")
    chosen = splits.loc[splits.split_id.astype(int) == int(split_id)].copy()
    if chosen.empty:
        raise ValueError(f"split_id {split_id} is absent from {split_path}")
    if chosen.case_id.duplicated().any():
        raise ValueError("The selected split contains duplicate case_id rows")
    unknown_partition = set(chosen.partition) - {"train", "test"}
    if unknown_partition:
        raise ValueError(f"Unexpected partition values: {sorted(unknown_partition)}")
    table = chosen.merge(manifest, on="case_id", how="left", validate="one_to_one")
    if table.image_path.isna().any():
        missing_cases = table.loc[table.image_path.isna(), "case_id"].tolist()
        raise ValueError(f"Cases missing from manifest: {missing_cases[:5]}")
    table["label"] = pd.to_numeric(table.label, errors="raise").astype(int)
    if set(table.label) - {0, 1}:
        raise ValueError("Classifier labels must be binary integers 0/1")
    table["resolved_image_path"] = table.image_path.map(
        lambda value: str(
            resolve_local_path(value, manifest_path=manifest_path, data_root=data_root)
        )
    )
    gaze_column = ARMS[arm]
    if gaze_column is not None:
        if gaze_column not in table:
            raise ValueError(
                f"Arm {arm!r} requires manifest column {gaze_column!r}; "
                "each value must point to a normalized gaze target"
            )
        if table[gaze_column].isna().any():
            raise ValueError(f"Arm {arm!r} has missing values in {gaze_column!r}")
        table["resolved_gaze_path"] = table[gaze_column].map(
            lambda value: str(
                resolve_local_path(value, manifest_path=manifest_path, data_root=data_root)
            )
        )
    else:
        table["resolved_gaze_path"] = ""
    for column in ("resolved_image_path", "resolved_gaze_path"):
        if column == "resolved_gaze_path" and gaze_column is None:
            continue
        missing_paths = [value for value in table[column] if not Path(value).exists()]
        if missing_paths:
            raise FileNotFoundError(f"Missing input files in {column}: {missing_paths[:3]}")
    train_cases = set(table.loc[table.partition == "train", "case_id"])
    test_cases = set(table.loc[table.partition == "test", "case_id"])
    if train_cases & test_cases:
        raise ValueError("Case leakage detected between train and test partitions")
    return table.reset_index(drop=True)


class ClassifierDataset:
    """Load radiographs and optional gaze targets from a case table."""

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        image_size: int,
        training: bool,
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        from torchvision import transforms

        if not 0 <= horizontal_flip_probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        self.table = table.reset_index(drop=True)
        operations: list[Any] = [transforms.Resize((image_size, image_size))]
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.image_transform = transforms.Compose(operations)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def __len__(self) -> int:
        return len(self.table)

    def _load_gaze(self, path: str):
        import torch
        from torch.nn import functional

        if not path:
            return torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
        source = Path(path)
        if source.suffix.lower() == ".npy":
            array = np.load(source)
        else:
            array = np.asarray(Image.open(source).convert("F"), dtype=np.float32)
        array = np.asarray(array, dtype=np.float32).squeeze()
        if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"Invalid gaze target: {source}")
        target = torch.from_numpy(array)[None, None]
        target = functional.interpolate(
            target,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        total = target.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError(f"Gaze target has zero or invalid mass: {source}")
        return target / total

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.table.iloc[index]
        image = Image.open(row.resolved_image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        gaze = self._load_gaze(str(row.resolved_gaze_path))
        if self.training and torch.rand(()) < self.horizontal_flip_probability:
            image_tensor = torch.flip(image_tensor, dims=(2,))
            gaze = torch.flip(gaze, dims=(1,))
        return {
            "case_id": str(row.case_id),
            "image": image_tensor,
            "label": int(row.label),
            "gaze": gaze,
        }


def classifier_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("Labels and probabilities must be non-empty and equally sized")
    if set(labels) != {0, 1}:
        raise ValueError("Both binary classes are required to calculate AUROC")
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity) if sensitivity is not None else None,
        "specificity": float(specificity) if specificity is not None else None,
        "ppv": float(ppv) if ppv is not None else None,
        "npv": float(npv) if npv is not None else None,
    }


def checkpoint_is_better(
    *,
    rule: str,
    epoch: int,
    max_epochs: int,
    validation_metrics: dict[str, float] | None,
    best_value: float | None,
) -> tuple[bool, float | None]:
    if rule == "last_epoch":
        return epoch == max_epochs, None
    if validation_metrics is None:
        raise ValueError(f"Checkpoint rule {rule!r} requires a training-only validation set")
    if rule == "best_val_loss":
        value = float(validation_metrics["loss"])
        return best_value is None or value < best_value, value
    if rule == "best_val_auroc":
        value = float(validation_metrics["auroc"])
        return best_value is None or value > best_value, value
    raise ValueError(f"Unknown checkpoint rule: {rule}")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
