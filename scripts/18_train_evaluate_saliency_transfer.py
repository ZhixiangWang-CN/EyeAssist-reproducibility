#!/usr/bin/env python3
"""Train saliency-target arms and export held-out predictions locally.

The runner consumes only an authorized local manifest and partition table.
Checkpoints and predictions are written below the requested output directory
and are not part of the public source-code release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from eyeassist.classifier_pipeline import atomic_torch_save, resolve_local_path
from eyeassist.models import (
    available_device,
    make_saliency_model,
    saliency_objective,
    seed_everything,
)

TARGET_COLUMNS = {
    "expert_consensus": "saliency_expert_consensus_path",
    "generalist_consensus": "saliency_generalist_consensus_path",
    "pre_report": "saliency_pre_report_path",
    "post_report": "saliency_post_report_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--execution-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=[*TARGET_COLUMNS, "all"],
        default=["all"],
        help="Train selected target arms; the default trains all four arms",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--correlation-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pretrained-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tables(
    manifest_path: Path,
    split_path: Path,
    *,
    partition_id: int,
    data_root: Path | None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"case_id": str, "patient_id": str})
    splits = pd.read_csv(split_path, dtype={"case_id": str})
    required_manifest = {"case_id", "patient_id", "image_path", *TARGET_COLUMNS.values()}
    missing = required_manifest.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if manifest.case_id.duplicated().any():
        raise ValueError("Manifest must contain one row per case_id")

    id_column = "partition_id" if "partition_id" in splits else "split_id"
    required_splits = {id_column, "case_id", "partition"}
    missing = required_splits.difference(splits.columns)
    if missing:
        raise ValueError(f"Split table is missing columns: {sorted(missing)}")
    chosen = splits.loc[splits[id_column].astype(int) == int(partition_id)].copy()
    if chosen.empty:
        raise ValueError(f"Partition {partition_id} is absent from {split_path}")
    if chosen.case_id.duplicated().any():
        raise ValueError("Selected partition contains duplicate case_id rows")
    unexpected = set(chosen.partition) - {"train", "validation", "test"}
    if unexpected:
        raise ValueError(f"Unexpected partition values: {sorted(unexpected)}")
    if set(chosen.partition) != {"train", "validation", "test"}:
        raise ValueError("Selected partition must contain train, validation and test cases")

    table = chosen.merge(manifest, on="case_id", how="left", validate="one_to_one")
    if table.image_path.isna().any():
        absent = table.loc[table.image_path.isna(), "case_id"].tolist()
        raise ValueError(f"Cases missing from manifest: {absent[:5]}")
    table["resolved_image_path"] = table.image_path.map(
        lambda value: str(
            resolve_local_path(value, manifest_path=manifest_path, data_root=data_root)
        )
    )
    for target, column in TARGET_COLUMNS.items():
        if table[column].isna().any():
            raise ValueError(f"Target {target!r} has missing paths in {column!r}")
        table[f"resolved_{target}_path"] = table[column].map(
            lambda value: str(
                resolve_local_path(value, manifest_path=manifest_path, data_root=data_root)
            )
        )
    path_columns = ["resolved_image_path"] + [f"resolved_{key}_path" for key in TARGET_COLUMNS]
    for column in path_columns:
        absent = [value for value in table[column] if not Path(value).exists()]
        if absent:
            raise FileNotFoundError(f"Missing local files in {column}: {absent[:3]}")

    partitions = ("train", "validation", "test")
    for index, first in enumerate(partitions):
        first_patients = set(table.loc[table.partition == first, "patient_id"])
        for second in partitions[index + 1 :]:
            overlap = first_patients & set(table.loc[table.partition == second, "patient_id"])
            if overlap:
                raise ValueError(
                    f"Patient leakage between {first} and {second}: {sorted(overlap)[:5]}"
                )
    return table.reset_index(drop=True)


class SaliencyDataset:
    def __init__(self, table: pd.DataFrame, *, target: str, image_size: int) -> None:
        from torchvision import transforms

        self.table = table.reset_index(drop=True)
        self.target = target
        self.image_size = int(image_size)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.table)

    def _load_target(self, path: Path):
        import torch
        from torch.nn import functional

        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
        array = np.asarray(array, dtype=np.float32).squeeze()
        if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"Invalid saliency target: {path}")
        target = torch.from_numpy(array)[None, None]
        target = functional.interpolate(
            target,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mass = target.sum()
        if not torch.isfinite(mass) or mass <= 0:
            raise ValueError(f"Saliency target has zero or invalid mass: {path}")
        return target / mass

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        image = self.image_transform(Image.open(row.resolved_image_path).convert("RGB"))
        target_path = Path(row[f"resolved_{self.target}_path"])
        return {
            "case_id": str(row.case_id),
            "image": image,
            "target": self._load_target(target_path),
        }


def make_loader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def mean_loss(model, loader, *, device, kl_weight: float, correlation_weight: float) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            loss = saliency_objective(
                model(images)[:, 0],
                targets,
                kl_weight=kl_weight,
                correlation_weight=correlation_weight,
            )
            size = len(batch["case_id"])
            total += float(loss) * size
            count += size
    return total / count


def train_target(args: argparse.Namespace, table: pd.DataFrame, target: str, root: Path) -> None:
    import torch

    target_dir = root / target
    checkpoint_path = target_dir / "final_epoch.pt"
    history_path = target_dir / "history.jsonl"
    prediction_dir = target_dir / "test_predictions"
    index_path = target_dir / "test_predictions.csv"
    if any(path.exists() for path in (checkpoint_path, history_path, index_path)):
        raise FileExistsError(f"Refusing to overwrite existing run: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    train_table = table.loc[table.partition == "train"].copy()
    validation_table = table.loc[table.partition == "validation"].copy()
    test_table = table.loc[table.partition == "test"].copy()
    train_dataset = SaliencyDataset(train_table, target=target, image_size=args.image_size)
    validation_dataset = SaliencyDataset(
        validation_table, target=target, image_size=args.image_size
    )
    test_dataset = SaliencyDataset(test_table, target=target, image_size=args.image_size)

    seed_everything(args.seed)
    device = available_device()
    model = make_saliency_model(pretrained_encoder=args.pretrained_encoder).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    with history_path.open("w") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            for batch in train_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = saliency_objective(
                    model(images)[:, 0],
                    targets,
                    kl_weight=args.kl_weight,
                    correlation_weight=args.correlation_weight,
                )
                loss.backward()
                optimizer.step()
                size = len(batch["case_id"])
                running += float(loss.detach()) * size
                seen += size
            scheduler.step()
            record = {
                "epoch": epoch,
                "train_loss": running / seen,
                "validation_loss": mean_loss(
                    model,
                    validation_loader,
                    device=device,
                    kl_weight=args.kl_weight,
                    correlation_weight=args.correlation_weight,
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.write(json.dumps(record) + "\n")
            history.flush()

    payload = {
        "epoch": args.epochs,
        "checkpoint_rule": "final_epoch",
        "partition_id": args.partition_id,
        "execution_id": args.execution_id,
        "target": target,
        "seed": args.seed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "manifest_sha256": file_sha256(args.manifest),
        "splits_sha256": file_sha256(args.splits),
        "train_case_ids": sorted(train_table.case_id.tolist()),
        "validation_case_ids": sorted(validation_table.case_id.tolist()),
        "test_case_ids": sorted(test_table.case_id.tolist()),
        "run_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "kl_weight": args.kl_weight,
            "correlation_weight": args.correlation_weight,
            "pretrained_encoder": args.pretrained_encoder,
            "optimizer": "adam",
            "scheduler": "cosine_annealing",
        },
    }
    atomic_torch_save(payload, checkpoint_path)

    prediction_dir.mkdir(parents=True, exist_ok=True)
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            predictions = model(batch["image"].to(device))[:, 0].clamp_min(0)
            predictions = predictions / predictions.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            for case_id, prediction in zip(batch["case_id"], predictions.cpu().numpy()):
                path = prediction_dir / f"{case_id}.npy"
                np.save(path, prediction.astype(np.float32))
                rows.append(
                    {
                        "case": case_id,
                        "partition_id": args.partition_id,
                        "execution_id": args.execution_id,
                        "trained_on": target,
                        "checkpoint_epoch": args.epochs,
                        "checkpoint_rule": "final_epoch",
                        "prediction_path": str(path.relative_to(target_dir)),
                    }
                )
    pd.DataFrame(rows).sort_values("case").to_csv(index_path, index=False)


def main() -> None:
    args = parse_args()
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install model dependencies with: pip install -e '.[models]'") from exc
    if args.epochs <= 0 or args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("epochs, batch-size and image-size must be positive")
    if args.execution_id < 0:
        raise ValueError("execution-id cannot be negative")
    targets = list(TARGET_COLUMNS) if "all" in args.targets else list(dict.fromkeys(args.targets))

    args.manifest = args.manifest.expanduser().resolve()
    args.splits = args.splits.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    table = load_tables(
        args.manifest,
        args.splits,
        partition_id=args.partition_id,
        data_root=data_root,
    )
    root = (
        output_dir
        / f"partition_{args.partition_id:03d}"
        / f"execution_{args.execution_id:02d}"
    )
    for target in targets:
        train_target(args, table, target, root)


if __name__ == "__main__":
    main()
