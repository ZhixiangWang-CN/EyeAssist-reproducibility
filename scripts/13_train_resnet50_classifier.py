#!/usr/bin/env python3
"""Train one ResNet-50 arm for 60 epochs and save the final-epoch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eyeassist.classifier_pipeline import (
    ARMS,
    FINAL_CLASSIFIER_EPOCH,
    ClassifierDataset,
    atomic_torch_save,
    load_case_and_split_tables,
    write_json,
)
from eyeassist.models import (
    attention_kl,
    classifier_forward_with_cam,
    make_classifier,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-id", type=int, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gaze-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize from ImageNet-1K weights (default: true)",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument(
        "--cam-class",
        choices=["true_label", "positive_class"],
        default="true_label",
        help="Class whose differentiable CAM is matched to gaze during training",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def forward_loss(model, batch, *, arm: str, gaze_weight: float, cam_class: str, device):
    import torch
    from torch.nn import functional

    images = batch["image"].to(device)
    labels = batch["label"].to(device, dtype=torch.long)
    class_index = labels if cam_class == "true_label" else torch.ones_like(labels)
    logits, cam = classifier_forward_with_cam(model, images, class_index)
    classification_loss = functional.cross_entropy(logits, labels)
    gaze_loss = torch.zeros((), device=device)
    if arm != "image_only":
        gaze = batch["gaze"].to(device, dtype=torch.float32)
        gaze = functional.interpolate(
            gaze[:, None], size=cam.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        gaze_loss = attention_kl(cam, gaze)
    total = classification_loss + gaze_weight * gaze_loss
    return total, classification_loss, gaze_loss, logits, labels


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install model dependencies with: pip install -e '.[models]'") from exc
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.arm == "image_only" and args.gaze_loss_weight != 0.5:
        print("Note: gaze-loss-weight is ignored for the image_only arm")

    manifest = args.manifest.expanduser().resolve()
    splits = args.splits.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    table = load_case_and_split_tables(
        manifest,
        splits,
        split_id=args.split_id,
        arm=args.arm,
        data_root=args.data_root,
    )
    train_table = table.loc[table.partition == "train"].copy()
    test_table = table.loc[table.partition == "test"].copy()
    if train_table.empty or test_table.empty:
        raise ValueError("Both train and test partitions must be present")

    if set(train_table.case_id) & set(test_table.case_id):
        raise AssertionError("Test leakage detected")

    train_dataset = ClassifierDataset(
        train_table,
        image_size=args.image_size,
        training=True,
        horizontal_flip_probability=args.horizontal_flip_probability,
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_classifier(pretrained=args.pretrained, n_classes=2).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINAL_CLASSIFIER_EPOCH)
        if args.scheduler == "cosine"
        else None
    )

    run_dir = output_dir / f"split_{args.split_id:03d}" / args.arm
    last_path = run_dir / "last.pt"
    final_path = run_dir / "final.pt"
    history_path = run_dir / "history.jsonl"
    start_epoch = 1
    if last_path.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to mix or overwrite an existing run at {run_dir}; "
            "use --resume or choose a new --output-dir"
        )
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Cannot resume; missing {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["split_id"] != args.split_id or checkpoint["arm"] != args.arm:
            raise ValueError("Resume checkpoint does not match split/arm")
        if checkpoint.get("format_version") != 2:
            raise ValueError("Resume checkpoint uses an incompatible format")
        previous = checkpoint["run_config"]
        locked = {
            "manifest_sha256": file_sha256(manifest),
            "splits_sha256": file_sha256(splits),
            "epochs": FINAL_CLASSIFIER_EPOCH,
            "seed": args.seed,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gaze_loss_weight": args.gaze_loss_weight,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "scheduler": args.scheduler,
            "cam_class": args.cam_class,
            "pretrained": args.pretrained,
        }
        mismatches = {
            key: (previous.get(key), value)
            for key, value in locked.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Resume configuration differs from checkpoint: {mismatches}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > FINAL_CLASSIFIER_EPOCH:
            raise ValueError(
                f"Checkpoint already reached epoch {checkpoint['epoch']}; "
                f"protocol endpoint={FINAL_CLASSIFIER_EPOCH}"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update(
        {
            "manifest": str(manifest),
            "splits": str(splits),
            "manifest_sha256": file_sha256(manifest),
            "splits_sha256": file_sha256(splits),
            "epochs": FINAL_CLASSIFIER_EPOCH,
            "train_cases": sorted(train_table.case_id.astype(str)),
            "test_cases_held_out": sorted(test_table.case_id.astype(str)),
            "device": str(device),
        }
    )
    write_json(run_config, run_dir / "run_config.json")

    for epoch in range(start_epoch, FINAL_CLASSIFIER_EPOCH + 1):
        seed_everything(args.seed + epoch)
        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            num_workers=args.num_workers,
        )
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            total, _, _, _, _ = forward_loss(
                model,
                batch,
                arm=args.arm,
                gaze_weight=args.gaze_loss_weight,
                cam_class=args.cam_class,
                device=device,
            )
            total.backward()
            optimizer.step()
            epoch_losses.append(float(total.detach().cpu()))
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "format_version": 2,
            "epoch": epoch,
            "split_id": args.split_id,
            "arm": args.arm,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "run_config": run_config,
        }
        atomic_torch_save(payload, last_path)
        if epoch == FINAL_CLASSIFIER_EPOCH:
            atomic_torch_save(payload, final_path)
        print(json.dumps(record))

    if not final_path.exists():
        raise RuntimeError("Training finished without producing final.pt")
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
