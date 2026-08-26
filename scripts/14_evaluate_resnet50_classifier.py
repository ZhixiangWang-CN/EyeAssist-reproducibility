#!/usr/bin/env python3
"""Evaluate one saved ResNet-50 checkpoint once on its held-out test partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from eyeassist.classifier_pipeline import (
    ARMS,
    ClassifierDataset,
    classifier_metrics,
    load_case_and_split_tables,
    write_json,
)
from eyeassist.models import make_classifier, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split-id", type=int, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def choose_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install model dependencies with: pip install -e '.[models]'") from exc
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["split_id"]) != args.split_id or checkpoint["arm"] != args.arm:
        raise ValueError("Checkpoint split/arm does not match the requested evaluation")
    run_config = checkpoint["run_config"]
    if checkpoint_path.name != "selected.pt":
        raise ValueError("Evaluation requires the explicitly selected checkpoint named selected.pt")

    manifest = args.manifest.expanduser().resolve()
    splits = args.splits.expanduser().resolve()
    if file_sha256(manifest) != run_config["manifest_sha256"]:
        raise ValueError("Manifest contents differ from the table locked in the checkpoint")
    if file_sha256(splits) != run_config["splits_sha256"]:
        raise ValueError("Split-table contents differ from the table locked in the checkpoint")
    table = load_case_and_split_tables(
        manifest,
        splits,
        split_id=args.split_id,
        arm="image_only",
        data_root=args.data_root,
    )
    test_table = table.loc[table.partition == "test"].copy()
    expected_test = set(map(str, run_config["test_cases_held_out"]))
    observed_test = set(test_table.case_id.astype(str))
    if observed_test != expected_test:
        raise ValueError("Current test cases do not match the cases locked in the checkpoint")
    if set(test_table.case_id.astype(str)) & set(map(str, run_config["train_cases"])):
        raise ValueError("Checkpoint metadata indicates train/test leakage")

    seed_everything(int(run_config["seed"]))
    model = make_classifier(pretrained=False, n_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = ClassifierDataset(
        test_table,
        image_size=int(run_config["image_size"]),
        training=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = batch["label"].cpu().numpy()
            for case_id, label, score in zip(batch["case_id"], labels, probability):
                rows.append(
                    {
                        "split_id": args.split_id,
                        "arm": args.arm,
                        "case_id": str(case_id),
                        "y_true": int(label),
                        "probability_abnormal": float(score),
                        "checkpoint_epoch": int(checkpoint["epoch"]),
                        "checkpoint_rule": str(checkpoint["checkpoint_rule"]),
                    }
                )
    predictions = pd.DataFrame(rows).sort_values("case_id")
    if len(predictions) != len(test_table) or predictions.case_id.duplicated().any():
        raise RuntimeError("Test prediction output is incomplete or duplicated")
    metrics = classifier_metrics(
        predictions.y_true.to_numpy(), predictions.probability_abnormal.to_numpy()
    )
    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    report = {
        "split_id": args.split_id,
        "arm": args.arm,
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_rule": str(checkpoint["checkpoint_rule"]),
        "n_test_cases": len(predictions),
        "test_cases_used_for_checkpoint_selection": False,
        "metrics": metrics,
    }
    write_json(report, output_csv.with_suffix(".metrics.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
