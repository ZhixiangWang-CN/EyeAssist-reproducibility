"""Command-line entry points for validation and deterministic split creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, unresolved_fields
from .io import read_manifest
from .splits import repeated_group_stratified_splits


def _validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest_path = Path(config["data"]["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = Path(args.config).resolve().parent.parent / manifest_path
    manifest = read_manifest(manifest_path)
    group_column = config["splits"]["group_column"]
    if group_column not in manifest:
        raise ValueError(f"Configured grouping column is absent: {group_column}")
    groups = manifest[group_column].nunique()
    report = {
        "manifest": str(manifest_path),
        "cases": int(len(manifest)),
        "group_column": group_column,
        "unique_groups": int(groups),
        "labels": manifest["label"].value_counts(dropna=False).to_dict(),
        "unresolved_configuration": unresolved_fields(config),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if not report["unresolved_configuration"] else 2


def _make_splits(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = Path(args.config).resolve().parent.parent
    manifest_path = Path(config["data"]["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = read_manifest(manifest_path)
    split_config = config["splits"][args.analysis]
    split_count = split_config.get("repeats", split_config.get("partitions"))
    if split_count is None:
        raise ValueError(
            f"Split configuration for {args.analysis!r} requires 'repeats' or 'partitions'"
        )
    splits = repeated_group_stratified_splits(
        manifest,
        repeats=int(split_count),
        test_groups=args.test_groups,
        seed=int(config["project"]["seed"]),
        group_column=config["splits"]["group_column"],
        label_column=config["splits"]["label_column"],
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(output, index=False)
    print(f"Wrote {len(splits):,} split-case rows to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eyeassist")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="validate manifest and release gates")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_validate)

    split = sub.add_parser("make-splits", help="create group-disjoint repeated splits")
    split.add_argument("--config", required=True)
    split.add_argument("--analysis", choices=["classifier", "saliency"], default="classifier")
    split.add_argument("--test-groups", type=int, required=True)
    split.add_argument("--output", default="outputs/splits.csv")
    split.set_defaults(func=_make_splits)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
