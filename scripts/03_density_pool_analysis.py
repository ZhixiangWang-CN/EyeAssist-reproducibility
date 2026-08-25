#!/usr/bin/env python3
"""Configuration-driven held-out gaze-pool analysis.

This script expects one canonical fixation CSV per reader and a case manifest.
It produces case-reader-level matched/half-mixed/opposite/all-record scores so
all manuscript summaries can be regenerated from an auditable intermediate.
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import pandas as pd

from eyeassist.config import load_config
from eyeassist.gaze import density_map, fixation_log_score
from eyeassist.io import read_fixations, read_manifest
from eyeassist.pooling import held_out_configuration_scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis.yaml")
    parser.add_argument("--axis", choices=["reader_group", "session_condition"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(args.config).resolve().parent.parent
    manifest_path = Path(config["data"]["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = read_manifest(manifest_path).set_index("case_id")
    columns = config["fixation_columns"]

    if args.axis == "reader_group":
        sources = {
            "subspecialist": config["data"]["fixation_files"]["reader_group_subspecialist"],
            "general_radiologist": config["data"]["fixation_files"]["reader_group_general_radiologist"],
        }
    else:
        sources = {
            "session_1": config["data"]["fixation_files"]["session_1"],
            "session_2": config["data"]["fixation_files"]["session_2"],
        }

    tables: dict[str, pd.DataFrame] = {}
    states: dict[str, str] = {}
    identities: dict[str, str] = {}
    for state, pattern in sources.items():
        path_pattern = str(root / pattern) if not Path(pattern).is_absolute() else pattern
        for path_string in sorted(glob(path_pattern)):
            path = Path(path_string)
            identity = path.stem.replace("_fixations", "")
            reader = f"{state}:{identity}"
            tables[reader] = read_fixations(path, columns, reader_id=reader)
            states[reader] = state
            identities[reader] = identity

    if not tables:
        raise FileNotFoundError("No fixation CSV files matched the configured patterns")

    density_config = config["density"]
    pool_config = config["pool_comparison"][args.axis]
    rows = []
    for case_id, meta in manifest.iterrows():
        case_tables = {
            reader: table[table["case_id"] == str(case_id)] for reader, table in tables.items()
        }
        case_tables = {reader: table for reader, table in case_tables.items() if len(table)}
        maps = {
            reader: density_map(
                table,
                (int(meta.height), int(meta.width)),
                downsample_factor=density_config["downsample_factor"],
                sigma_pixels=density_config["sigma_pixels"],
                smoothing_mass=density_config["smoothing_mass"],
                weighting=density_config["weighting"],
                coordinate_policy=density_config["coordinate_policy"],
            )
            for reader, table in case_tables.items()
        }
        for target_reader, target_fixations in case_tables.items():
            score = lambda probability, f=target_fixations: fixation_log_score(
                f,
                probability,
                downsample_factor=density_config["downsample_factor"],
                base=config["pool_comparison"]["log_base"],
            )
            values = held_out_configuration_scores(
                maps=maps,
                target_reader=target_reader,
                state_by_reader=states,
                identity_by_reader=identities,
                score=score,
                matched_size=pool_config["matched_size"],
                half_mixed_target=pool_config["half_mixed_target"],
                half_mixed_off_state=pool_config["half_mixed_off_state"],
                opposite_size=pool_config["opposite_size"],
            )
            rows.append(
                {
                    "case_id": case_id,
                    "target_reader": target_reader,
                    "target_identity": identities[target_reader],
                    "target_state": states[target_reader],
                    **values,
                }
            )

    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows):,} held-out case-reader rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
