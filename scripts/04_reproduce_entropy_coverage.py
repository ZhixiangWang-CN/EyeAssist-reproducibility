#!/usr/bin/env python3
"""Reproduce reader-group entropy and 10x10 coverage from fixation CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eyeassist.behavior import mean_case_entropy_coverage
from eyeassist.io import read_fixations
from eyeassist.statistics import exact_group_label_permutation


COLUMNS = {
    "case": "Source_File",
    "x": "CURRENT_FIX_X",
    "y": "CURRENT_FIX_Y",
    "duration": "CURRENT_FIX_DURATION",
    "fixation_index": "CURRENT_FIX_INDEX",
    "reader": "RECORDING_SESSION_LABEL",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="directory containing Expert/ and General/")
    parser.add_argument("--width", type=float, default=2000.0)
    parser.add_argument("--height", type=float, default=2800.0)
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs/entropy_coverage"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    patterns = {
        "subspecialist": "Expert/expert*_fixations.csv",
        "general_radiologist": "General/generalist*_fixations.csv",
    }
    for group, pattern in patterns.items():
        for path in sorted(args.root.glob(pattern)):
            reader = path.stem.replace("_fixations", "")
            fixations = read_fixations(path, COLUMNS, reader_id=reader)
            entropy, coverage = mean_case_entropy_coverage(
                fixations, width=args.width, height=args.height, grid=args.grid
            )
            rows.append(
                {
                    "group": group,
                    "reader_id": reader,
                    "entropy": entropy,
                    "coverage": coverage,
                    "fixations": len(fixations),
                    "cases": fixations.case_id.nunique(),
                }
            )

    table = pd.DataFrame(rows)
    if table.groupby("group").size().to_dict() != {
        "general_radiologist": 5,
        "subspecialist": 5,
    }:
        raise ValueError("Expected five subspecialist and five general-radiologist files")
    table.to_csv(args.output / "reader_level.csv", index=False)

    summary = {}
    for metric in ["entropy", "coverage"]:
        first = table.loc[table.group == "subspecialist", metric].to_numpy()
        second = table.loc[table.group == "general_radiologist", metric].to_numpy()
        difference, p_value, exceed, total = exact_group_label_permutation(first, second)
        summary[metric] = {
            "subspecialist_mean": float(first.mean()),
            "general_radiologist_mean": float(second.mean()),
            "difference": difference,
            "exact_p": p_value,
            "exceed": exceed,
            "permutations": total,
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
