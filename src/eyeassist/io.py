"""Input normalization and data-integrity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


CANONICAL_FIXATION_COLUMNS = ["case_id", "reader_id", "x", "y", "duration", "fixation_index"]


def canonical_case_name(value: object) -> str:
    name = str(value).strip()
    return name.replace("60.jpeg", "60.jpg")


def read_fixations(
    path: str | Path,
    column_map: Mapping[str, str],
    reader_id: str | None = None,
) -> pd.DataFrame:
    """Read one fixation CSV into a stable, typed schema.

    `column_map` maps canonical names (case, x, y, duration, fixation_index,
    reader) to source-column names. Rows with missing coordinates or durations
    are removed and reported through the returned DataFrame's attrs.
    """

    path = Path(path)
    raw = pd.read_csv(path)
    required = ["case", "x", "y", "duration"]
    missing = [column_map[k] for k in required if column_map.get(k) not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing fixation columns {missing}")

    out = pd.DataFrame(
        {
            "case_id": raw[column_map["case"]].map(canonical_case_name),
            "x": pd.to_numeric(raw[column_map["x"]], errors="coerce"),
            "y": pd.to_numeric(raw[column_map["y"]], errors="coerce"),
            "duration": pd.to_numeric(raw[column_map["duration"]], errors="coerce"),
        }
    )
    if reader_id is not None:
        out["reader_id"] = reader_id
    elif column_map.get("reader") in raw.columns:
        out["reader_id"] = raw[column_map["reader"]].astype(str)
    else:
        raise ValueError(f"{path}: reader ID not supplied and reader column is absent")

    if column_map.get("fixation_index") in raw.columns:
        out["fixation_index"] = pd.to_numeric(
            raw[column_map["fixation_index"]], errors="coerce"
        )
    else:
        out["fixation_index"] = out.groupby(["case_id", "reader_id"]).cumcount() + 1

    before = len(out)
    out = out.dropna(subset=["case_id", "reader_id", "x", "y", "duration"]).copy()
    out = out[out["duration"] >= 0].copy()
    out.attrs["source"] = str(path)
    out.attrs["rows_input"] = before
    out.attrs["rows_retained"] = len(out)
    out.attrs["rows_removed"] = before - len(out)
    return out[CANONICAL_FIXATION_COLUMNS].sort_values(
        ["case_id", "reader_id", "fixation_index"]
    )


def concatenate_fixations(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(frames)
    if not frames:
        raise ValueError("No fixation tables were supplied")
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["case_id", "reader_id", "fixation_index"])


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"case_id", "image_path", "width", "height", "label"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if manifest["case_id"].isna().any():
        raise ValueError("case_id must be complete")
    if manifest["case_id"].duplicated().any():
        duplicates = manifest.loc[manifest["case_id"].duplicated(), "case_id"].tolist()
        raise ValueError(f"Duplicate case_id values: {duplicates[:5]}")
    if "include" in manifest:
        manifest = manifest[manifest["include"].astype(bool)].copy()
    manifest["width"] = pd.to_numeric(manifest["width"], errors="raise").astype(int)
    manifest["height"] = pd.to_numeric(manifest["height"], errors="raise").astype(int)
    if (manifest[["width", "height"]].to_numpy() <= 0).any():
        raise ValueError("Image dimensions must be positive")
    return manifest.reset_index(drop=True)


def validate_fixations_against_manifest(
    fixations: pd.DataFrame, manifest: pd.DataFrame
) -> dict[str, object]:
    known = set(manifest["case_id"].astype(str))
    observed = set(fixations["case_id"].astype(str))
    unknown = sorted(observed - known)
    absent = sorted(known - observed)
    dimensions = manifest.set_index("case_id")[["width", "height"]]
    merged = fixations.join(dimensions, on="case_id", how="left")
    outside = (
        (merged["x"] < 0)
        | (merged["y"] < 0)
        | (merged["x"] >= merged["width"])
        | (merged["y"] >= merged["height"])
    )
    return {
        "n_fixations": int(len(fixations)),
        "n_cases_manifest": int(len(manifest)),
        "n_cases_fixations": int(len(observed)),
        "unknown_cases": unknown,
        "cases_without_fixations": absent,
        "out_of_bounds_fixations": int(np.nansum(outside.to_numpy())),
        "pass": not unknown,
    }
