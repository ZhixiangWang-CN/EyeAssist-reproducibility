# Data schema and QC ledger

## Case manifest

One row per imaging case:

| Field | Type | Required | Meaning |
|---|---|:---:|---|
| `case_id` | string | ✓ | stable de-identified imaging-case key |
| `patient_id` | string | ✓ | stable de-identified patient key; one-to-one with `case_id` in EyeAssist-Neo |
| `image_path` | relative path | ✓ | local controlled-access path |
| `width`, `height` | integer | ✓ | displayed canvas in pixels |
| `label` | integer/string | ✓ | locked task label with provenance in the data dictionary |
| `subset` | category | ✓ | EyeAssist-Neo or EyeAssist-PE |
| `include` | 0/1 | ✓ | analysis inclusion flag |
| `exclusion_reason` | string | when excluded | prespecified/acquisition/QC reason |

For ResNet-50 training, the three gaze-supervised arms use the following local-only manifest
columns. They are not required for the image-only arm.

| Field | Required arm | Meaning |
|---|---|---|
| `gaze_generalist_path` | generalist gaze | path to the normalized general-radiologist target |
| `gaze_cold_read_path` | first-session gaze | path to the normalized first-session target |
| `gaze_informed_path` | second-session gaze | path to the normalized second-session target |

Targets may be non-negative two-dimensional NumPy arrays or grayscale images. The loader resizes
each target to the model input grid, renormalizes it to unit mass and applies any configured
horizontal flip jointly to the radiograph and target.

Computational partitions use `patient_id` as the grouping field. In EyeAssist-Neo, the 75 cases
map one-to-one to 75 patients, so patient-grouped and case-grouped partitions are identical.

## Canonical fixation table

| Field | Type | Meaning |
|---|---|---|
| `case_id` | string | joins the case manifest |
| `reader_id` | string | de-identified reader/recording key |
| `x`, `y` | float | displayed-image coordinates |
| `duration` | float | fixation duration in the acquisition unit |
| `fixation_index` | integer | within-recording order |
| `state` | category | reader group or session state |

The loader accepts the packaged EyeAssist-Neo column names through the YAML column mapping. For
EyeAssist-PE, preserve scan ID, timestamp and synchronized slice index in a separate ordered
screen-plane gaze table.

## Required QC ledger

For every metric, record:

- designed reader-case pairs;
- acquired recordings;
- excluded recordings and exact reason;
- retained fixation events;
- out-of-bounds events and whether clipped or discarded;
- missing/ambiguous decision calls;
- final analysis unit and resampling unit.

The ledger should report the analysis-specific complete EyeAssist-PE pair count and the exact
stream-availability or metric-completeness rule without relying on a generic “after quality
control” phrase.
