# Data schema and QC ledger

## Case manifest

One row per imaging case:

| Field | Type | Required | Meaning |
|---|---|:---:|---|
| `case_id` | string | ✓ | stable de-identified imaging-case key |
| `image_path` | relative path | ✓ | local controlled-access path |
| `width`, `height` | integer | ✓ | displayed canvas in pixels |
| `label` | integer/string | ✓ | locked task label with provenance in the data dictionary |
| `subset` | category | ✓ | EyeAssist-Neo or EyeAssist-PE |
| `include` | 0/1 | ✓ | analysis inclusion flag |
| `exclusion_reason` | string | when excluded | prespecified/acquisition/QC reason |

An optional grouping field may be supplied when a study design requires grouping beyond the
imaging-case identifier.

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
