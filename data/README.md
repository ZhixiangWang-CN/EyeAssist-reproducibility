# Data placement

No clinical data are committed to this repository.

Create `data/private/` locally and place controlled-access files under the paths declared in
`configs/analysis.yaml`. The directory is ignored by Git. Every analysis requires a manifest with
one row per imaging case. The grouping field used for a particular analysis is declared explicitly
in `configs/analysis.yaml`.

Required fixation fields for the packaged EyeAssist-Neo format are:

- `Source_File`
- `CURRENT_FIX_X`, `CURRENT_FIX_Y`
- `CURRENT_FIX_DURATION`
- `CURRENT_FIX_INDEX`
- `RECORDING_SESSION_LABEL` or a reader identifier supplied by the file mapping

For EyeAssist-PE ordered screen-plane analysis, preserve `Slice Number`, timestamps and the scan
identifier. See `docs/DATA_SCHEMA.md` for the release-level schema and QC ledger.
