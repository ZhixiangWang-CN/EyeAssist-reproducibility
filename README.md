<div align="center">

# EyeAssist Reproducibility

**Conditional radiology-gaze targets across reader groups and reading states**

Code, configuration and audit trails for the EyeAssist analyses described in the accompanying
Nature Machine Intelligence manuscript.

[Quick start](#quick-start) · [Analysis map](#analysis-map) · [Data access](#data-access)

</div>

---

## What this repository reproduces

EyeAssist treats gaze supervision as a conditional target rather than a fixed image label. This
repository implements the manuscript's auditable analysis backbone:

- normalized fixation-density construction on case-specific canvases;
- reader-specific global session alignment with a case-specific-canvas leave-one-case-out sensitivity analysis;
- held-out matched, half-mixed, opposite and all-record reference pools;
- NSS, Pearson correlation, similarity, KL divergence and predictive log score;
- reader-, case- and split-level resampling with an explicit sampling unit;
- reader-group entropy and 10×10 coverage reconstructed from fixation CSVs;
- four-metric saliency-transfer robustness and target-entropy/effective-support sensitivity;
- all-reader EyeAssist-PE slice-stable session attribution after immediate slice transitions are excluded;
- fixed-pool task analysis for the public GazeVaLM cohort, including its data audit and synthetic test;
- configurable case-grouped repeated partitions;
- ResNet-34 U-Net-style saliency and ResNet-50 auxiliary gaze-supervision model factories;
- leakage-safe ResNet-50 training, checkpoint selection and held-out evaluation entry points;
- deterministic synthetic tests that require no clinical data.

Case-level bootstrap inference retains all appearances of a sampled case and all paired model arms
together.

## Quick start

```bash
git clone https://github.com/ZhixiangWang-CN/EyeAssist-reproducibility.git
cd EyeAssist-reproducibility
conda env create -f environment.yml
conda activate eyeassist-reproducibility
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python examples/synthetic_demo.py --output outputs/synthetic_demo
```

The synthetic demo runs without clinical images or gaze records.

For controlled-access EyeAssist data:

```bash
cp configs/analysis.example.yaml configs/analysis.yaml
# Edit only configs/analysis.yaml and the local data manifest.
eyeassist validate-data --config configs/analysis.yaml
eyeassist make-splits \
  --config configs/analysis.yaml \
  --analysis classifier \
  --test-groups 15 \
  --output outputs/classifier_splits.csv
python scripts/03_density_pool_analysis.py \
  --config configs/analysis.yaml \
  --axis reader_group \
  --output outputs/reader_group_pool_scores.csv
```

## Analysis map

| Manuscript component | Repository implementation | Primary output |
|---|---|---|
| Fixation preprocessing and density maps | `eyeassist.gaze.density_map` | unit-mass case-reader maps |
| Session translation | `reader_offset`, `leave_one_case_out_offsets` | primary and cross-fitted reader offsets on the same case-specific manifest dimensions |
| Alignment sensitivity | `scripts/05_alignment_sensitivity.py` | relocation and density-score point-estimate comparison on the canonical case-specific canvases; primary intervals come from the Fig. 5 case-bootstrap pipeline |
| Saliency partition inference | `scripts/06_saliency_partition_summary.py` | 17 partition-level contrasts |
| Saliency multi-metric and target-concentration sensitivity | `scripts/11_saliency_robustness.py` | four 4×4 metric matrices, paired contrasts and entropy/effective support |
| EyeAssist-PE slice-stable sensitivity | `scripts/12_pe_slice_stable_sensitivity.py` | stream-eligibility audit, paired summaries and case-/reader-held-out attribution |
| Finite-panel scoring | `held_out_configuration_scores` | case-reader pool scores |
| Saliency metrics | `nss`, `pearson_cc`, `similarity`, `kl_divergence` | paired metric table |
| Reader/case inference | `eyeassist.statistics` | point estimates and intervals |
| Repeated internal evaluation | `eyeassist.splits` | configurable group-held-out split manifest |
| Classification results | `scripts/10_case_cluster_auc.py --source <case-level-workbook.xlsx>` | mean within-split AUROC and paired case-cluster intervals across 50 shared runs |
| Saliency transfer | `make_saliency_model`, `saliency_objective` | per-split target matrix |
| Gaze-supervised classification | `make_classifier`, `attention_kl` | per-split AUROC table |
| ResNet-50 training and checkpointing | `scripts/13_train_resnet50_classifier.py` | `last.pt`, leakage-safe `selected.pt` and local training history |
| ResNet-50 held-out evaluation | `scripts/14_evaluate_resnet50_classifier.py` | local case-level probabilities and operating-point metrics |
| GazeVaLM fixed-pool task analysis | `external/gazevalm/run_fixed_pool.py` | locally generated target-, stimulus- and source-study-level contrasts |

The detailed claim-to-code trace is in [docs/METHODS_TO_CODE.md](docs/METHODS_TO_CODE.md).

## ResNet-50 training and testing

Install the optional model dependencies:

```bash
python -m pip install -e '.[models]'
```

The manifest uses one row per case. The three gaze-supervised arms additionally require
`gaze_generalist_path`, `gaze_cold_read_path` and `gaze_informed_path`; values may point to a
non-negative `.npy` density or grayscale density image. Generated checkpoints and predictions are
written below `outputs/` and remain excluded from Git.

Train one arm on one stored split:

```bash
python scripts/13_train_resnet50_classifier.py \
  --manifest data/private/neo/classifier_manifest.csv \
  --splits data/private/neo/classifier_splits.csv \
  --split-id 0 \
  --arm informed_gaze \
  --epochs <locked-epoch-count> \
  --seed <locked-optimization-seed> \
  --pretrained \
  --checkpoint-rule last_epoch \
  --output-dir outputs/classifier
```

`last_epoch` is the default rule for a train/test-only split and saves the fixed final epoch as
`selected.pt`. Alternatively, `best_val_loss` or `best_val_auroc` requires `--validation-cases`;
that validation subset is drawn only from the training cases. The held-out test cases are never
loaded by the training loop and cannot select a checkpoint.

Evaluate the selected checkpoint once:

```bash
python scripts/14_evaluate_resnet50_classifier.py \
  --manifest data/private/neo/classifier_manifest.csv \
  --splits data/private/neo/classifier_splits.csv \
  --split-id 0 \
  --arm informed_gaze \
  --checkpoint outputs/classifier/split_000/informed_gaze/selected.pt \
  --output-csv outputs/classifier/split_000/informed_gaze/test_predictions.csv
```

Every checkpoint records the split and arm, model/optimizer/scheduler states, epoch, selection
rule, run parameters, input-table hashes and the exact train, validation and held-out test case IDs.
An existing run directory is never silently overwritten; continuing it requires `--resume`, which
also verifies the locked input hashes and all optimization/model-selection settings.

## Data access

Clinical images and identifiable or linkable records are not part of this Git repository.
The EyeAssist dataset is available through the gated Hugging Face repository
[fvewa/EyeAssist](https://huggingface.co/datasets/fvewa/EyeAssist). Authorized users place the
downloaded files under `data/private/`, which is ignored by Git, and provide a case-level manifest.

See [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) for required fields and QC provenance.

## Repository layout

```text
.
├── configs/                 # versioned analysis settings
├── data/                    # example manifest; controlled data are git-ignored
├── docs/                    # data schema and methods-to-code mapping
├── examples/                # data-free end-to-end smoke test
├── external/gazevalm/       # executable public-cohort analysis and data-free tests
├── scripts/                 # ordered analysis entry points
├── src/eyeassist/           # reusable implementation
├── tests/                   # deterministic standard-library unit tests
└── outputs/                 # locally generated artifacts; intentionally not committed
```

The repository distributes source code, configurations and tests. It does not distribute
EyeAssist predictions, fitted-model outputs or derived result tables.

## Citation

Please cite the accompanying EyeAssist manuscript when using this analysis code.
