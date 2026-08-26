# GazeVaLM fixed-pool analysis

This directory contains the executable second-cohort analysis used in the manuscript. It reads
the public [GazeVaLM](https://huggingface.co/datasets/davidcwong/GazeVaLM) scanpaths and generates
all audit and statistical output locally. No derived results are committed to this repository.

For each image, target task and target reader, the target reader is excluded from the reference
pool. The same four remaining reader identities define matched-task, balanced 2+2 mixed-task and
opposite-task references. The primary outcome is the held-out fixation log score relative to a
uniform image density, expressed in bits per fixation.

The primary analysis uses the 16 participant columns shared by the two public expert-result tables.
Target reader--image records must be available in both tasks. Confidence intervals resample the 30
source-study identifiers, retaining each real/synthetic image pair in the same cluster.

Run the data-free test:

```bash
python external/gazevalm/test_synthetic.py
```

Run the primary analysis after downloading and unpacking GazeVaLM:

```bash
python external/gazevalm/run_fixed_pool.py \
  --data-root /path/to/GazeVaLM \
  --output-dir outputs/gazevalm_sigma40 \
  --cache-dir outputs/gazevalm_cache \
  --config external/gazevalm/config_primary.json
```

Repeat with `config_sigma30.json` and `config_sigma60.json` for the kernel-width sensitivity
analysis. Generated files remain under `outputs/`, which is intentionally excluded from Git.

