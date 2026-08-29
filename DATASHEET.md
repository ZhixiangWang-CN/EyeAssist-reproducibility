# EyeAssist datasheet

## Resource summary

EyeAssist is a de-identified radiology eye-tracking resource with two analysis subsets. EyeAssist-Neo contains 75 neonatal radiographs from 75 patients, read by a ten-reader first-session panel; three readers also completed a second session after a washout of at least two weeks. EyeAssist-PE contains 40 CTPA volumes from 40 patients, read by seven radiologists in two sessions, yielding 280 matched reader-case pairs.

## Tasks and labels

EyeAssist-Neo supports gaze analysis and normal-versus-abnormal image classification. Status was derived from the clinician-recorded diagnostic field in the original analysis table: records explicitly coded `NORMAL` were classified as normal and all other recorded diagnoses as abnormal. EyeAssist-PE supports gaze and reading-behaviour analysis around 30-day prognosis; the cohort contains 20 survivors and 20 deaths determined by clinical follow-up.

## Acquisition metadata

Neo gaze was sampled at 500 Hz and PE gaze at 1,000 Hz on an LG 12 MP radiology monitor (4,200 x 2,800 pixels). A forehead target supported head-movement compensation without a chin rest. Calibration and validation used a 13-point grid with a target accuracy below 1 degree. Neo was recorded through WebLink/EyeLink EDF output; PE used custom software with gaze coordinates synchronized to the displayed CT slice. In the PE interface, coordinates outside the image region were not recorded.

## Analysis-critical conditioning variables

The gaze target must be indexed by the intended reader population, imaging task, session and information state. The repeated-session contrast is a composite of information availability, session order and repeated exposure. Pool size, member weighting, smoothing, uniform floor, alignment, held-out member construction and inference unit must be declared with every pooled reference.

## Access and governance

The dataset is available through the gated repository at <https://huggingface.co/datasets/fvewa/EyeAssist>; users must accept its access conditions. Clinical images, raw controlled gaze records, case-level predictions and fitted-model outputs are not distributed through this code repository. The EyeAssist study was approved by the Northwestern University Institutional Review Board under protocol STU00215575.

## Intended use

The resource supports research on radiology gaze, human-derived supervision, reference construction, saliency evaluation and gaze-supervised learning. It is not a standalone clinical device or a substitute for clinical diagnosis.

## Companion artifacts

- Machine-readable metadata: [`croissant.json`](croissant.json)
- Final analysis specification: [`FINAL_ANALYSIS_SPECIFICATION.md`](FINAL_ANALYSIS_SPECIFICATION.md)
- Methods-to-code trace: [`docs/METHODS_TO_CODE.md`](docs/METHODS_TO_CODE.md)
