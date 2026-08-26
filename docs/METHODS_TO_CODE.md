# Methods-to-code traceability

This ledger maps the current manuscript's methods to executable code. It separates values stated
in the manuscript or recovered from packaged analysis code from settings that still require the
locked study record.

| Method/claim | Implementation | Locked setting or invariant | Status |
|---|---|---|---|
| Case-specific fixation density | `src/eyeassist/gaze.py::density_map` | fourfold downsampling; configurable Gaussian sigma; unit mass; uniform smoothing mass | implemented |
| Equal member contribution | `equal_reader_pool` | normalize each member map, then average | implemented |
| Held-out target reader/record | `pooling.py::held_out_configuration_scores` | target excluded from every pool | implemented/tested |
| Reader-group matched pool | same | 4 same-state maps after one target reader is held out | implemented |
| Reader-group half-mixed pool | same | 2 target-state + 2 off-state maps | implemented |
| Reader-group opposite pool | same | 4 off-state maps | implemented |
| Session matched/half/opposite | same | 2; 1+1; 2 recording maps | implemented |
| Predictive log score | `gaze.py::fixation_log_score` | mean log2 probability at held-out fixations | implemented |
| Session translation | `reader_offset`, `leave_one_case_out_offsets` | duration-weighted case CoM; cross-fitted by held-out case; both offset estimators and the primary density analysis read the same case-specific dimensions from the manifest; the sensitivity script reports point estimates only, so primary intervals remain governed by the single Fig. 5 case-bootstrap pipeline | implemented; real-data sensitivity reconstructed |
| NSS/CC/SIM/KL | `gaze.py` | constant/empty-map rules explicit | implemented/tested |
| Entropy and 10x10 coverage | `behavior.py`, `scripts/04_reproduce_entropy_coverage.py` | clipped 2000x2800 canvas and case-first aggregation | implemented/reconstructed |
| Saliency-transfer multi-metric robustness | `scripts/11_saliency_robustness.py` | NSS/CC/KL/sAUC; three executions averaged within 17 partitions; paired target-fixed contrasts | implemented/reconstructed |
| Saliency target concentration | `scripts/11_saliency_robustness.py` | Shannon entropy and Shannon/Simpson effective support; equal-reader and duration pooling; 0%/1% floors | implemented/reconstructed |
| PE slice-stable sensitivity | `scripts/12_pe_slice_stable_sensitivity.py` | maximal unchanged-slice runs; 2/3/5-row definitions; fixed-C logistic regression; LOCO/LORO | implemented/reconstructed |
| Reader-level exact sign test | `statistics.py::exact_sign_permutation` | inference unit is reader | implemented/tested |
| Finite-panel case bootstrap | `statistics.py`, `scripts/03_density_pool_analysis.py` | 2,000 case resamples; seed 20260822; case-first contrast summary | implemented/configured |
| Case/split paired bootstrap | `statistics.py`, `scripts/10_case_cluster_auc.py` | paired differences; overlapping case appearances retained within cluster; seed versioned | implemented/tested |
| Classifier repeated partitions | locally supplied case-level predictions | 50 overlapping label-stratified case splits; arms paired within split | analysis implemented/audited; local input required |
| Classifier repeated-split inference | `scripts/10_case_cluster_auc.py` with an author-provided case-level workbook | mean of 50 within-split AUROCs; 20,000 label-stratified resamples of 75 case clusters with all four arms paired | implemented/audited; local input required |
| ResNet-50 auxiliary gaze loss | `models.py` | CE + 0.5 KL(human || CAM); normalized rectified `layer4` CAM for the true class | implemented |
| ResNet-50 training/checkpoint selection | `scripts/13_train_resnet50_classifier.py` | no augmentation in the locked example; test cases excluded; fixed-final-epoch selection; atomic model/optimizer/scheduler checkpoints | implemented |
| ResNet-50 held-out testing | `scripts/14_evaluate_resnet50_classifier.py` | selected checkpoint only; locked split/arm and test-case verification; case-level abnormal probability | implemented |
| U-Net-style saliency model | `models.py` | ResNet-34 encoder; KL + correlation objective; Adam; cosine; patience 15; 60 epochs | implemented |
| GazeVaLM fixed-pool task analysis | `external/gazevalm/run_fixed_pool.py` | target reader excluded; paired source identities; exact four-reader pools; source-study-cluster bootstrap | implemented/tested |

## Provenance rules

1. A numerical default is committed only if it appears in the manuscript or analysis code used for
   the figures.
2. Real-data outputs must be generated from a committed config and split manifest; no number in a
   paper table should be typed manually into an output artifact.
3. Every stochastic job records the analysis seed, package versions and Git
   commit.
