# Final analysis specification

Version: 2026-08-28

This document freezes the analysis families used in the manuscript and companion code. It is a final analysis specification assembled before submission, not a prospective registration.

## Finite-panel density scoring

The primary family contains six declared contrasts. Four hold pool size fixed and test composition: reader-group matched versus half-mixed; reader-group matched versus opposite; session matched versus half-mixed; and session matched versus opposite. Two compare the larger all-record pool with the matched pool and are interpreted as coverage contrasts because pool size and composition change together. Scores use base-2 held-out fixation log probability, case-level averaging and case-cluster bootstrap intervals. The reader-group matched-versus-opposite contrast additionally uses the exact allocation distribution.

## Behaviour and spatial summaries

Eight reader-group behavioural contrasts form one Benjamini-Hochberg family. Spatial localization, density scoring and attribution analyses retain their declared sampling units and report null results together with resolved contrasts.

## Computational comparisons

Fixed-model scoring changes only the gaze reference. Saliency transfer changes only the gaze training target within a shared partition and training schedule. Classification compares four shared-split arms and reports AUROC, PR-AUC, fixed-threshold operating metrics and sensitivity at 80% specificity. Repeated overlapping splits are not independent inference units; uncertainty is clustered by the 75 unique cases while retaining all arms and split membership.

## GazeVaLM task analysis

Matched, half-mixed and opposite-task pools contain the same four source-reader identities after target-reader exclusion. The primary interpretation is the diagnostic-versus-authenticity task-direction interaction. Kernel dependence, real/synthetic stratification, entropy and effective support are reported with the main contrast; the symmetric average is secondary.

## Multiplicity and reporting

The six density contrasts are reported as a complete family, with equal-size composition contrasts distinguished from coverage contrasts. The eight behavioural contrasts use BH adjustment. Computational comparisons are reported as paired effect estimates with intervals across their declared target families; resolved and unresolved intervals are presented together rather than selected by significance.
