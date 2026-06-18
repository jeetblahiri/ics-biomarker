# ICS Biomarker Experiment Log

Central record for Aarogya ICS biomarker preprocessing and model runs.

## Dataset Build

- Raw source: `/share/data/lakshya/data-aarogya-raw-final`
- Preprocessed output: `/share/data/lakshya/jeet-biomarker-dataset`
- Split manifest: `splits/aarogya_ics_subject_splits_seed0.csv`
- Split policy: subject-level Train/Eval/Test split by `site::subject_id`
- Included IED patients as `Epileptic`
- Preprocessed format: `.npy`, 10-second epochs, 19 EEG channels, 1250 samples per epoch
- Manifest EDFs: 2162
- Successfully preprocessed: 2116
- Failed preprocessing: 46

## Results

| Date | Run | Site Filter | Epoch Cap | Train Subjects | Eval Subjects | Test Subjects | Eval Threshold | Train AUC | Eval AUC | Test AUC | Test Acc | Test Sens | Test Spec | Output |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-18 | aarogya_all_sites_10min | all | 60 | 1117 | 238 | 243 | 0.4705 | 0.695 | 0.648 | 0.733 | 0.663 | 0.682 | 0.616 | `aarogya_ics_results_10min` |
| 2026-06-18 | aarogya_max_only_10min | max | 60 | 595 | 127 | 131 | 0.3950 | 0.740 | 0.697 | 0.635 | 0.603 | 0.691 | 0.460 | `aarogya_ics_results_max_10min` |
| 2026-06-18 | aarogya_ihbas_only_10min | ihbas | 60 | 464 | 99 | 100 | 0.7674 | 0.783 | 0.621 | 0.827 | 0.500 | 0.417 | 0.938 | `aarogya_ics_results_ihbas_10min` |
| 2026-06-18 | aarogya_max_only_all_epochs | max | all | 595 | 127 | 131 | 0.4748 | 0.764 | 0.676 | 0.646 | 0.557 | 0.531 | 0.600 | `aarogya_ics_results_max_all_epochs` |

## Notes

- The all-site run used `--max_epochs_per_record 60`, equivalent to the first 10 minutes per recording.
- A 30-minute cap was attempted but stopped before completion because it was too slow for an interactive first pass.
- The single-site runs are intended to test whether the biomarker signal holds within one acquisition environment instead of depending on site/workflow differences.
- IHBAS-only Eval/Test splits are highly imbalanced, with only 16 mimicker subjects in each split. Interpret threshold-dependent metrics cautiously.
