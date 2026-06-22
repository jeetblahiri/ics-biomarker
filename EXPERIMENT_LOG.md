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
| 2026-06-18 | aarogya_max_10fold_cv_all_epochs | max | all | 853 CV pool | n/a | n/a | 0.5000 | n/a | n/a | 0.691 CV | 0.632 | 0.605 | 0.677 | `aarogya_ics_results_max_10fold_cv_all_epochs` |
| 2026-06-18 | aarogya_ihbas_10fold_cv_all_epochs | ihbas | all | 663 CV pool | n/a | n/a | 0.5000 | n/a | n/a | 0.698 CV | 0.632 | 0.636 | 0.611 | `aarogya_ics_results_ihbas_10fold_cv_all_epochs` |
| 2026-06-18 | aarogya_max_10fold_cv_all_epochs_no_len | max | all | 853 CV pool | n/a | n/a | 0.5000 | n/a | n/a | 0.693 CV | 0.619 | 0.577 | 0.690 | `aarogya_ics_results_max_10fold_cv_all_epochs_no_len` |
| 2026-06-18 | aarogya_ihbas_10fold_cv_all_epochs_no_len | ihbas | all | 663 CV pool | n/a | n/a | 0.5000 | n/a | n/a | 0.686 CV | 0.618 | 0.623 | 0.593 | `aarogya_ics_results_ihbas_10fold_cv_all_epochs_no_len` |
| 2026-06-19 | aarogya_max_10fold_cv_cap150_no_len | max | 150 | 853 CV pool | n/a | n/a | 0.5000 | n/a | n/a | 0.671 CV | 0.605 | 0.568 | 0.668 | `aarogya_ics_results_max_10fold_cv_cap150_no_len` |
| 2026-06-19 | aarogya_ihbas_10fold_cv_cap150_no_len | ihbas | 150 | 663 CV pool | n/a | n/a | 0.5000 | n/a | n/a | 0.694 CV | 0.630 | 0.632 | 0.620 | `aarogya_ics_results_ihbas_10fold_cv_cap150_no_len` |
| 2026-06-19 | aarogya_cross_center_cap150_no_len | train max -> test ihbas | 150 | 853 | n/a | 663 | 0.5000 | n/a | n/a | 0.696 | 0.564 | 0.532 | 0.731 | `aarogya_ics_results_cross_center_cap150_no_len/train_max_test_ihbas` |
| 2026-06-19 | aarogya_cross_center_cap150_no_len | train ihbas -> test max | 150 | 663 | n/a | 853 | 0.5000 | n/a | n/a | 0.624 | 0.607 | 0.652 | 0.532 | `aarogya_ics_results_cross_center_cap150_no_len/train_ihbas_test_max` |
| 2026-06-19 | mishra_finaldata_plus_lakshya_healthy_max_dedup_10fold_cv_no_len | data_max | all | 692 CV pool | n/a | n/a | 0.5000 | n/a | n/a | 0.642 CV | 0.591 | 0.567 | 0.628 | `mishra_ics_results_finaldata_plus_lakshya_healthy_max_dedup_10fold_no_len` |

## Notes

- The all-site run used `--max_epochs_per_record 60`, equivalent to the first 10 minutes per recording.
- A 30-minute cap was attempted but stopped before completion because it was too slow for an interactive first pass.
- The single-site runs are intended to test whether the biomarker signal holds within one acquisition environment instead of depending on site/workflow differences.
- IHBAS-only Eval/Test splits are highly imbalanced, with only 16 mimicker subjects in each split. Interpret threshold-dependent metrics cautiously.
- The MAX 10-fold CV all-epochs run pools MAX Train/Eval/Test subjects, then evaluates by subject-level stratified folds. It uses threshold 0.5 because there is no separate Eval split inside CV.
- The `_no_len` CV runs remove `log_n_epochs`, the explicit subject-level recording-length feature. Probability distribution features remain because they summarize epoch-level model outputs, not recording duration directly.
- No-length CV AUC CIs: MAX `0.658-0.725`, IHBAS `0.634-0.733`.
- IHBAS all-epochs per-record distribution before cap selection: mean `148.4`, median `150.5`, so cap-150 runs use at most the first 150 epochs per recording.
- Cap-150 no-length CV AUC CIs: MAX `0.635-0.707`, IHBAS `0.643-0.740`.
- Cap-150 no-length cross-center AUC CIs: train MAX -> test IHBAS `0.642-0.746`; train IHBAS -> test MAX `0.587-0.664`.
- Mishra prepared-data run uses `/share/tmp/mishra/codebase/epilepsy-inference/prepared_data/finaldata_plus_lakshya_healthy_max_dedup/manifest.jsonl`, labels `Epileptic=1`, `Mimickers/Healthy=0`, and patient-aware folds from `/share/tmp/mishra/codebase/epilepsy-inference/epi_data/kfold.py`.
- Mishra prepared-data no-length 10-fold CV: `177,520` segments, `692` records, `635` patient groups; AUC CI `0.598-0.684`.
