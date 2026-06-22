#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run subject-level K-fold CV for one Aarogya site using the ICS two-stage model.

This script reads the same split manifest as aarogya_train_eval_test.py, but uses
all available splits from the requested site as one within-site CV pool. Feature
caching is compatible with aarogya_train_eval_test.py because each split is
loaded separately before concatenation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from aarogya_train_eval_test import (
    SPLITS,
    compute_split_features,
    load_records_from_manifest,
    result_dict,
    write_predictions,
)
from ics_complete_analysis_optimized import (
    analyze_errors,
    build_subject_features,
    compute_feature_importance,
    get_logistic_regression,
    make_stratified_subject_folds,
    plot_calibration_curve,
    plot_decision_curve,
    plot_error_analysis,
    plot_feature_importance,
    plot_roc,
    plot_score_distributions,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Within-site K-fold CV for Aarogya ICS biomarker")
    ap.add_argument("--site", required=True)
    ap.add_argument("--data_root", type=Path, default=Path("/share/data/lakshya/jeet-biomarker-dataset"))
    ap.add_argument("--manifest", type=Path, default=Path("splits/aarogya_ics_subject_splits_seed0.csv"))
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--seg_sec", type=float, default=10.0)
    ap.add_argument("--target_fs", type=float, default=None)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--k_folds", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs_per_record", type=int, default=None)
    ap.add_argument(
        "--exclude_recording_length",
        action="store_true",
        help="Exclude log_n_epochs from subject-level features.",
    )
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    records_by_split = {
        split: load_records_from_manifest(args.manifest, args.data_root, split, missing, {args.site})
        for split in SPLITS
    }
    print(f"Site filter: {args.site}")
    print(f"Missing preprocessed recordings listed in manifest: {len(missing)}")

    split_features = {
        split: compute_split_features(
            records_by_split[split],
            args.seg_sec,
            args.target_fs,
            args.n_jobs,
            args.cache_dir,
            split,
            args.max_epochs_per_record,
        )
        for split in SPLITS
    }

    feature_names = split_features["Train"][2]
    X_epoch = np.vstack([split_features[split][0] for split in SPLITS])
    y_epoch = np.concatenate([split_features[split][1] for split in SPLITS])
    subj_ids_epoch = []
    for split in SPLITS:
        subj_ids_epoch.extend(split_features[split][3])

    subj_to_label = {}
    for sid, label in zip(subj_ids_epoch, y_epoch):
        subj_to_label.setdefault(sid, int(label))
    subject_ids = sorted(subj_to_label)
    y_subject = np.array([subj_to_label[sid] for sid in subject_ids], dtype=int)

    print(
        f"CV pool: {len(subject_ids)} subjects "
        f"(Epileptic={int(y_subject.sum())}, Mimickers={int((y_subject == 0).sum())}), "
        f"{X_epoch.shape[0]} epochs"
    )
    if args.exclude_recording_length:
        print("Recording-length features: excluded (log_n_epochs removed)")

    folds = make_stratified_subject_folds(subject_ids, y_subject, args.k_folds, rng)
    cv_preds = np.zeros(len(subject_ids), dtype=np.float32)
    all_importance = defaultdict(list)
    subj_index = {sid: idx for idx, sid in enumerate(subject_ids)}

    for fold_idx, val_subjects in enumerate(folds):
        val_subjects = set(val_subjects)
        train_subjects = set(subject_ids) - val_subjects

        train_mask = np.array([sid in train_subjects for sid in subj_ids_epoch])
        val_mask = np.array([sid in val_subjects for sid in subj_ids_epoch])

        scaler_e = StandardScaler()
        X_train_e = scaler_e.fit_transform(X_epoch[train_mask])
        X_val_e = scaler_e.transform(X_epoch[val_mask])

        clf_epoch = get_logistic_regression(class_weight="balanced", random_state=args.seed + fold_idx)
        clf_epoch.fit(X_train_e, y_epoch[train_mask])
        p_train = clf_epoch.predict_proba(X_train_e)[:, 1]
        p_val = clf_epoch.predict_proba(X_val_e)[:, 1]

        X_subj_train, y_subj_train, subj_feature_names, _ = build_subject_features(
            X_epoch[train_mask],
            y_epoch[train_mask],
            [sid for sid, keep in zip(subj_ids_epoch, train_mask) if keep],
            p_train,
            feature_names,
            include_log_n_epochs=not args.exclude_recording_length,
        )
        X_subj_val, _, _, sids_val = build_subject_features(
            X_epoch[val_mask],
            y_epoch[val_mask],
            [sid for sid, keep in zip(subj_ids_epoch, val_mask) if keep],
            p_val,
            feature_names,
            include_log_n_epochs=not args.exclude_recording_length,
        )

        scaler_s = StandardScaler()
        X_subj_train_scaled = scaler_s.fit_transform(X_subj_train)
        X_subj_val_scaled = scaler_s.transform(X_subj_val)

        clf_subject = get_logistic_regression(class_weight="balanced", random_state=args.seed + fold_idx)
        clf_subject.fit(X_subj_train_scaled, y_subj_train)
        p_subj_val = clf_subject.predict_proba(X_subj_val_scaled)[:, 1]

        for sid, prob in zip(sids_val, p_subj_val):
            cv_preds[subj_index[sid]] = prob
        for name, value in compute_feature_importance(clf_subject, scaler_s, subj_feature_names).items():
            all_importance[name].append(value)

        print(f"Fold {fold_idx + 1}/{args.k_folds}: {len(val_subjects)} validation subjects")

    threshold = 0.5
    metrics = result_dict(
        f"{args.site}_{args.k_folds}fold_cv",
        y_subject,
        cv_preds,
        subject_ids,
        args.n_boot,
        rng,
        threshold,
    )
    importance = {name: float(np.mean(values)) for name, values in all_importance.items()}

    write_predictions(args.output_dir / "cv_subject_predictions.csv", subject_ids, y_subject, cv_preds, threshold)
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "site": args.site,
                "k_folds": args.k_folds,
                "threshold": threshold,
                "max_epochs_per_record": args.max_epochs_per_record,
                "include_log_n_epochs": not args.exclude_recording_length,
                "missing_preprocessed_recordings": missing,
                "metrics": metrics,
                "feature_importance": importance,
            },
            f,
            indent=2,
        )

    if not args.no_plots:
        plot_roc(y_subject, cv_preds, f"Aarogya {args.site} {args.k_folds}-fold CV ROC", args.output_dir / "cv_roc.png")
        plot_score_distributions(y_subject, cv_preds, f"Aarogya {args.site} CV Scores", args.output_dir / "cv_scores.png")
        plot_calibration_curve(y_subject, cv_preds, f"Aarogya {args.site} CV Calibration", args.output_dir / "cv_calibration.png")
        plot_decision_curve(y_subject, cv_preds, f"Aarogya {args.site} CV Decision Curve", args.output_dir / "cv_decision_curve.png")
        plot_error_analysis(
            analyze_errors(y_subject, (cv_preds >= threshold).astype(int), cv_preds, subject_ids),
            f"Aarogya {args.site} CV Error Analysis",
            args.output_dir / "cv_error_analysis.png",
        )
        plot_feature_importance(importance, f"Aarogya {args.site} CV Feature Importance", args.output_dir / "cv_feature_importance.png")

    print(
        f"\n{args.site} {args.k_folds}-fold CV: "
        f"AUC={metrics['auc']:.3f} "
        f"Acc={metrics['accuracy']:.3f} "
        f"Sens={metrics['sensitivity']:.3f} "
        f"Spec={metrics['specificity']:.3f}"
    )
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
