#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cross-center transfer for Aarogya ICS biomarker."""

from __future__ import annotations

import argparse
import json
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
    plot_calibration_curve,
    plot_decision_curve,
    plot_error_analysis,
    plot_feature_importance,
    plot_roc,
    plot_score_distributions,
)


def load_site_features(args, site: str, cache_dir: Path):
    missing: list[str] = []
    records_by_split = {
        split: load_records_from_manifest(args.manifest, args.data_root, split, missing, {site})
        for split in SPLITS
    }
    split_features = {
        split: compute_split_features(
            records_by_split[split],
            args.seg_sec,
            args.target_fs,
            args.n_jobs,
            cache_dir,
            split,
            args.max_epochs_per_record,
        )
        for split in SPLITS
    }
    X = np.vstack([split_features[split][0] for split in SPLITS])
    y = np.concatenate([split_features[split][1] for split in SPLITS])
    subj_ids = []
    for split in SPLITS:
        subj_ids.extend(split_features[split][3])
    return X, y, split_features["Train"][2], subj_ids, missing


def transfer_once(
    train_site: str,
    test_site: str,
    train_data,
    test_data,
    output_dir: Path,
    n_boot: int,
    rng: np.random.RandomState,
    include_log_n_epochs: bool,
    threshold: float,
    no_plots: bool,
):
    X_train, y_train, feature_names, sids_train_epoch, _ = train_data
    X_test, y_test, _, sids_test_epoch, _ = test_data

    scaler_e = StandardScaler()
    X_train_scaled = scaler_e.fit_transform(X_train)
    X_test_scaled = scaler_e.transform(X_test)

    clf_epoch = get_logistic_regression(class_weight="balanced", random_state=0)
    clf_epoch.fit(X_train_scaled, y_train)
    p_train_epoch = clf_epoch.predict_proba(X_train_scaled)[:, 1]
    p_test_epoch = clf_epoch.predict_proba(X_test_scaled)[:, 1]

    X_train_subj, y_train_subj, subj_feature_names, sids_train = build_subject_features(
        X_train,
        y_train,
        sids_train_epoch,
        p_train_epoch,
        feature_names,
        include_log_n_epochs=include_log_n_epochs,
    )
    X_test_subj, y_test_subj, _, sids_test = build_subject_features(
        X_test,
        y_test,
        sids_test_epoch,
        p_test_epoch,
        feature_names,
        include_log_n_epochs=include_log_n_epochs,
    )

    scaler_s = StandardScaler()
    X_train_subj_scaled = scaler_s.fit_transform(X_train_subj)
    X_test_subj_scaled = scaler_s.transform(X_test_subj)

    clf_subject = get_logistic_regression(class_weight="balanced", random_state=0)
    clf_subject.fit(X_train_subj_scaled, y_train_subj)
    p_test_subject = clf_subject.predict_proba(X_test_subj_scaled)[:, 1]

    run_name = f"train_{train_site}_test_{test_site}"
    metrics = result_dict(run_name, y_test_subj, p_test_subject, sids_test, n_boot, rng, threshold)
    importance = compute_feature_importance(clf_subject, scaler_s, subj_feature_names)

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(run_dir / "subject_predictions.csv", sids_test, y_test_subj, p_test_subject, threshold)

    if not no_plots:
        title = f"Train {train_site} -> Test {test_site}"
        plot_roc(y_test_subj, p_test_subject, f"{title} ROC", run_dir / "roc.png")
        plot_score_distributions(y_test_subj, p_test_subject, f"{title} Scores", run_dir / "scores.png")
        plot_calibration_curve(y_test_subj, p_test_subject, f"{title} Calibration", run_dir / "calibration.png")
        plot_decision_curve(y_test_subj, p_test_subject, f"{title} Decision Curve", run_dir / "decision_curve.png")
        plot_error_analysis(
            analyze_errors(y_test_subj, (p_test_subject >= threshold).astype(int), p_test_subject, sids_test),
            f"{title} Error Analysis",
            run_dir / "error_analysis.png",
        )
        plot_feature_importance(importance, f"{title} Feature Importance", run_dir / "feature_importance.png")

    print(
        f"{run_name}: AUC={metrics['auc']:.3f} "
        f"Acc={metrics['accuracy']:.3f} "
        f"Sens={metrics['sensitivity']:.3f} "
        f"Spec={metrics['specificity']:.3f}"
    )

    return {
        "train_site": train_site,
        "test_site": test_site,
        "train_subjects": int(len(set(sids_train))),
        "test_subjects": int(len(sids_test)),
        "train_epochs": int(X_train.shape[0]),
        "test_epochs": int(X_test.shape[0]),
        "metrics": metrics,
        "feature_importance": importance,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aarogya cross-center transfer")
    ap.add_argument("--data_root", type=Path, default=Path("/share/data/lakshya/jeet-biomarker-dataset"))
    ap.add_argument("--manifest", type=Path, default=Path("splits/aarogya_ics_subject_splits_seed0.csv"))
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_cache_dir", type=Path, required=True)
    ap.add_argument("--ihbas_cache_dir", type=Path, required=True)
    ap.add_argument("--seg_sec", type=float, default=10.0)
    ap.add_argument("--target_fs", type=float, default=None)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs_per_record", type=int, default=150)
    ap.add_argument("--exclude_recording_length", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.max_cache_dir.mkdir(parents=True, exist_ok=True)
    args.ihbas_cache_dir.mkdir(parents=True, exist_ok=True)

    include_log_n_epochs = not args.exclude_recording_length
    print(f"Max epochs per recording: {args.max_epochs_per_record}")
    print(f"Include log_n_epochs: {include_log_n_epochs}")

    max_data = load_site_features(args, "max", args.max_cache_dir)
    ihbas_data = load_site_features(args, "ihbas", args.ihbas_cache_dir)

    threshold = 0.5
    results = {
        "train_max_test_ihbas": transfer_once(
            "max",
            "ihbas",
            max_data,
            ihbas_data,
            args.output_dir,
            args.n_boot,
            rng,
            include_log_n_epochs,
            threshold,
            args.no_plots,
        ),
        "train_ihbas_test_max": transfer_once(
            "ihbas",
            "max",
            ihbas_data,
            max_data,
            args.output_dir,
            args.n_boot,
            rng,
            include_log_n_epochs,
            threshold,
            args.no_plots,
        ),
    }

    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "max_epochs_per_record": args.max_epochs_per_record,
                "include_log_n_epochs": include_log_n_epochs,
                "threshold": threshold,
                "max_missing": max_data[4],
                "ihbas_missing": ihbas_data[4],
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
