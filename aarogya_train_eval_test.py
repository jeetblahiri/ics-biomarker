#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train/evaluate/test the ICS biomarker on the prepared Aarogya dataset.

Expected preprocessed layout:
    root/
      Train/{Epileptic,Mimickers}/*.npy
      Eval/{Epileptic,Mimickers}/*.npy
      Test/{Epileptic,Mimickers}/*.npy

The split manifest is used to group recordings by patient-level subject_key,
so multiple EDF recordings from the same patient are aggregated before scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from ics_complete_analysis_optimized import (
    AnalysisResults,
    SubjectRecord,
    bootstrap_auc_ci,
    build_epoch_features_parallel,
    build_subject_features,
    compute_feature_importance,
    compute_metrics,
    get_logistic_regression,
    plot_calibration_curve,
    plot_decision_curve,
    plot_error_analysis,
    plot_feature_importance,
    plot_roc,
    plot_score_distributions,
    analyze_errors,
)


SPLITS = ("Train", "Eval", "Test")


def safe_output_stem(row: dict[str, str]) -> str:
    stem = Path(row["recording_name"]).stem
    subject = row["subject_id"]
    if stem == subject:
        return subject
    return f"{subject}__{stem}"


def load_records_from_manifest(
    manifest_csv: Path,
    data_root: Path,
    split: str,
    missing: list[str],
    sites: set[str] | None,
) -> list[SubjectRecord]:
    records: list[SubjectRecord] = []
    with manifest_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if sites is not None and row["site"] not in sites:
                continue
            path = data_root / row["split"] / row["class_dir"] / f"{safe_output_stem(row)}.npy"
            if not path.exists():
                missing.append(row["relative_path"])
                continue
            subject_id = row["subject_key"]
            label = int(row["label"])
            records.append(SubjectRecord(subject_id, path, label, split, row["site"]))
    return records


def summarize_records(records_by_split: dict[str, list[SubjectRecord]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split, records in records_by_split.items():
        subjects = {}
        for record in records:
            subjects[record.subject_id] = record.label
        counts = Counter(subjects.values())
        summary[split] = {
            "recordings": len(records),
            "subjects": len(subjects),
            "epileptic_subjects": int(counts.get(1, 0)),
            "mimicker_subjects": int(counts.get(0, 0)),
        }
    return summary


def compute_split_features(
    records: list[SubjectRecord],
    seg_sec: float,
    target_fs: float | None,
    n_jobs: int,
    cache_dir: Path | None,
    split: str,
    max_epochs_per_record: int | None,
):
    print(f"\nComputing {split} epoch features from {len(records)} recordings...")
    X, y, feature_names, subj_ids = build_epoch_features_parallel(
        records,
        seg_sec,
        target_fs=target_fs,
        n_jobs=n_jobs,
        cache_dir=cache_dir,
        show_progress=True,
        max_epochs_per_record=max_epochs_per_record,
    )
    print(f"{split}: {X.shape[0]} epochs, {X.shape[1]} features")
    return X, y, feature_names, subj_ids


def score_subjects(
    X_epoch: np.ndarray,
    y_epoch: np.ndarray,
    subj_ids_epoch: list[str],
    feature_names: list[str],
    epoch_scaler: StandardScaler,
    epoch_clf,
    subject_scaler: StandardScaler,
    subject_clf,
):
    X_epoch_scaled = epoch_scaler.transform(X_epoch)
    p_epoch = epoch_clf.predict_proba(X_epoch_scaled)[:, 1]
    X_subj, y_subj, subj_feature_names, subj_ids = build_subject_features(
        X_epoch,
        y_epoch,
        subj_ids_epoch,
        p_epoch,
        feature_names,
    )
    p_subj = subject_clf.predict_proba(subject_scaler.transform(X_subj))[:, 1]
    return X_subj, y_subj, p_subj, subj_ids, subj_feature_names


def choose_eval_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5
    youden = tpr[finite] - fpr[finite]
    return float(thresholds[finite][int(np.argmax(youden))])


def result_dict(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    subject_ids: list[str],
    n_boot: int,
    rng: np.random.RandomState,
    threshold: float,
) -> dict[str, object]:
    metrics = compute_metrics(y_true, y_score, threshold=threshold)
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_score, n_boot=n_boot, rng=rng)
    return {
        "split": name,
        "threshold": threshold,
        "auc": float(metrics["auc"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "accuracy": float(metrics["accuracy"]),
        "sensitivity": float(metrics["sensitivity"]),
        "specificity": float(metrics["specificity"]),
        "brier_score": float(metrics["brier_score"]),
        "n_subjects": int(len(subject_ids)),
        "n_epileptic": int(np.sum(y_true == 1)),
        "n_mimickers": int(np.sum(y_true == 0)),
    }


def write_predictions(path: Path, subject_ids: Iterable[str], y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "label", "prob_epileptic", "prediction"])
        writer.writeheader()
        for sid, label, score in zip(subject_ids, y_true, y_score):
            writer.writerow({
                "subject_id": sid,
                "label": int(label),
                "prob_epileptic": float(score),
                "prediction": int(score >= threshold),
            })


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/eval/test ICS biomarker on prepared Aarogya data")
    ap.add_argument("--data_root", type=Path, default=Path("/share/data/lakshya/jeet-biomarker-dataset"))
    ap.add_argument("--manifest", type=Path, default=Path("splits/aarogya_ics_subject_splits_seed0.csv"))
    ap.add_argument("--output_dir", type=Path, default=Path("aarogya_ics_results"))
    ap.add_argument("--cache_dir", type=Path, default=Path("aarogya_feature_cache"))
    ap.add_argument("--seg_sec", type=float, default=10.0)
    ap.add_argument("--target_fs", type=float, default=None)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--site",
        action="append",
        default=None,
        help="Restrict to one site. Can be passed more than once, e.g. --site max --site ihbas.",
    )
    ap.add_argument(
        "--max_epochs_per_record",
        type=int,
        default=None,
        help="Optionally use only the first N epochs from each preprocessed recording.",
    )
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if args.no_cache else args.cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    sites = set(args.site) if args.site else None
    if sites:
        print(f"Site filter: {', '.join(sorted(sites))}")

    missing: list[str] = []
    records_by_split = {
        split: load_records_from_manifest(args.manifest, args.data_root, split, missing, sites)
        for split in SPLITS
    }
    summary = summarize_records(records_by_split)
    print("Dataset summary:")
    for split in SPLITS:
        print(f"  {split}: {summary[split]}")
    print(f"Missing preprocessed recordings listed in manifest: {len(missing)}")

    features = {
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

    X_train, y_train, feature_names, sids_train_epoch = features["Train"]
    print("\nTraining epoch-level model on Train...")
    epoch_scaler = StandardScaler()
    X_train_scaled = epoch_scaler.fit_transform(X_train)
    epoch_clf = get_logistic_regression(class_weight="balanced", random_state=args.seed)
    epoch_clf.fit(X_train_scaled, y_train)
    p_train_epoch = epoch_clf.predict_proba(X_train_scaled)[:, 1]

    X_train_subj, y_train_subj, subj_feature_names, sids_train = build_subject_features(
        X_train,
        y_train,
        sids_train_epoch,
        p_train_epoch,
        feature_names,
    )
    print(f"Training subject-level model on {len(sids_train)} Train subjects...")
    subject_scaler = StandardScaler()
    X_train_subj_scaled = subject_scaler.fit_transform(X_train_subj)
    subject_clf = get_logistic_regression(class_weight="balanced", random_state=args.seed)
    subject_clf.fit(X_train_subj_scaled, y_train_subj)

    split_scores = {}
    split_subject_features = {}
    for split in SPLITS:
        X_epoch, y_epoch, _, subj_ids_epoch = features[split]
        X_subj, y_subj, p_subj, subj_ids, _ = score_subjects(
            X_epoch,
            y_epoch,
            subj_ids_epoch,
            feature_names,
            epoch_scaler,
            epoch_clf,
            subject_scaler,
            subject_clf,
        )
        split_scores[split] = (y_subj, p_subj, subj_ids)
        split_subject_features[split] = X_subj

    eval_y, eval_p, _ = split_scores["Eval"]
    eval_threshold = choose_eval_threshold(eval_y, eval_p)
    print(f"\nEval-selected threshold: {eval_threshold:.4f}")

    metrics = {}
    for split in SPLITS:
        y_subj, p_subj, subj_ids = split_scores[split]
        threshold = 0.5 if split == "Train" else eval_threshold
        metrics[split] = result_dict(split, y_subj, p_subj, subj_ids, args.n_boot, rng, threshold)
        write_predictions(args.output_dir / f"{split.lower()}_subject_predictions.csv", subj_ids, y_subj, p_subj, threshold)
        print(
            f"{split}: AUC={metrics[split]['auc']:.3f} "
            f"Acc={metrics[split]['accuracy']:.3f} "
            f"Sens={metrics[split]['sensitivity']:.3f} "
            f"Spec={metrics[split]['specificity']:.3f}"
        )

    importance = compute_feature_importance(subject_clf, subject_scaler, subj_feature_names)
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "data_root": str(args.data_root),
                "manifest": str(args.manifest),
                "site_filter": sorted(sites) if sites else None,
                "max_epochs_per_record": args.max_epochs_per_record,
                "summary": summary,
                "missing_preprocessed_recordings": missing,
                "eval_selected_threshold": eval_threshold,
                "metrics": metrics,
                "feature_importance": importance,
            },
            f,
            indent=2,
        )

    if not args.no_plots:
        for split in ("Eval", "Test"):
            y_subj, p_subj, subj_ids = split_scores[split]
            threshold = eval_threshold
            plot_roc(y_subj, p_subj, f"Aarogya {split} ROC", args.output_dir / f"{split.lower()}_roc.png")
            plot_score_distributions(y_subj, p_subj, f"Aarogya {split} Scores", args.output_dir / f"{split.lower()}_scores.png")
            plot_calibration_curve(y_subj, p_subj, f"Aarogya {split} Calibration", args.output_dir / f"{split.lower()}_calibration.png")
            plot_decision_curve(y_subj, p_subj, f"Aarogya {split} Decision Curve", args.output_dir / f"{split.lower()}_decision_curve.png")
            y_pred = (p_subj >= threshold).astype(int)
            plot_error_analysis(
                analyze_errors(y_subj, y_pred, p_subj, subj_ids),
                f"Aarogya {split} Error Analysis",
                args.output_dir / f"{split.lower()}_error_analysis.png",
            )
        plot_feature_importance(importance, "Aarogya Train Feature Importance", args.output_dir / "train_feature_importance.png")

    print(f"\nSaved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
