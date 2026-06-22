#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run ICS 10-fold CV on Mishra prepared manifest datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from aarogya_train_eval_test import result_dict, write_predictions
from ics_complete_analysis_optimized import (
    analyze_errors,
    build_subject_features,
    compute_epoch_ics,
    compute_feature_importance,
    get_logistic_regression,
    plot_calibration_curve,
    plot_decision_curve,
    plot_error_analysis,
    plot_feature_importance,
    plot_roc,
    plot_score_distributions,
    resample_epoch,
)


@dataclass(frozen=True)
class SegmentRow:
    path: Path
    label: int
    record_id: str
    patient_id: str
    label_name: str
    segment_index: int


def parse_manifest_record(obj: dict, label_map: dict[str, int], rid_to_patient: dict[str, str]) -> SegmentRow | None:
    parts = obj["record"].split("/")
    if len(parts) < 4:
        return None
    source = parts[1]
    label_name = parts[2]
    rid = parts[3]
    if label_name not in label_map:
        return None
    record_id = f"{source}:{rid}"
    return SegmentRow(
        path=Path(obj["preproc_local"]),
        label=int(label_map[label_name]),
        record_id=record_id,
        patient_id=rid_to_patient.get(record_id, record_id),
        label_name=label_name,
        segment_index=int(obj.get("segment_index", 0)),
    )


def read_jsonl_manifest(path: Path, label_map: dict[str, int], rid_to_patient: dict[str, str]) -> list[SegmentRow]:
    rows: list[SegmentRow] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            objs = obj if isinstance(obj, list) else [obj]
            for item in objs:
                row = parse_manifest_record(item, label_map, rid_to_patient)
                if row is not None and row.path.exists():
                    rows.append(row)
    return rows


def read_rid_to_patient(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["rid"]: row["patient_id"] for row in csv.DictReader(f)}


def make_record_table(rows: list[SegmentRow]) -> tuple[list[str], np.ndarray, dict[str, str]]:
    label_by_rid: dict[str, int] = {}
    patient_by_rid: dict[str, str] = {}
    conflicts = []
    for row in rows:
        if row.record_id in label_by_rid and label_by_rid[row.record_id] != row.label:
            conflicts.append(row.record_id)
        label_by_rid.setdefault(row.record_id, row.label)
        patient_by_rid.setdefault(row.record_id, row.patient_id)
    if conflicts:
        raise RuntimeError(f"Cross-label record conflicts: {sorted(set(conflicts))[:10]}")
    record_ids = sorted(label_by_rid)
    labels = np.array([label_by_rid[rid] for rid in record_ids], dtype=np.int64)
    return record_ids, labels, patient_by_rid


def _compute_segment_features(args):
    row_tuple, seg_sec, target_fs, source_fs = args
    path_str, label, record_id = row_tuple
    arr = np.load(path_str)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected segment file with one epoch or 2D array: {path_str} shape={arr.shape}")
        epoch = arr[0]
    elif arr.ndim == 2:
        epoch = arr
    else:
        raise ValueError(f"Unsupported segment shape {arr.shape}: {path_str}")

    epoch = np.asarray(epoch, dtype=np.float32)
    if target_fs is not None and not math.isclose(source_fs, target_fs, rel_tol=1e-6):
        epoch = resample_epoch(epoch, source_fs, target_fs, seg_sec)
    return compute_epoch_ics(epoch, seg_sec), int(label), record_id


def load_or_compute_epoch_features(
    rows: list[SegmentRow],
    cache_path: Path,
    seg_sec: float,
    target_fs: float | None,
    source_fs: float,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["feature_names"].tolist(), data["record_ids"].tolist()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_feats = []
    labels = []
    record_ids = []
    feature_names = None

    args_iter = [
        ((str(row.path), row.label, row.record_id), seg_sec, target_fs, source_fs)
        for row in rows
    ]

    if n_jobs <= 1:
        iterator = map(_compute_segment_features, args_iter)
        for idx, (feats, label, record_id) in enumerate(iterator, start=1):
            if feature_names is None:
                feature_names = sorted(feats)
            all_feats.append([float(feats.get(name, 0.0)) for name in feature_names])
            labels.append(label)
            record_ids.append(record_id)
            if idx % 10000 == 0:
                print(f"  computed {idx}/{len(rows)} segment features", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            iterator = executor.map(_compute_segment_features, args_iter, chunksize=100)
            for idx, (feats, label, record_id) in enumerate(iterator, start=1):
                if feature_names is None:
                    feature_names = sorted(feats)
                all_feats.append([float(feats.get(name, 0.0)) for name in feature_names])
                labels.append(label)
                record_ids.append(record_id)
                if idx % 10000 == 0:
                    print(f"  computed {idx}/{len(rows)} segment features", flush=True)

    X = np.asarray(all_feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    feature_names = feature_names or []
    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        feature_names=np.asarray(feature_names, dtype=object),
        record_ids=np.asarray(record_ids, dtype=object),
    )
    return X, y, feature_names, record_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="ICS k-fold CV for Mishra prepared manifest")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--rid_to_patient", type=Path, default=None)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--feature_cache", type=Path, required=True)
    ap.add_argument("--k_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--seg_sec", type=float, default=10.0)
    ap.add_argument("--source_fs", type=float, default=200.0)
    ap.add_argument("--target_fs", type=float, default=None)
    ap.add_argument("--exclude_recording_length", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--epi_data_root", type=Path, default=Path("/share/tmp/mishra/codebase/epilepsy-inference"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.epi_data_root))
    from epi_data.kfold import make_fold_partition

    label_map = {"Mimickers": 0, "Healthy": 0, "Epileptic": 1}
    rid_to_patient = read_rid_to_patient(args.rid_to_patient)
    rows = read_jsonl_manifest(args.manifest, label_map, rid_to_patient)
    if not rows:
        raise RuntimeError(f"No usable manifest rows found in {args.manifest}")

    record_ids, record_labels, patient_by_rid = make_record_table(rows)
    folds = make_fold_partition(
        record_ids,
        record_labels,
        n_folds=args.k_folds,
        seed=args.seed,
        rid_to_group=patient_by_rid,
        stratify=True,
    )

    print(f"Rows: {len(rows)} segments")
    print(f"Records: {len(record_ids)}")
    print(f"Patients: {len(set(patient_by_rid.values()))}")
    print(f"Labels: epileptic={int(record_labels.sum())}, non_epileptic={int((record_labels == 0).sum())}")

    X_epoch, y_epoch, feature_names, rid_epoch = load_or_compute_epoch_features(
        rows,
        args.feature_cache,
        args.seg_sec,
        args.target_fs,
        args.source_fs,
        args.n_jobs,
    )
    print(f"Epoch features: {X_epoch.shape[0]} epochs, {X_epoch.shape[1]} features")

    subject_ids = sorted(set(rid_epoch))
    subj_to_label = {}
    for rid, label in zip(rid_epoch, y_epoch):
        subj_to_label.setdefault(rid, int(label))
    y_subject = np.asarray([subj_to_label[rid] for rid in subject_ids], dtype=np.int64)
    subj_index = {rid: idx for idx, rid in enumerate(subject_ids)}

    cv_preds = np.zeros(len(subject_ids), dtype=np.float32)
    all_importance = defaultdict(list)

    for fold_idx, test_rids in enumerate(folds):
        test_rids = set(test_rids)
        train_rids = set(record_ids) - test_rids
        train_mask = np.asarray([rid in train_rids for rid in rid_epoch], dtype=bool)
        test_mask = np.asarray([rid in test_rids for rid in rid_epoch], dtype=bool)

        scaler_e = StandardScaler()
        X_train_e = scaler_e.fit_transform(X_epoch[train_mask])
        X_test_e = scaler_e.transform(X_epoch[test_mask])

        clf_epoch = get_logistic_regression(class_weight="balanced", random_state=args.seed + fold_idx)
        clf_epoch.fit(X_train_e, y_epoch[train_mask])
        p_train = clf_epoch.predict_proba(X_train_e)[:, 1]
        p_test = clf_epoch.predict_proba(X_test_e)[:, 1]

        X_subj_train, y_subj_train, subj_feature_names, _ = build_subject_features(
            X_epoch[train_mask],
            y_epoch[train_mask],
            [rid for rid, keep in zip(rid_epoch, train_mask) if keep],
            p_train,
            feature_names,
            include_log_n_epochs=not args.exclude_recording_length,
        )
        X_subj_test, _, _, sids_test = build_subject_features(
            X_epoch[test_mask],
            y_epoch[test_mask],
            [rid for rid, keep in zip(rid_epoch, test_mask) if keep],
            p_test,
            feature_names,
            include_log_n_epochs=not args.exclude_recording_length,
        )

        scaler_s = StandardScaler()
        X_subj_train_scaled = scaler_s.fit_transform(X_subj_train)
        X_subj_test_scaled = scaler_s.transform(X_subj_test)

        clf_subject = get_logistic_regression(class_weight="balanced", random_state=args.seed + fold_idx)
        clf_subject.fit(X_subj_train_scaled, y_subj_train)
        p_subj_test = clf_subject.predict_proba(X_subj_test_scaled)[:, 1]

        for rid, prob in zip(sids_test, p_subj_test):
            cv_preds[subj_index[rid]] = prob
        for name, value in compute_feature_importance(clf_subject, scaler_s, subj_feature_names).items():
            all_importance[name].append(value)

        print(f"Fold {fold_idx + 1}/{args.k_folds}: {len(test_rids)} test records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold = 0.5
    rng = np.random.RandomState(args.seed)
    metrics = result_dict(
        "mishra_finaldata_plus_lakshya_healthy_max_dedup_10fold_cv",
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
                "manifest": str(args.manifest),
                "rid_to_patient": str(args.rid_to_patient) if args.rid_to_patient else None,
                "feature_cache": str(args.feature_cache),
                "k_folds": args.k_folds,
                "threshold": threshold,
                "include_log_n_epochs": not args.exclude_recording_length,
                "label_map": label_map,
                "n_segments": int(len(rows)),
                "n_records": int(len(record_ids)),
                "n_patients": int(len(set(patient_by_rid.values()))),
                "metrics": metrics,
                "feature_importance": importance,
            },
            f,
            indent=2,
        )

    if not args.no_plots:
        plot_roc(y_subject, cv_preds, "Mishra prepared data 10-fold CV ROC", args.output_dir / "cv_roc.png")
        plot_score_distributions(y_subject, cv_preds, "Mishra prepared data CV Scores", args.output_dir / "cv_scores.png")
        plot_calibration_curve(y_subject, cv_preds, "Mishra prepared data CV Calibration", args.output_dir / "cv_calibration.png")
        plot_decision_curve(y_subject, cv_preds, "Mishra prepared data CV Decision Curve", args.output_dir / "cv_decision_curve.png")
        plot_error_analysis(
            analyze_errors(y_subject, (cv_preds >= threshold).astype(int), cv_preds, subject_ids),
            "Mishra prepared data CV Error Analysis",
            args.output_dir / "cv_error_analysis.png",
        )
        plot_feature_importance(importance, "Mishra prepared data CV Feature Importance", args.output_dir / "cv_feature_importance.png")

    print(
        f"\nMishra prepared data 10-fold CV: "
        f"AUC={metrics['auc']:.3f} "
        f"Acc={metrics['accuracy']:.3f} "
        f"Sens={metrics['sensitivity']:.3f} "
        f"Spec={metrics['specificity']:.3f}"
    )
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
