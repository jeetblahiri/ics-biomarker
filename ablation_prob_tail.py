"""Ablation: contribution of probability tail features (5) on top of ICS summaries (26).

For each centre we run the same two-stage subject-wise five-fold CV used in the
manuscript, with and without the five Stage-2 probability-tail features
(p_mean, p_std, p_q90, p_q95, p_mean_top20). Recording length is excluded as
a predictor in both arms so that the comparison isolates the probability-tail
contribution.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from ics_code import (  # noqa: E402
    build_epoch_features,
    find_all_subjects,
    get_logistic_regression,
    make_stratified_subject_folds,
)


ROOT = Path(__file__).parent
IHBAS = ROOT / "preprocessed_ihbas"
MAX = ROOT / "preprocessed_max"
SEG = 10.0
K = 5
SEED = 0
EPS = 1e-12


def build_subject_features_split(
    X_epoch: np.ndarray,
    y_epoch: np.ndarray,
    subj_ids_epoch,
    proba_epoch: np.ndarray,
):
    """Return (means+stds [26], prob tails [5], labels, subject_ids)."""
    idx_by_sid = defaultdict(list)
    for i, sid in enumerate(subj_ids_epoch):
        idx_by_sid[sid].append(i)
    subj_ids = sorted(idx_by_sid.keys())

    ics_part, tail_part, labels = [], [], []
    for sid in subj_ids:
        idxs = np.asarray(idx_by_sid[sid], dtype=int)
        Xe, pe, ye = X_epoch[idxs], proba_epoch[idxs], y_epoch[idxs]
        labels.append(int(np.bincount(ye).argmax()))
        ics_part.append(np.concatenate([Xe.mean(axis=0), Xe.std(axis=0)]))
        p = np.asarray(pe, dtype=np.float64)
        k_top = max(1, int(round(0.2 * p.size))) if p.size > 0 else 1
        tail_part.append(np.array([
            p.mean() if p.size > 0 else 0.0,
            p.std() if p.size > 0 else 0.0,
            np.percentile(p, 90.0) if p.size > 0 else 0.0,
            np.percentile(p, 95.0) if p.size > 0 else 0.0,
            np.sort(p)[-k_top:].mean() if p.size > 0 else 0.0,
        ], dtype=np.float64))

    return (
        np.nan_to_num(np.stack(ics_part, axis=0), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.stack(tail_part, axis=0), nan=0.0, posinf=0.0, neginf=0.0),
        np.asarray(labels, dtype=np.int64),
        subj_ids,
    )


def cv_auc(X_subj_full, X_subj_ics, y_subj_true, folds, subject_ids, epoch_class_weight=None,
            X_epoch=None, y_epoch=None, subj_ids_epoch=None):
    """Run the two-stage CV; return (auc_full31, auc_ics26)."""
    preds_full = np.zeros(len(subject_ids), dtype=np.float32)
    preds_ics = np.zeros(len(subject_ids), dtype=np.float32)
    sid_to_idx = {s: i for i, s in enumerate(subject_ids)}

    for fold_idx in range(K):
        val_subs = set(folds[fold_idx])
        train_subs = set(subject_ids) - val_subs

        train_mask_e = np.array([sid in train_subs for sid in subj_ids_epoch])
        val_mask_e = np.array([sid in val_subs for sid in subj_ids_epoch])

        scaler_e = StandardScaler()
        Xtr_e = scaler_e.fit_transform(X_epoch[train_mask_e])
        Xva_e = scaler_e.transform(X_epoch[val_mask_e])

        clf_e = get_logistic_regression(class_weight=epoch_class_weight, random_state=fold_idx)
        clf_e.fit(Xtr_e, y_epoch[train_mask_e])
        p_train = clf_e.predict_proba(Xtr_e)[:, 1]
        p_val = clf_e.predict_proba(Xva_e)[:, 1]

        ics_tr, tail_tr, y_tr, sids_tr = build_subject_features_split(
            X_epoch[train_mask_e], y_epoch[train_mask_e],
            [s for s, m in zip(subj_ids_epoch, train_mask_e) if m], p_train)
        ics_va, tail_va, _, sids_va = build_subject_features_split(
            X_epoch[val_mask_e], y_epoch[val_mask_e],
            [s for s, m in zip(subj_ids_epoch, val_mask_e) if m], p_val)

        # 31-feature model
        Xs_tr_full = np.concatenate([ics_tr, tail_tr], axis=1)
        Xs_va_full = np.concatenate([ics_va, tail_va], axis=1)
        sc_full = StandardScaler()
        Xs_tr_full_s = sc_full.fit_transform(Xs_tr_full)
        Xs_va_full_s = sc_full.transform(Xs_va_full)
        clf_full = get_logistic_regression(random_state=fold_idx)
        clf_full.fit(Xs_tr_full_s, y_tr)
        pv_full = clf_full.predict_proba(Xs_va_full_s)[:, 1]

        # 26-feature model (ICS means + stds only)
        sc_ics = StandardScaler()
        Xs_tr_ics_s = sc_ics.fit_transform(ics_tr)
        Xs_va_ics_s = sc_ics.transform(ics_va)
        clf_ics = get_logistic_regression(random_state=fold_idx)
        clf_ics.fit(Xs_tr_ics_s, y_tr)
        pv_ics = clf_ics.predict_proba(Xs_va_ics_s)[:, 1]

        for sid, pf, pi in zip(sids_va, pv_full, pv_ics):
            preds_full[sid_to_idx[sid]] = pf
            preds_ics[sid_to_idx[sid]] = pi

    return (
        roc_auc_score(y_subj_true, preds_full),
        roc_auc_score(y_subj_true, preds_ics),
        preds_full,
        preds_ics,
    )


def bootstrap_delta_ci(y_true, p_full, p_ics, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            deltas[b] = np.nan
            continue
        deltas[b] = roc_auc_score(yt, p_full[idx]) - roc_auc_score(yt, p_ics[idx])
    deltas = deltas[np.isfinite(deltas)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run_centre(name: str, root: Path, epoch_class_weight):
    rng = np.random.RandomState(SEED)
    records = find_all_subjects(root, name)
    print(f"\n=== {name}: {len(records)} subjects ===")
    X_ics, y_epoch, _, subj_ids_epoch, _ = build_epoch_features(records, SEG)
    subj_to_label = {}
    for sid, y in zip(subj_ids_epoch, y_epoch):
        subj_to_label.setdefault(sid, int(y))
    subject_ids = sorted(subj_to_label.keys())
    y_subj = np.array([subj_to_label[s] for s in subject_ids], dtype=int)
    folds = make_stratified_subject_folds(subject_ids, y_subj, K, rng)

    auc_full, auc_ics, p_full, p_ics = cv_auc(
        None, None, y_subj, folds, subject_ids,
        epoch_class_weight=epoch_class_weight,
        X_epoch=X_ics, y_epoch=y_epoch, subj_ids_epoch=subj_ids_epoch,
    )
    lo, hi = bootstrap_delta_ci(y_subj, p_full, p_ics, n_boot=1000, seed=SEED)
    print(f"  AUC 31-feature (ICS + prob-tail): {auc_full:.3f}")
    print(f"  AUC 26-feature (ICS only):       {auc_ics:.3f}")
    print(f"  Delta (tail contribution):       {auc_full - auc_ics:+.3f} (95% CI: {lo:+.3f} to {hi:+.3f})")
    return auc_full, auc_ics, lo, hi


def main():
    summary = {}
    summary["IHBAS"] = run_centre("IHBAS", IHBAS, epoch_class_weight=None)
    summary["MAX"] = run_centre("MAX", MAX, epoch_class_weight="balanced")
    print("\n=== Summary ===")
    for k, (af, ai, lo, hi) in summary.items():
        print(f"  {k}: 31-feat AUC {af:.3f} vs 26-feat AUC {ai:.3f} (Δ {af-ai:+.3f}, CI {lo:+.3f}..{hi:+.3f})")


if __name__ == "__main__":
    main()
