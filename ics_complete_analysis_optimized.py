#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete ICS-based epilepsy vs mimicker biomarker analysis.

Usage:
    python ics_complete_analysis_optimized.py \
        --ihbas_root preprocessed_ihbas \
        --max_root preprocessed_max \
        --seg_sec 10 \
        --k_folds 5 \
        --n_boot 1000 \
        --n_jobs -1 \
        --cache_dir feature_cache \
        --output_dir ics_results_final \
        --run_sampling_invariance
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import hashlib
import json
import math
import os
import pickle
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, resample_poly
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ----------------------------- Constants -----------------------------

COMMON_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
CH_INDEX = {name: idx for idx, name in enumerate(COMMON_ORDER)}


@dataclass
class SubjectRecord:
    subject_id: str
    path: Path
    label: int
    split: str
    center: str = ""


@dataclass
class AnalysisResults:
    auc: float
    ci_low: float
    ci_high: float
    accuracy: float
    sensitivity: float
    specificity: float
    brier_score: float
    u_stat: float
    p_val: float
    y_true: np.ndarray
    y_score: np.ndarray
    subject_ids: List[str]
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)


# ----------------------------- IO Helpers -----------------------------


def find_all_subjects(root: Path, center_name: str = "") -> List[SubjectRecord]:
    records: List[SubjectRecord] = []
    for split in ["Train", "Test"]:
        for cls in ["Epileptic", "Mimickers"]:
            base = root / split / cls
            if not base.exists():
                continue
            for f in sorted(base.glob("*.npy")):
                if f.name.startswith("._"):
                    continue
                label = 1 if cls == "Epileptic" else 0
                records.append(SubjectRecord(f.stem, f, label, split, center_name))
    
    by_id: Dict[str, int] = {}
    chosen: Dict[str, SubjectRecord] = {}
    for r in records:
        sid = r.subject_id
        if sid not in by_id:
            by_id[sid] = r.label
            chosen[sid] = r
        elif by_id[sid] != r.label:
            by_id[sid] = -1
            chosen.pop(sid, None)
    
    return [rec for sid, rec in chosen.items() if by_id[sid] in (0, 1)]


def get_channel_index(name: str, n_channels: int) -> int:
    idx = CH_INDEX.get(name, None)
    return idx if idx is not None and idx < n_channels else -1


# ----------------------------- Feature Caching -----------------------------


def get_cache_key(records: List[SubjectRecord], seg_sec: float, target_fs: Optional[float]) -> str:
    """Generate a unique cache key based on data files and parameters."""
    paths_str = "|".join(sorted(str(r.path) for r in records))
    params_str = f"seg={seg_sec}_fs={target_fs}"
    combined = f"{paths_str}|{params_str}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def load_cached_features(cache_dir: Path, cache_key: str) -> Optional[Tuple]:
    """Load cached features if available."""
    cache_file = cache_dir / f"features_{cache_key}.npz"
    meta_file = cache_dir / f"features_{cache_key}_meta.pkl"
    
    if cache_file.exists() and meta_file.exists():
        try:
            data = np.load(cache_file)
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            return data['X'], data['y'], meta['feature_names'], meta['subj_ids']
        except Exception:
            return None
    return None


def save_cached_features(cache_dir: Path, cache_key: str, X: np.ndarray, y: np.ndarray, 
                         feature_names: List[str], subj_ids: List[str]):
    """Save features to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"features_{cache_key}.npz"
    meta_file = cache_dir / f"features_{cache_key}_meta.pkl"
    
    np.savez_compressed(cache_file, X=X, y=y)
    with open(meta_file, 'wb') as f:
        pickle.dump({'feature_names': feature_names, 'subj_ids': subj_ids}, f)


# ----------------------------- Feature Computation -----------------------------


def compute_epoch_spectral_features(epoch: np.ndarray, sfreq: float) -> Dict[str, float]:
    C, T = epoch.shape
    eps = 1e-8
    
    if sfreq <= 0 or T < 16:
        return {
            "global_slowing_index": 0.0, "rel_delta_power": 0.0, "rel_theta_power": 0.0,
            "rel_alpha_power": 0.0, "rel_beta_power": 0.0, "occipital_pdr_freq": 0.0,
            "occipital_alpha_asymmetry": 0.0, "frontal_slowing_index": 0.0,
            "temporal_slowing_index": 0.0, "occipital_slowing_index": 0.0,
        }
    
    nperseg = min(int(4 * sfreq), T)
    if nperseg < 16:
        nperseg = T
    
    freqs, psd = welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)
    
    band_edges = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    
    band_powers: Dict[str, np.ndarray] = {}
    total_power = np.zeros(C, dtype=np.float64)
    
    for band, (fmin, fmax) in band_edges.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = psd[:, mask].sum(axis=1) if np.any(mask) else np.zeros(C, dtype=np.float64)
        band_powers[band] = bp
        total_power += bp
    
    rel_powers = {band: band_powers[band] / (total_power + eps) for band in band_edges}
    
    slowing = (band_powers["delta"] + band_powers["theta"]).mean()
    fast = (band_powers["alpha"] + band_powers["beta"]).mean()
    global_slowing_index = slowing / (fast + eps)
    
    regions = {
        "frontal": ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"],
        "temporal": ["F7", "T7", "P7", "F8", "T8", "P8"],
        "occipital": ["O1", "O2"],
    }
    
    def region_slowing(names):
        idxs = [i for i in [get_channel_index(n, C) for n in names] if i >= 0]
        if not idxs:
            return 0.0
        s = (band_powers["delta"][idxs] + band_powers["theta"][idxs]).mean()
        f = (band_powers["alpha"][idxs] + band_powers["beta"][idxs]).mean()
        return float(s / (f + eps))
    
    alpha_mask = (freqs >= 7.0) & (freqs <= 14.0)
    i_o1, i_o2 = get_channel_index("O1", C), get_channel_index("O2", C)
    
    pdr_freq, alpha_asym = 0.0, 0.0
    
    occ_idx = [i for i in [i_o1, i_o2] if i >= 0]
    if occ_idx and np.any(alpha_mask):
        alpha_psd = psd[occ_idx, :][:, alpha_mask].mean(axis=0)
        if alpha_psd.size > 0:
            pdr_freq = float(freqs[alpha_mask][np.argmax(alpha_psd)])
    
    if i_o1 >= 0 and i_o2 >= 0:
        a1, a2 = band_powers["alpha"][i_o1], band_powers["alpha"][i_o2]
        alpha_asym = float((a1 - a2) / (a1 + a2 + eps))
    
    feats = {
        "global_slowing_index": float(global_slowing_index),
        "rel_delta_power": float(rel_powers["delta"].mean()),
        "rel_theta_power": float(rel_powers["theta"].mean()),
        "rel_alpha_power": float(rel_powers["alpha"].mean()),
        "rel_beta_power": float(rel_powers["beta"].mean()),
        "occipital_pdr_freq": float(pdr_freq),
        "occipital_alpha_asymmetry": float(alpha_asym),
        "frontal_slowing_index": region_slowing(regions["frontal"]),
        "temporal_slowing_index": region_slowing(regions["temporal"]),
        "occipital_slowing_index": region_slowing(regions["occipital"]),
    }
    
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    
    return feats


def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < m * tau:
        return 0.0
    
    n_patterns = n - (m - 1) * tau
    embedded = np.column_stack([x[i*tau:i*tau + n_patterns] for i in range(m)])
    patterns = np.argsort(embedded, axis=1)
    
    pattern_dict: Dict[Tuple[int, ...], int] = {}
    for row in patterns:
        key = tuple(row)
        pattern_dict[key] = pattern_dict.get(key, 0) + 1
    
    counts = np.array(list(pattern_dict.values()), dtype=np.float64)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    H_norm = H / math.log(math.factorial(m))
    return float(H_norm)


def compute_network_features(epoch: np.ndarray) -> Dict[str, float]:
    C, T = epoch.shape
    if C < 2 or T < 4:
        return {"network_mean_corr": 0.0, "network_strength_std": 0.0}
    
    corr = np.corrcoef(epoch)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    iu = np.triu_indices(C, k=1)
    vals = np.abs(corr[iu])
    mean_corr = float(vals.mean()) if vals.size > 0 else 0.0
    
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    strengths = abs_corr.sum(axis=1)
    strength_std = float(strengths.std())
    
    return {"network_mean_corr": mean_corr, "network_strength_std": strength_std}


def compute_epoch_ics(epoch: np.ndarray, seg_sec: float) -> Dict[str, float]:
    C, T = epoch.shape
    sfreq = float(T) / float(seg_sec) if seg_sec > 0 else 0.0
    
    spec_feats = compute_epoch_spectral_features(epoch, sfreq)
    pe_vals = [permutation_entropy(epoch[ch], m=3, tau=1) for ch in range(C)]
    pe_mean = float(np.mean(pe_vals)) if pe_vals else 0.0
    net_feats = compute_network_features(epoch)
    
    feats = {}
    feats.update(spec_feats)
    feats["pe_mean"] = pe_mean
    feats.update(net_feats)
    
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0
    
    return feats


# ----------------------------- Resampling -----------------------------


def resample_epoch(epoch: np.ndarray, orig_fs: float, target_fs: float, seg_sec: float) -> np.ndarray:
    if math.isclose(orig_fs, target_fs, rel_tol=1e-6, abs_tol=1e-6):
        return epoch
    
    target_len = int(round(seg_sec * target_fs))
    frac = Fraction(int(round(target_fs)), int(round(orig_fs))).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    
    resampled = resample_poly(epoch, up, down, axis=-1)
    
    if resampled.shape[1] > target_len:
        resampled = resampled[:, :target_len]
    elif resampled.shape[1] < target_len:
        pad = target_len - resampled.shape[1]
        resampled = np.pad(resampled, ((0, 0), (0, pad)), mode="edge")
    
    return resampled


# ----------------------------- Parallel Feature Processing -----------------------------


def _process_single_record(args):
    """Process a single subject record (for parallel execution)."""
    rec_path, rec_label, rec_subject_id, seg_sec, target_fs = args
    
    results = []
    try:
        arr = np.load(rec_path)
        if arr.ndim != 3 or arr.shape[0] == 0:
            return results
        
        N_seg, C, T = arr.shape
        orig_fs = float(T) / float(seg_sec)
        
        for seg_idx in range(N_seg):
            epoch = arr[seg_idx]
            if target_fs is not None and not math.isclose(orig_fs, target_fs, rel_tol=1e-6):
                epoch = resample_epoch(epoch, orig_fs, target_fs, seg_sec)
            
            feats = compute_epoch_ics(epoch, seg_sec)
            results.append((feats, rec_label, rec_subject_id))
    except Exception:
        pass
    
    return results


def build_epoch_features_parallel(
    records: List[SubjectRecord], 
    seg_sec: float, 
    target_fs: Optional[float] = None,
    n_jobs: int = -1,
    cache_dir: Optional[Path] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build epoch-level features with parallel processing and caching."""
    
    # Check cache first
    if cache_dir is not None:
        cache_key = get_cache_key(records, seg_sec, target_fs)
        cached = load_cached_features(cache_dir, cache_key)
        if cached is not None:
            if show_progress:
                print(f"  Loaded {cached[0].shape[0]} epochs from cache")
            return cached
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    elif n_jobs <= 0:
        n_jobs = 1
    
    # Prepare arguments for parallel processing
    args_list = [
        (str(rec.path), rec.label, rec.subject_id, seg_sec, target_fs)
        for rec in records
    ]
    
    all_feats, labels, subj_ids = [], [], []
    
    if n_jobs == 1:
        # Sequential processing
        for i, args in enumerate(args_list):
            if show_progress and (i + 1) % 50 == 0:
                print(f"    Processing subject {i + 1}/{len(args_list)}...")
            results = _process_single_record(args)
            for feats, label, sid in results:
                all_feats.append(feats)
                labels.append(label)
                subj_ids.append(sid)
    else:
        # Parallel processing
        if show_progress:
            print(f"  Processing {len(records)} subjects with {n_jobs} workers...")
        
        # Use spawn context to avoid issues on Windows
        ctx = multiprocessing.get_context('spawn')
        
        completed = 0
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
            futures = {executor.submit(_process_single_record, args): i for i, args in enumerate(args_list)}
            
            for future in as_completed(futures):
                completed += 1
                if show_progress and completed % 100 == 0:
                    print(f"    Completed {completed}/{len(futures)} subjects...")
                
                try:
                    results = future.result()
                    for feats, label, sid in results:
                        all_feats.append(feats)
                        labels.append(label)
                        subj_ids.append(sid)
                except Exception:
                    pass
    
    if not all_feats:
        raise RuntimeError("No epoch features computed; check data paths.")
    
    # Convert to arrays
    feature_names = sorted(all_feats[0].keys())
    X = np.zeros((len(all_feats), len(feature_names)), dtype=np.float32)
    for i, fdict in enumerate(all_feats):
        for j, name in enumerate(feature_names):
            X[i, j] = float(fdict.get(name, 0.0))
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(labels, dtype=np.int64)
    
    # Save to cache
    if cache_dir is not None:
        save_cached_features(cache_dir, cache_key, X, y, feature_names, subj_ids)
        if show_progress:
            print(f"  Cached {X.shape[0]} epochs to disk")
    
    return X, y, feature_names, subj_ids


# ----------------------------- Subject-Level Features -----------------------------


def build_subject_features(X_epoch, y_epoch, subj_ids_epoch, proba_epoch, feature_names):
    """Build subject-level features with probability tail statistics."""
    subj_epoch_indices = defaultdict(list)
    for idx, sid in enumerate(subj_ids_epoch):
        subj_epoch_indices[sid].append(idx)
    
    subj_ids_unique = sorted(subj_epoch_indices.keys())
    subj_feature_list, subj_labels = [], []
    
    for sid in subj_ids_unique:
        idxs = np.asarray(subj_epoch_indices[sid], dtype=int)
        Xe, pe, ye = X_epoch[idxs], proba_epoch[idxs], y_epoch[idxs]
        
        lab = int(np.bincount(ye).argmax())
        subj_labels.append(lab)
        
        means, stds = Xe.mean(axis=0), Xe.std(axis=0)
        feats = [*means, *stds]
        
        p = np.asarray(pe, dtype=np.float64)
        if p.size > 0:
            p_mean = p.mean()
            p_std = p.std()
            p_q90 = np.percentile(p, 90.0)
            p_q95 = np.percentile(p, 95.0)
            k_top = max(1, int(round(0.2 * p.size)))
            p_top20 = np.sort(p)[-k_top:].mean()
        else:
            p_mean = p_std = p_q90 = p_q95 = p_top20 = 0.0
        
        feats.extend([p_mean, p_std, p_q90, p_q95, p_top20])
        feats.append(math.log(len(idxs) + 1.0))
        
        subj_feature_list.append(np.asarray(feats, dtype=np.float32))
    
    X_subj = np.stack(subj_feature_list, axis=0)
    y_subj = np.asarray(subj_labels, dtype=np.int64)
    
    subj_feature_names = [f"mean_{n}" for n in feature_names] + [f"std_{n}" for n in feature_names]
    subj_feature_names.extend(["p_mean", "p_std", "p_q90", "p_q95", "p_mean_top20", "log_n_epochs"])
    
    return np.nan_to_num(X_subj, nan=0.0, posinf=0.0, neginf=0.0), y_subj, subj_feature_names, subj_ids_unique


# ----------------------------- Model Utilities -----------------------------


def get_logistic_regression(class_weight=None, random_state=0):
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight=class_weight, random_state=random_state)


# ----------------------------- Analysis Functions -----------------------------


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * group1.var(ddof=1) + (n2 - 1) * group2.var(ddof=1)) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 1e-10 else 0.0


def compute_feature_effect_sizes(X, y, feature_names):
    return {name: cohens_d(X[y == 1, i], X[y == 0, i]) for i, name in enumerate(feature_names)}


def compute_feature_importance(clf, scaler, feature_names):
    coef = clf.coef_[0]
    scaled_coef = coef * scaler.scale_ if hasattr(scaler, 'scale_') and scaler.scale_ is not None else coef
    return {name: float(scaled_coef[i]) for i, name in enumerate(feature_names)}


def make_stratified_subject_folds(subject_ids, subject_labels, k_folds, rng):
    epi_sids = [sid for sid, lab in zip(subject_ids, subject_labels) if lab == 1]
    mim_sids = [sid for sid, lab in zip(subject_ids, subject_labels) if lab == 0]
    rng.shuffle(epi_sids)
    rng.shuffle(mim_sids)
    folds = [[] for _ in range(k_folds)]
    for i, sid in enumerate(epi_sids):
        folds[i % k_folds].append(sid)
    for i, sid in enumerate(mim_sids):
        folds[i % k_folds].append(sid)
    return folds


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    n = len(y_true)
    if n < 2:
        return (float("nan"), float("nan"))
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except:
            continue
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (float("nan"), float("nan"))


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = float("nan")
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        sens = spec = float("nan")
    
    epi_scores, mim_scores = y_score[y_true == 1], y_score[y_true == 0]
    u_stat, p_val = mannwhitneyu(epi_scores, mim_scores, alternative="two-sided") if len(epi_scores) > 0 and len(mim_scores) > 0 else (float("nan"), float("nan"))
    
    return {"auc": auc, "accuracy": accuracy_score(y_true, y_pred), "sensitivity": sens, "specificity": spec,
            "brier_score": brier_score_loss(y_true, y_score), "u_stat": u_stat, "p_val": p_val}


def analyze_errors(y_true, y_pred, y_prob, subject_ids):
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tp_mask = (y_pred == 1) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)
    return {
        "n_fp": fp_mask.sum(), "n_fn": fn_mask.sum(), "n_tp": tp_mask.sum(), "n_tn": tn_mask.sum(),
        "tp_mean_prob": y_prob[tp_mask].mean() if tp_mask.sum() > 0 else 0,
        "tn_mean_prob": y_prob[tn_mask].mean() if tn_mask.sum() > 0 else 0,
        "fp_mean_prob": y_prob[fp_mask].mean() if fp_mask.sum() > 0 else 0,
        "fn_mean_prob": y_prob[fn_mask].mean() if fn_mask.sum() > 0 else 0,
        "prob_distributions": {"tp": y_prob[tp_mask], "tn": y_prob[tn_mask], "fp": y_prob[fp_mask], "fn": y_prob[fn_mask]}
    }


# ----------------------------- Plotting -----------------------------


def plot_roc(y_true, y_score, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#d62728', linewidth=2, label=f"AUC = {roc_auc_score(y_true, y_score):.3f}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_score_distributions(y_true, y_score, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_score[y_true == 0], bins=20, alpha=0.6, label='Mimickers', color='#1f77b4')
    ax.hist(y_score[y_true == 1], bins=20, alpha=0.6, label='Epilepsy', color='#d62728')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_calibration_curve(y_true, y_prob, title, save_path, n_bins=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax1.plot(prob_pred, prob_true, 's-', color='#d62728', label='ICS biomarker')
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"{title}\nBrier: {brier_score_loss(y_true, y_prob):.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label='Mimickers', color='#1f77b4')
    ax2.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label='Epilepsy', color='#d62728')
    ax2.set_xlabel("Predicted probability")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_decision_curve(y_true, y_prob, title, save_path):
    thresholds = np.linspace(0.01, 0.99, 99)
    n, prevalence = len(y_true), y_true.mean()
    nb_model, nb_all = [], []
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        nb_model.append((tp / n) - (fp / n) * (pt / (1 - pt + 1e-10)))
        nb_all.append(prevalence - (1 - prevalence) * (pt / (1 - pt + 1e-10)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, nb_model, '-', color='#d62728', linewidth=2, label='ICS biomarker')
    ax.plot(thresholds, nb_all, '--', color='gray', linewidth=1.5, label='Treat all')
    ax.plot(thresholds, np.zeros_like(thresholds), '-', color='black', linewidth=1, label='Treat none')
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_feature_importance(importance, title, save_path, top_n=20):
    sorted_feats = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names, values = [x[0] for x in sorted_feats], [x[1] for x in sorted_feats]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in values]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.arange(len(names)), values, color=colors, alpha=0.8)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


def plot_effect_size_comparison(es_ihbas, es_max, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    feature_names = list(es_ihbas.keys())
    x = np.arange(len(feature_names))
    width = 0.35
    ihbas_d = [abs(es_ihbas[f]) for f in feature_names]
    max_d = [abs(es_max[f]) for f in feature_names]
    ax.bar(x - width/2, ihbas_d, width, label='IHBAS', alpha=0.8, color='#1f77b4')
    ax.bar(x + width/2, max_d, width, label='MAX', alpha=0.8, color='#d62728')
    mean_d_ihbas, mean_d_max = np.mean(ihbas_d), np.mean(max_d)
    ax.set_xlabel("Feature")
    ax.set_ylabel("|Cohen's d|")
    ax.set_title(f"Feature Separability\n(Mean |d|: IHBAS={mean_d_ihbas:.3f}, MAX={mean_d_max:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')
    return mean_d_ihbas, mean_d_max


def plot_error_analysis(error_results, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cm = np.array([[error_results["n_tn"], error_results["n_fp"]], [error_results["n_fn"], error_results["n_tp"]]])
    ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Pred: Mimicker', 'Pred: Epilepsy'])
    ax1.set_yticklabels(['True: Mimicker', 'True: Epilepsy'])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16)
    ax1.set_title("Confusion Matrix")
    
    categories = ['TN', 'FP', 'FN', 'TP']
    keys = ['tn', 'fp', 'fn', 'tp']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    data = [error_results["prob_distributions"].get(k, [0.5]) for k in keys]
    data = [d if len(d) > 0 else [0.5] for d in data]
    bp = ax2.boxplot(data, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Predicted probability")
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close('all')


# ----------------------------- Single Center CV -----------------------------


def run_center_cv(center_name, records, seg_sec, k_folds, n_boot, epoch_class_weight, rng, output_dir,
                  n_jobs=1, cache_dir=None, target_fs=None, generate_plots=True):
    """Run subject-level K-fold CV for a single center."""
    if rng is None:
        rng = np.random.RandomState(0)
    if output_dir is None:
        output_dir = Path(".")
    
    print(f"\n{'='*60}")
    print(f"Running {center_name} center analysis" + (f" @ {target_fs} Hz" if target_fs else ""))
    print(f"{'='*60}")
    
    # Build epoch features
    print("Computing epoch-level ICS features...")
    X_ics, y_epoch, feature_names, subj_ids_epoch = build_epoch_features_parallel(
        records, seg_sec, target_fs=target_fs, n_jobs=n_jobs, cache_dir=cache_dir
    )
    print(f"Epoch features: {X_ics.shape[0]} epochs, {X_ics.shape[1]} features")
    
    # Get subject list
    subj_to_label = {}
    for sid, y in zip(subj_ids_epoch, y_epoch):
        if sid not in subj_to_label:
            subj_to_label[sid] = int(y)
    
    subject_ids = sorted(subj_to_label.keys())
    y_subj_true = np.array([subj_to_label[sid] for sid in subject_ids], dtype=int)
    n_subjects = len(subject_ids)
    
    print(f"Subjects: {n_subjects} (Epilepsy={y_subj_true.sum()}, Mimickers={(y_subj_true==0).sum()})")
    
    epochs_per_subj = defaultdict(int)
    for sid in subj_ids_epoch:
        epochs_per_subj[sid] += 1
    print(f"Mean epochs per subject: {np.mean(list(epochs_per_subj.values())):.1f}")
    
    # CV predictions
    cv_preds = np.zeros(n_subjects, dtype=np.float32)
    folds = make_stratified_subject_folds(subject_ids, y_subj_true, k_folds, rng)
    all_importance = defaultdict(list)
    
    print(f"\nRunning {k_folds}-fold cross-validation...")
    
    for fold_idx in range(k_folds):
        val_subs = set(folds[fold_idx])
        train_subs = set(subject_ids) - val_subs
        
        train_mask = np.array([sid in train_subs for sid in subj_ids_epoch])
        val_mask = np.array([sid in val_subs for sid in subj_ids_epoch])
        
        scaler_e = StandardScaler()
        X_train_e, X_val_e = scaler_e.fit_transform(X_ics[train_mask]), scaler_e.transform(X_ics[val_mask])
        
        clf_epoch = get_logistic_regression(class_weight=epoch_class_weight, random_state=fold_idx)
        clf_epoch.fit(X_train_e, y_epoch[train_mask])
        p_train, p_val = clf_epoch.predict_proba(X_train_e)[:, 1], clf_epoch.predict_proba(X_val_e)[:, 1]
        
        X_subj_train, y_subj_train, subj_feat_names, sids_train = build_subject_features(
            X_ics[train_mask], y_epoch[train_mask], [sid for sid, m in zip(subj_ids_epoch, train_mask) if m], p_train, feature_names)
        X_subj_val, _, _, sids_val = build_subject_features(
            X_ics[val_mask], y_epoch[val_mask], [sid for sid, m in zip(subj_ids_epoch, val_mask) if m], p_val, feature_names)
        
        scaler_s = StandardScaler()
        X_subj_train_scaled, X_subj_val_scaled = scaler_s.fit_transform(X_subj_train), scaler_s.transform(X_subj_val)
        
        clf_subj = get_logistic_regression(random_state=fold_idx)
        clf_subj.fit(X_subj_train_scaled, y_subj_train)
        p_subj_val = clf_subj.predict_proba(X_subj_val_scaled)[:, 1]
        
        subj_index = {sid: i for i, sid in enumerate(subject_ids)}
        for sid, p in zip(sids_val, p_subj_val):
            cv_preds[subj_index[sid]] = p
        
        for fname, val in compute_feature_importance(clf_subj, scaler_s, subj_feat_names).items():
            all_importance[fname].append(val)
        
        print(f"  Fold {fold_idx + 1}/{k_folds}: {len(val_subs)} validation subjects")
    
    avg_importance = {fname: np.mean(vals) for fname, vals in all_importance.items()}
    metrics = compute_metrics(y_subj_true, cv_preds)
    ci_low, ci_high = bootstrap_auc_ci(y_subj_true, cv_preds, n_boot=n_boot, rng=rng)
    
    print(f"\n{center_name} Results:")
    print(f"  AUC: {metrics['auc']:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    
    # Generate plots only for main analysis (not sampling invariance)
    if generate_plots:
        suffix = f"_{int(target_fs)}Hz" if target_fs else ""
        plot_roc(y_subj_true, cv_preds, f"{center_name} ROC (CV){suffix}", output_dir / f"{center_name}_roc{suffix}.png")
        plot_score_distributions(y_subj_true, cv_preds, f"{center_name} Scores{suffix}", output_dir / f"{center_name}_scores{suffix}.png")
        plot_feature_importance(avg_importance, f"{center_name} Feature Importance{suffix}", output_dir / f"{center_name}_feature_importance{suffix}.png")
        plot_calibration_curve(y_subj_true, cv_preds, f"{center_name} Calibration{suffix}", output_dir / f"{center_name}_calibration{suffix}.png")
        plot_decision_curve(y_subj_true, cv_preds, f"{center_name} Decision Curve{suffix}", output_dir / f"{center_name}_decision_curve{suffix}.png")
        
        y_pred = (cv_preds >= 0.5).astype(int)
        error_results = analyze_errors(y_subj_true, y_pred, cv_preds, subject_ids)
        plot_error_analysis(error_results, f"{center_name} Error Analysis{suffix}", output_dir / f"{center_name}_error_analysis{suffix}.png")
        
        print(f"\n{center_name} Error Analysis:")
        print(f"  TP: {error_results['n_tp']} (prob: {error_results['tp_mean_prob']:.3f})")
        print(f"  TN: {error_results['n_tn']} (prob: {error_results['tn_mean_prob']:.3f})")
        print(f"  FP: {error_results['n_fp']} (prob: {error_results['fp_mean_prob']:.3f})")
        print(f"  FN: {error_results['n_fn']} (prob: {error_results['fn_mean_prob']:.3f})")
    
    return AnalysisResults(
        auc=metrics["auc"], ci_low=ci_low, ci_high=ci_high, accuracy=metrics["accuracy"],
        sensitivity=metrics["sensitivity"], specificity=metrics["specificity"], brier_score=metrics["brier_score"],
        u_stat=metrics["u_stat"], p_val=metrics["p_val"], y_true=y_subj_true, y_score=cv_preds,
        subject_ids=subject_ids, feature_names=subj_feat_names, feature_importance=avg_importance,
    ), X_ics, y_epoch, feature_names, subj_ids_epoch


# ----------------------------- Cross-Center Generalization (CORRECTED) -----------------------------


def cross_center_generalization_corrected(X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max, feat_names):
    """
    CORRECTED cross-center generalization.
    Probability tail features for TEST center come from TRAIN center model.
    """
    results = {}
    
    print("\n  Computing cross-center generalization (corrected)...")
    
    # Train on MAX, test on IHBAS
    scaler_e_max = StandardScaler()
    X_max_scaled = scaler_e_max.fit_transform(X_max)
    clf_e_max = get_logistic_regression(class_weight="balanced", random_state=0)
    clf_e_max.fit(X_max_scaled, y_max)
    
    X_ihbas_scaled_by_max = scaler_e_max.transform(X_ihbas)
    p_ihbas_from_max = clf_e_max.predict_proba(X_ihbas_scaled_by_max)[:, 1]
    p_max_train = clf_e_max.predict_proba(X_max_scaled)[:, 1]
    
    X_subj_max_train, y_subj_max_train, _, _ = build_subject_features(X_max, y_max, sids_max, p_max_train, feat_names)
    X_subj_ihbas_test, y_subj_ihbas_test, _, _ = build_subject_features(X_ihbas, y_ihbas, sids_ihbas, p_ihbas_from_max, feat_names)
    
    scaler_s = StandardScaler()
    X_subj_max_scaled = scaler_s.fit_transform(X_subj_max_train)
    X_subj_ihbas_scaled = scaler_s.transform(X_subj_ihbas_test)
    
    clf_s = get_logistic_regression(random_state=0)
    clf_s.fit(X_subj_max_scaled, y_subj_max_train)
    p_ihbas_final = clf_s.predict_proba(X_subj_ihbas_scaled)[:, 1]
    
    results["train_MAX_test_IHBAS"] = roc_auc_score(y_subj_ihbas_test, p_ihbas_final)
    
    # Train on IHBAS, test on MAX
    scaler_e_ihbas = StandardScaler()
    X_ihbas_scaled = scaler_e_ihbas.fit_transform(X_ihbas)
    clf_e_ihbas = get_logistic_regression(random_state=0)
    clf_e_ihbas.fit(X_ihbas_scaled, y_ihbas)
    
    X_max_scaled_by_ihbas = scaler_e_ihbas.transform(X_max)
    p_max_from_ihbas = clf_e_ihbas.predict_proba(X_max_scaled_by_ihbas)[:, 1]
    p_ihbas_train = clf_e_ihbas.predict_proba(X_ihbas_scaled)[:, 1]
    
    X_subj_ihbas_train, y_subj_ihbas_train, _, _ = build_subject_features(X_ihbas, y_ihbas, sids_ihbas, p_ihbas_train, feat_names)
    X_subj_max_test, y_subj_max_test, _, _ = build_subject_features(X_max, y_max, sids_max, p_max_from_ihbas, feat_names)
    
    scaler_s2 = StandardScaler()
    X_subj_ihbas_scaled2 = scaler_s2.fit_transform(X_subj_ihbas_train)
    X_subj_max_scaled2 = scaler_s2.transform(X_subj_max_test)
    
    clf_s2 = get_logistic_regression(random_state=0)
    clf_s2.fit(X_subj_ihbas_scaled2, y_subj_ihbas_train)
    p_max_final = clf_s2.predict_proba(X_subj_max_scaled2)[:, 1]
    
    results["train_IHBAS_test_MAX"] = roc_auc_score(y_subj_max_test, p_max_final)
    
    return results


# ----------------------------- Pooled Multi-Center CV -----------------------------


def run_pooled_cv(X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max, feat_names, k_folds, n_boot, rng, output_dir):
    """Run properly cross-validated pooled multi-center analysis."""
    print("\n" + "="*60)
    print("POOLED MULTI-CENTER MODEL (WITH PROPER CV)")
    print("="*60)
    
    X_all = np.vstack([X_ihbas, X_max])
    y_all = np.concatenate([y_ihbas, y_max])
    sids_all = [f"IHBAS_{sid}" for sid in sids_ihbas] + [f"MAX_{sid}" for sid in sids_max]
    
    subj_to_label = {}
    for sid, y in zip(sids_all, y_all):
        if sid not in subj_to_label:
            subj_to_label[sid] = int(y)
    
    subject_ids = sorted(subj_to_label.keys())
    y_subj_true = np.array([subj_to_label[sid] for sid in subject_ids], dtype=int)
    n_subjects = len(subject_ids)
    
    print(f"Combined: {n_subjects} subjects (IHBAS: {sum(1 for s in subject_ids if s.startswith('IHBAS_'))}, MAX: {sum(1 for s in subject_ids if s.startswith('MAX_'))})")
    
    cv_preds = np.zeros(n_subjects, dtype=np.float32)
    folds = make_stratified_subject_folds(subject_ids, y_subj_true, k_folds, rng)
    
    print(f"\nRunning {k_folds}-fold CV on pooled data...")
    
    for fold_idx in range(k_folds):
        val_subs, train_subs = set(folds[fold_idx]), set(subject_ids) - set(folds[fold_idx])
        train_mask = np.array([sid in train_subs for sid in sids_all])
        val_mask = np.array([sid in val_subs for sid in sids_all])
        
        scaler_e = StandardScaler()
        X_train_e, X_val_e = scaler_e.fit_transform(X_all[train_mask]), scaler_e.transform(X_all[val_mask])
        
        clf_epoch = get_logistic_regression(class_weight="balanced", random_state=fold_idx)
        clf_epoch.fit(X_train_e, y_all[train_mask])
        p_train, p_val = clf_epoch.predict_proba(X_train_e)[:, 1], clf_epoch.predict_proba(X_val_e)[:, 1]
        
        X_subj_train, y_subj_train, _, sids_train = build_subject_features(
            X_all[train_mask], y_all[train_mask], [sid for sid, m in zip(sids_all, train_mask) if m], p_train, feat_names)
        X_subj_val, _, _, sids_val = build_subject_features(
            X_all[val_mask], y_all[val_mask], [sid for sid, m in zip(sids_all, val_mask) if m], p_val, feat_names)
        
        scaler_s = StandardScaler()
        clf_subj = get_logistic_regression(random_state=fold_idx)
        clf_subj.fit(scaler_s.fit_transform(X_subj_train), y_subj_train)
        p_subj_val = clf_subj.predict_proba(scaler_s.transform(X_subj_val))[:, 1]
        
        subj_index = {sid: i for i, sid in enumerate(subject_ids)}
        for sid, p in zip(sids_val, p_subj_val):
            cv_preds[subj_index[sid]] = p
        
        print(f"  Fold {fold_idx + 1}/{k_folds}: {len(val_subs)} validation subjects")
    
    metrics = compute_metrics(y_subj_true, cv_preds)
    ci_low, ci_high = bootstrap_auc_ci(y_subj_true, cv_preds, n_boot=n_boot, rng=rng)
    
    ihbas_mask = np.array([sid.startswith("IHBAS_") for sid in subject_ids])
    max_mask = np.array([sid.startswith("MAX_") for sid in subject_ids])
    
    print(f"\nPooled CV Results:")
    print(f"  Overall AUC: {metrics['auc']:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
    print(f"  IHBAS-only AUC: {roc_auc_score(y_subj_true[ihbas_mask], cv_preds[ihbas_mask]):.3f}")
    print(f"  MAX-only AUC: {roc_auc_score(y_subj_true[max_mask], cv_preds[max_mask]):.3f}")
    
    plot_roc(y_subj_true, cv_preds, "Pooled ROC (CV)", output_dir / "pooled_roc.png")
    plot_calibration_curve(y_subj_true, cv_preds, "Pooled Calibration", output_dir / "pooled_calibration.png")
    
    return {
        "overall_auc": metrics["auc"], "overall_ci_low": ci_low, "overall_ci_high": ci_high,
        "ihbas_auc": roc_auc_score(y_subj_true[ihbas_mask], cv_preds[ihbas_mask]),
        "max_auc": roc_auc_score(y_subj_true[max_mask], cv_preds[max_mask]),
    }


# ----------------------------- Sampling Rate Invariance (FIXED) -----------------------------


def run_sampling_invariance(ihbas_root, max_root, ihbas_records, max_records, seg_sec, sampling_rates,
                            k_folds, n_boot, output_dir, rng, n_jobs, cache_dir):
    """
    FIXED: Run single-center CV separately for each center at each sampling rate.
    This ensures consistency with main single-center results.
    """
    print("\n" + "="*60)
    print("SAMPLING RATE INVARIANCE ANALYSIS (FIXED)")
    print("="*60)
    
    if not max_records or not ihbas_records:
        print("Error: Could not find subjects.")
        return {}
    
    # Get original sampling rates
    arr_max = np.load(max_records[0].path)
    orig_fs_max = int(round(arr_max.shape[2] / seg_sec))
    arr_ihb = np.load(ihbas_records[0].path)
    orig_fs_ihb = int(round(arr_ihb.shape[2] / seg_sec))
    
    print(f"Original fs: MAX={orig_fs_max} Hz, IHBAS={orig_fs_ihb} Hz")
    
    fs_max_allowed = min(orig_fs_max, orig_fs_ihb)
    
    results = {"ihbas": {}, "max": {}, "pooled": {}}
    
    for fs_target in sampling_rates:
        if fs_target > fs_max_allowed + 1e-6:
            print(f"\nSkipping fs={fs_target} Hz (> max {fs_max_allowed} Hz)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Target sampling rate: {fs_target} Hz")
        print(f"{'='*60}")
        
        # Run single-center CV for IHBAS
        ihbas_result, X_ihbas, y_ihbas, feat_names, sids_ihbas = run_center_cv(
            "IHBAS", ihbas_records, seg_sec, k_folds, n_boot, None, rng, output_dir,
            n_jobs=n_jobs, cache_dir=cache_dir, target_fs=fs_target, generate_plots=False
        )
        results["ihbas"][fs_target] = ihbas_result.auc
        
        # Run single-center CV for MAX
        max_result, X_max, y_max, _, sids_max = run_center_cv(
            "MAX", max_records, seg_sec, k_folds, n_boot, "balanced", rng, output_dir,
            n_jobs=n_jobs, cache_dir=cache_dir, target_fs=fs_target, generate_plots=False
        )
        results["max"][fs_target] = max_result.auc
        
        # Run pooled CV
        pooled_result = run_pooled_cv(
            X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max,
            feat_names, k_folds, n_boot, rng, output_dir
        )
        results["pooled"][fs_target] = pooled_result["overall_auc"]
        
        print(f"\nSummary @ {fs_target} Hz:")
        print(f"  IHBAS: AUC = {ihbas_result.auc:.3f}")
        print(f"  MAX:   AUC = {max_result.auc:.3f}")
        print(f"  Pooled: AUC = {pooled_result['overall_auc']:.3f}")
    
    # Plot results
    if results["ihbas"]:
        fs_sorted = sorted(results["ihbas"].keys(), reverse=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fs_sorted, [results["ihbas"][fs] for fs in fs_sorted], 'o-', label='IHBAS', linewidth=2, markersize=8)
        ax.plot(fs_sorted, [results["max"][fs] for fs in fs_sorted], 's-', label='MAX', linewidth=2, markersize=8)
        ax.plot(fs_sorted, [results["pooled"][fs] for fs in fs_sorted], '^-', label='Pooled', linewidth=2, markersize=8)
        
        ax.set_xlabel("Sampling rate (Hz)", fontsize=12)
        ax.set_ylabel("AUC", fontsize=12)
        ax.set_title("Single-Center Performance vs Sampling Rate", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.legend(fontsize=11)
        ax.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        plt.savefig(output_dir / "sampling_invariance_auc.png", dpi=150)
        plt.close('all')
        print(f"\nSaved: {output_dir / 'sampling_invariance_auc.png'}")
    
    return results


# ----------------------------- Main Analysis -----------------------------


def run_full_analysis(ihbas_root, max_root, seg_sec, k_folds, n_boot, output_dir, rng,
                      n_jobs=1, cache_dir=None, run_sampling_invariance_flag=False, sampling_rates=None):
    print("\n" + "="*60)
    print("ICS BIOMARKER ANALYSIS (OPTIMIZED VERSION)")
    print("="*60)
    
    ihbas_records = find_all_subjects(ihbas_root, "IHBAS")
    max_records = find_all_subjects(max_root, "MAX")
    
    print(f"\nIHBAS: {len(ihbas_records)} subjects")
    print(f"MAX: {len(max_records)} subjects")
    print(f"Using {n_jobs} parallel workers")
    if cache_dir:
        print(f"Feature cache: {cache_dir}")
    
    # Single-center analyses
    ihbas_results, X_ihbas, y_ihbas, feat_names, sids_ihbas = run_center_cv(
        "IHBAS", ihbas_records, seg_sec, k_folds, n_boot, None, rng, output_dir,
        n_jobs=n_jobs, cache_dir=cache_dir
    )
    max_results, X_max, y_max, _, sids_max = run_center_cv(
        "MAX", max_records, seg_sec, k_folds, n_boot, "balanced", rng, output_dir,
        n_jobs=n_jobs, cache_dir=cache_dir
    )
    
    # Effect size analysis
    print("\n" + "="*60)
    print("EFFECT SIZE ANALYSIS")
    print("="*60)
    
    es_ihbas = compute_feature_effect_sizes(X_ihbas, y_ihbas, feat_names)
    es_max = compute_feature_effect_sizes(X_max, y_max, feat_names)
    mean_d_ihbas, mean_d_max = plot_effect_size_comparison(es_ihbas, es_max, output_dir / "effect_size_comparison.png")
    
    print(f"\nMean |Cohen's d|: IHBAS={mean_d_ihbas:.3f}, MAX={mean_d_max:.3f}")
    
    # Cross-center generalization
    print("\n" + "="*60)
    print("CROSS-CENTER GENERALIZATION (CORRECTED)")
    print("="*60)
    
    cross_results = cross_center_generalization_corrected(
        X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max, feat_names)
    
    print(f"\nCross-center generalization:")
    print(f"  Train MAX → Test IHBAS: AUC = {cross_results['train_MAX_test_IHBAS']:.3f}")
    print(f"  Train IHBAS → Test MAX: AUC = {cross_results['train_IHBAS_test_MAX']:.3f}")
    
    # Pooled analysis
    pooled_results = run_pooled_cv(X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max,
                                    feat_names, k_folds, n_boot, rng, output_dir)
    
    # Epochs per subject
    print("\n" + "="*60)
    print("EPOCHS PER SUBJECT")
    print("="*60)
    
    epochs_ihbas = defaultdict(int)
    for sid in sids_ihbas:
        epochs_ihbas[sid] += 1
    epochs_max = defaultdict(int)
    for sid in sids_max:
        epochs_max[sid] += 1
    
    print(f"IHBAS: {np.mean(list(epochs_ihbas.values())):.1f} ± {np.std(list(epochs_ihbas.values())):.1f}")
    print(f"MAX: {np.mean(list(epochs_max.values())):.1f} ± {np.std(list(epochs_max.values())):.1f}")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE SUMMARY                           │
├─────────────┬──────────────────┬──────────────┬─────────────────┤
│   Center    │       AUC        │  Sensitivity │   Specificity   │
├─────────────┼──────────────────┼──────────────┼─────────────────┤
│   IHBAS     │ {ihbas_results.auc:.3f} ({ihbas_results.ci_low:.2f}-{ihbas_results.ci_high:.2f})  │    {ihbas_results.sensitivity:.3f}     │     {ihbas_results.specificity:.3f}       │
│   MAX       │ {max_results.auc:.3f} ({max_results.ci_low:.2f}-{max_results.ci_high:.2f})  │    {max_results.sensitivity:.3f}     │     {max_results.specificity:.3f}       │
│   Pooled    │ {pooled_results['overall_auc']:.3f} ({pooled_results['overall_ci_low']:.2f}-{pooled_results['overall_ci_high']:.2f})  │      -       │       -         │
└─────────────┴──────────────────┴──────────────┴─────────────────┘

Cross-Center: MAX→IHBAS: {cross_results['train_MAX_test_IHBAS']:.3f}, IHBAS→MAX: {cross_results['train_IHBAS_test_MAX']:.3f}
Effect Sizes: IHBAS={mean_d_ihbas:.3f}, MAX={mean_d_max:.3f}

NOTE: All metrics from proper cross-validation (no data leakage).
    """)
    
    # Sampling invariance
    if run_sampling_invariance_flag:
        if sampling_rates is None:
            sampling_rates = [125, 100, 80, 60, 40, 20, 10, 5, 4, 3, 2, 1]
        run_sampling_invariance(
            ihbas_root, max_root, ihbas_records, max_records,
            seg_sec, sampling_rates, k_folds, n_boot, output_dir, rng, n_jobs, cache_dir
        )
    
    return {
        "ihbas": ihbas_results, "max": max_results, "pooled": pooled_results,
        "effect_sizes": {"ihbas": es_ihbas, "max": es_max},
        "cross_center": cross_results,
    }


# ----------------------------- Main -----------------------------


def main():
    ap = argparse.ArgumentParser(description="Complete ICS biomarker analysis (OPTIMIZED)")
    ap.add_argument("--ihbas_root", type=str, default="preprocessed_ihbas")
    ap.add_argument("--max_root", type=str, default="preprocessed_max")
    ap.add_argument("--seg_sec", type=float, default=10.0)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel workers (-1 = all CPUs)")
    ap.add_argument("--cache_dir", type=str, default="feature_cache", help="Directory for cached features")
    ap.add_argument("--output_dir", type=str, default="ics_results_final")
    ap.add_argument("--run_sampling_invariance", action="store_true")
    ap.add_argument("--sampling_rates", type=float, nargs="+", default=[125, 100, 80, 60, 40, 20, 10, 5, 4, 3, 2, 1])
    ap.add_argument("--no_cache", action="store_true", help="Disable feature caching")
    args = ap.parse_args()
    
    rng = np.random.RandomState(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    run_full_analysis(
        Path(args.ihbas_root).expanduser().resolve(),
        Path(args.max_root).expanduser().resolve(),
        args.seg_sec, args.k_folds, args.n_boot, output_dir, rng,
        args.n_jobs, cache_dir, args.run_sampling_invariance, args.sampling_rates
    )
    
    print(f"\nAll plots saved to: {output_dir}")
    if cache_dir:
        print(f"Feature cache saved to: {cache_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
