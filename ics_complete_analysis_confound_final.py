#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete ICS-based epilepsy vs mimicker biomarker analysis - RECORDING LENGTH CONTROLS.

This version adds critical controls for the recording-length confound:
1. Option to EXCLUDE log_n_epochs from subject features
2. Fixed-duration truncation analysis (10/20/30 min)
3. Fixed number of epochs per subject analysis
4. Comprehensive within-center and cross-center reporting for each condition

The goal is to test whether biomarker performance holds when recording length
is NOT available as a predictor - addressing reviewer concerns about detecting
"clinical workflow" vs. "epilepsy".

Usage:
    python ics_complete_analysis_corrected.py \
        --ihbas_root preprocessed_ihbas \
        --max_root preprocessed_max \
        --seg_sec 10 \
        --k_folds 5 \
        --n_boot 1000 \
        --run_length_controls
"""

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import argparse
import math
import warnings
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

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

# ----------------------------- Constants -----------------------------

COMMON_ORDER = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]
CH_INDEX = {name: idx for idx, name in enumerate(COMMON_ORDER)}

FEATURE_GROUPS = {
    "spectral_slowing": ["global_slowing_index", "frontal_slowing_index", 
                         "temporal_slowing_index", "occipital_slowing_index"],
    "band_powers": ["rel_delta_power", "rel_theta_power", "rel_alpha_power", "rel_beta_power"],
    "pdr": ["occipital_pdr_freq", "occipital_alpha_asymmetry"],
    "complexity": ["pe_mean"],
    "connectivity": ["network_mean_corr", "network_strength_std"],
}


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
        else:
            if by_id[sid] != r.label:
                by_id[sid] = -1
                if sid in chosen:
                    del chosen[sid]
    
    return [rec for sid, rec in chosen.items() if by_id[sid] in (0, 1)]


def get_channel_index(name: str, n_channels: int) -> int:
    idx = CH_INDEX.get(name, None)
    return idx if idx is not None and idx < n_channels else -1


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


# ----------------------------- Epoch Feature Building -----------------------------


def build_epoch_features(records: List[SubjectRecord], seg_sec: float, 
                         target_fs: Optional[float] = None,
                         max_epochs_per_subject: Optional[int] = None,
                         max_duration_min: Optional[float] = None):
    """
    Build epoch-level features with optional truncation controls.
    
    Parameters:
    -----------
    records : list of SubjectRecord
    seg_sec : float
        Duration of each epoch in seconds
    target_fs : float, optional
        Target sampling frequency for resampling
    max_epochs_per_subject : int, optional
        Maximum number of epochs to use per subject (truncation control)
    max_duration_min : float, optional
        Maximum recording duration in minutes (truncation control)
        Takes precedence over max_epochs_per_subject if both specified
    """
    all_feats, labels, subj_ids = [], [], []
    truncation_stats = {"original_epochs": 0, "used_epochs": 0, "subjects_truncated": 0}
    
    # Convert max_duration_min to max_epochs
    if max_duration_min is not None:
        max_epochs_from_duration = int(max_duration_min * 60 / seg_sec)
        if max_epochs_per_subject is None or max_epochs_from_duration < max_epochs_per_subject:
            max_epochs_per_subject = max_epochs_from_duration
    
    for rec in records:
        try:
            arr = np.load(rec.path)
        except Exception:
            continue
            
        if arr.ndim != 3 or arr.shape[0] == 0:
            continue
        
        N_seg, C, T = arr.shape
        orig_fs = float(T) / float(seg_sec)
        
        truncation_stats["original_epochs"] += N_seg
        
        # Apply truncation if specified
        if max_epochs_per_subject is not None and N_seg > max_epochs_per_subject:
            # Use FIRST N epochs (clinical standard: beginning of recording most relevant)
            N_seg_use = max_epochs_per_subject
            truncation_stats["subjects_truncated"] += 1
        else:
            N_seg_use = N_seg
        
        truncation_stats["used_epochs"] += N_seg_use
        
        for seg_idx in range(N_seg_use):
            epoch = arr[seg_idx]
            if target_fs is not None and not math.isclose(orig_fs, target_fs, rel_tol=1e-6):
                epoch = resample_epoch(epoch, orig_fs, target_fs, seg_sec)
            
            feats = compute_epoch_ics(epoch, seg_sec)
            all_feats.append(feats)
            labels.append(rec.label)
            subj_ids.append(rec.subject_id)
    
    if not all_feats:
        raise RuntimeError("No epoch features computed; check data paths.")
    
    feature_names = sorted(all_feats[0].keys())
    X = np.zeros((len(all_feats), len(feature_names)), dtype=np.float32)
    for i, fdict in enumerate(all_feats):
        for j, name in enumerate(feature_names):
            X[i, j] = float(fdict.get(name, 0.0))
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(labels, dtype=np.int64)
    
    return X, y, feature_names, subj_ids, truncation_stats


def build_epoch_features_multicenter(records: List[SubjectRecord], seg_sec: float, 
                                      target_fs: Optional[float] = None,
                                      max_epochs_per_subject: Optional[int] = None,
                                      max_duration_min: Optional[float] = None):
    """Multi-center version with truncation controls."""
    all_feats, labels, subj_ids, centers = [], [], [], []
    truncation_stats = {"original_epochs": 0, "used_epochs": 0, "subjects_truncated": 0}
    
    if max_duration_min is not None:
        max_epochs_from_duration = int(max_duration_min * 60 / seg_sec)
        if max_epochs_per_subject is None or max_epochs_from_duration < max_epochs_per_subject:
            max_epochs_per_subject = max_epochs_from_duration
    
    for rec in records:
        try:
            arr = np.load(rec.path)
        except Exception:
            continue
            
        if arr.ndim != 3 or arr.shape[0] == 0:
            continue
        
        N_seg, C, T = arr.shape
        orig_fs = float(T) / float(seg_sec)
        subj_prefix = f"{rec.center}_{rec.subject_id}"
        center_idx = 1 if rec.center == "MAX" else 0
        
        truncation_stats["original_epochs"] += N_seg
        
        if max_epochs_per_subject is not None and N_seg > max_epochs_per_subject:
            N_seg_use = max_epochs_per_subject
            truncation_stats["subjects_truncated"] += 1
        else:
            N_seg_use = N_seg
        
        truncation_stats["used_epochs"] += N_seg_use
        
        for seg_idx in range(N_seg_use):
            epoch = arr[seg_idx]
            if target_fs is not None and not math.isclose(orig_fs, target_fs, rel_tol=1e-6):
                epoch = resample_epoch(epoch, orig_fs, target_fs, seg_sec)
            
            feats = compute_epoch_ics(epoch, seg_sec)
            all_feats.append(feats)
            labels.append(rec.label)
            subj_ids.append(subj_prefix)
            centers.append(center_idx)
    
    if not all_feats:
        raise RuntimeError("No epoch features computed.")
    
    feature_names = sorted(all_feats[0].keys())
    X = np.zeros((len(all_feats), len(feature_names)), dtype=np.float32)
    for i, fdict in enumerate(all_feats):
        for j, name in enumerate(feature_names):
            X[i, j] = float(fdict.get(name, 0.0))
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(labels, dtype=np.int64)
    centers_arr = np.asarray(centers, dtype=np.int64)
    
    return X, y, feature_names, subj_ids, centers_arr, truncation_stats


# ----------------------------- Subject-Level Features -----------------------------


def build_subject_features(X_epoch, y_epoch, subj_ids_epoch, proba_epoch, feature_names, 
                           include_extended_tails=False,
                           include_recording_length=True):  # NEW PARAMETER
    """
    Build subject-level features from epoch-level features.
    
    Parameters:
    -----------
    include_recording_length : bool
        If False, EXCLUDES log_n_epochs from features. This is critical for
        addressing the recording-length confound concern.
    """
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
        p_mean = p.mean() if p.size > 0 else 0.0
        p_std = p.std() if p.size > 0 else 0.0
        p_q90 = np.percentile(p, 90.0) if p.size > 0 else 0.0
        p_q95 = np.percentile(p, 95.0) if p.size > 0 else 0.0
        k_top = max(1, int(round(0.2 * p.size))) if p.size > 0 else 1
        p_top20 = np.sort(p)[-k_top:].mean() if p.size > 0 else 0.0
        
        feats.extend([p_mean, p_std, p_q90, p_q95, p_top20])
        
        if include_extended_tails and p.size > 0:
            eps = 1e-6
            z = np.log(p + eps) - np.log(1.0 - p + eps)
            z_mean, z_max = float(z.mean()), float(z.max())
            z_top20_mean = float(np.sort(z)[-k_top:].mean())
            frac_gt_08 = float((p >= 0.8).mean())
            frac_gt_09 = float((p >= 0.9).mean())
            frac_gt_095 = float((p >= 0.95).mean())
            llr_sum_scaled = float(z.sum() / math.sqrt(p.size))
            feats.extend([z_mean, z_max, z_top20_mean, frac_gt_08, frac_gt_09, frac_gt_095, llr_sum_scaled])
        
        # CRITICAL: Only add recording length if explicitly requested
        if include_recording_length:
            feats.append(math.log(len(idxs) + 1.0))
        
        subj_feature_list.append(np.asarray(feats, dtype=np.float32))
    
    X_subj = np.stack(subj_feature_list, axis=0)
    y_subj = np.asarray(subj_labels, dtype=np.int64)
    
    subj_feature_names = [f"mean_{n}" for n in feature_names] + [f"std_{n}" for n in feature_names]
    subj_feature_names.extend(["p_mean", "p_std", "p_q90", "p_q95", "p_mean_top20"])
    
    if include_extended_tails:
        subj_feature_names.extend([
            "logit_mean", "logit_max", "logit_top20_mean",
            "frac_p_gt_0_8", "frac_p_gt_0_9", "frac_p_gt_0_95", "llr_sum_scaled"
        ])
    
    if include_recording_length:
        subj_feature_names.append("log_n_epochs")
    
    return np.nan_to_num(X_subj, nan=0.0, posinf=0.0, neginf=0.0), y_subj, subj_feature_names, subj_ids_unique


# ----------------------------- Model Utilities -----------------------------


def get_logistic_regression(class_weight=None, random_state=0):
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight=class_weight, random_state=random_state)


# ----------------------------- Effect Size Analysis -----------------------------


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * group1.var(ddof=1) + (n2 - 1) * group2.var(ddof=1)) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 1e-10 else 0.0


def compute_feature_effect_sizes(X, y, feature_names):
    return {name: cohens_d(X[y == 1, i], X[y == 0, i]) for i, name in enumerate(feature_names)}


# ----------------------------- Feature Importance -----------------------------


def compute_feature_importance(clf, scaler, feature_names):
    coef = clf.coef_[0]
    scaled_coef = coef * scaler.scale_ if hasattr(scaler, 'scale_') and scaler.scale_ is not None else coef
    return {name: float(scaled_coef[i]) for i, name in enumerate(feature_names)}


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


# ----------------------------- Plotting Utilities -----------------------------


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


# ----------------------------- CV & Metrics -----------------------------


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


# ----------------------------- Single Center CV (UPDATED) -----------------------------


def run_center_cv(center_name, records, seg_sec, k_folds, n_boot, epoch_class_weight, rng, output_dir,
                  include_recording_length=True, max_epochs_per_subject=None, max_duration_min=None,
                  suffix=""):
    """
    Run single-center CV with optional recording-length controls.
    
    Parameters:
    -----------
    include_recording_length : bool
        Whether to include log_n_epochs as a feature
    max_epochs_per_subject : int, optional
        Truncate to this many epochs per subject
    max_duration_min : float, optional
        Truncate to this duration in minutes
    suffix : str
        Suffix for output filenames
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if output_dir is None:
        output_dir = Path(".")
    
    label = f"{center_name}"
    if not include_recording_length:
        label += " (no rec. length)"
    if max_duration_min:
        label += f" ({max_duration_min}min)"
    elif max_epochs_per_subject:
        label += f" ({max_epochs_per_subject}ep)"
    
    print(f"\n{'='*60}")
    print(f"Running {label} analysis")
    print(f"{'='*60}")
    
    print("Computing epoch-level ICS features...")
    X_ics, y_epoch, feature_names, subj_ids_epoch, trunc_stats = build_epoch_features(
        records, seg_sec, max_epochs_per_subject=max_epochs_per_subject, 
        max_duration_min=max_duration_min)
    
    print(f"Epoch features: {X_ics.shape[0]} epochs, {X_ics.shape[1]} features")
    if trunc_stats["subjects_truncated"] > 0:
        print(f"  Truncation: {trunc_stats['subjects_truncated']} subjects truncated")
        print(f"  Original epochs: {trunc_stats['original_epochs']}, Used: {trunc_stats['used_epochs']}")
    
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
            X_ics[train_mask], y_epoch[train_mask], 
            [sid for sid, m in zip(subj_ids_epoch, train_mask) if m], 
            p_train, feature_names,
            include_recording_length=include_recording_length)
        
        X_subj_val, _, _, sids_val = build_subject_features(
            X_ics[val_mask], y_epoch[val_mask], 
            [sid for sid, m in zip(subj_ids_epoch, val_mask) if m], 
            p_val, feature_names,
            include_recording_length=include_recording_length)
        
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
    
    print(f"\n{label} Results:")
    print(f"  AUC: {metrics['auc']:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"  Specificity: {metrics['specificity']:.3f}")
    
    fname_suffix = f"_{suffix}" if suffix else ""
    plot_roc(y_subj_true, cv_preds, f"{label} ROC (CV)", output_dir / f"{center_name}{fname_suffix}_roc.png")
    plot_score_distributions(y_subj_true, cv_preds, f"{label} Scores", output_dir / f"{center_name}{fname_suffix}_scores.png")
    plot_feature_importance(avg_importance, f"{label} Feature Importance", output_dir / f"{center_name}{fname_suffix}_feature_importance.png")
    
    return AnalysisResults(
        auc=metrics["auc"], ci_low=ci_low, ci_high=ci_high, accuracy=metrics["accuracy"],
        sensitivity=metrics["sensitivity"], specificity=metrics["specificity"], brier_score=metrics["brier_score"],
        u_stat=metrics["u_stat"], p_val=metrics["p_val"], y_true=y_subj_true, y_score=cv_preds,
        subject_ids=subject_ids, feature_names=subj_feat_names, feature_importance=avg_importance,
    ), X_ics, y_epoch, feature_names, subj_ids_epoch


# ----------------------------- Cross-Center Generalization (UPDATED) -----------------------------


def cross_center_generalization_corrected(X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max, feat_names,
                                          include_recording_length=True):
    """
    Cross-center generalization with optional recording-length control.
    """
    results = {}
    
    label = "corrected"
    if not include_recording_length:
        label += " (no rec. length)"
    
    print(f"\n  Computing cross-center generalization ({label})...")
    
    # =========================================================================
    # Train on MAX, test on IHBAS
    # =========================================================================
    
    scaler_e_max = StandardScaler()
    X_max_scaled = scaler_e_max.fit_transform(X_max)
    clf_e_max = get_logistic_regression(class_weight="balanced", random_state=0)
    clf_e_max.fit(X_max_scaled, y_max)
    
    X_ihbas_scaled_by_max = scaler_e_max.transform(X_ihbas)
    p_ihbas_from_max = clf_e_max.predict_proba(X_ihbas_scaled_by_max)[:, 1]
    
    p_max_train = clf_e_max.predict_proba(X_max_scaled)[:, 1]
    X_subj_max_train, y_subj_max_train, _, _ = build_subject_features(
        X_max, y_max, sids_max, p_max_train, feat_names,
        include_recording_length=include_recording_length)
    
    X_subj_ihbas_test, y_subj_ihbas_test, _, _ = build_subject_features(
        X_ihbas, y_ihbas, sids_ihbas, p_ihbas_from_max, feat_names,
        include_recording_length=include_recording_length)
    
    scaler_s = StandardScaler()
    X_subj_max_scaled = scaler_s.fit_transform(X_subj_max_train)
    X_subj_ihbas_scaled = scaler_s.transform(X_subj_ihbas_test)
    
    clf_s = get_logistic_regression(random_state=0)
    clf_s.fit(X_subj_max_scaled, y_subj_max_train)
    p_ihbas_final = clf_s.predict_proba(X_subj_ihbas_scaled)[:, 1]
    
    try:
        results["train_MAX_test_IHBAS"] = roc_auc_score(y_subj_ihbas_test, p_ihbas_final)
    except:
        results["train_MAX_test_IHBAS"] = float("nan")
    
    # =========================================================================
    # Train on IHBAS, test on MAX
    # =========================================================================
    
    scaler_e_ihbas = StandardScaler()
    X_ihbas_scaled = scaler_e_ihbas.fit_transform(X_ihbas)
    clf_e_ihbas = get_logistic_regression(random_state=0)
    clf_e_ihbas.fit(X_ihbas_scaled, y_ihbas)
    
    X_max_scaled_by_ihbas = scaler_e_ihbas.transform(X_max)
    p_max_from_ihbas = clf_e_ihbas.predict_proba(X_max_scaled_by_ihbas)[:, 1]
    
    p_ihbas_train = clf_e_ihbas.predict_proba(X_ihbas_scaled)[:, 1]
    X_subj_ihbas_train, y_subj_ihbas_train, _, _ = build_subject_features(
        X_ihbas, y_ihbas, sids_ihbas, p_ihbas_train, feat_names,
        include_recording_length=include_recording_length)
    
    X_subj_max_test, y_subj_max_test, _, _ = build_subject_features(
        X_max, y_max, sids_max, p_max_from_ihbas, feat_names,
        include_recording_length=include_recording_length)
    
    scaler_s2 = StandardScaler()
    X_subj_ihbas_scaled2 = scaler_s2.fit_transform(X_subj_ihbas_train)
    X_subj_max_scaled2 = scaler_s2.transform(X_subj_max_test)
    
    clf_s2 = get_logistic_regression(random_state=0)
    clf_s2.fit(X_subj_ihbas_scaled2, y_subj_ihbas_train)
    p_max_final = clf_s2.predict_proba(X_subj_max_scaled2)[:, 1]
    
    try:
        results["train_IHBAS_test_MAX"] = roc_auc_score(y_subj_max_test, p_max_final)
    except:
        results["train_IHBAS_test_MAX"] = float("nan")
    
    return results


# ----------------------------- Pooled Multi-Center CV (UPDATED) -----------------------------


def run_pooled_cv(X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max, feat_names, k_folds, n_boot, rng, output_dir,
                  include_recording_length=True, suffix=""):
    """Run pooled multi-center CV with recording-length control."""
    
    label = "POOLED MULTI-CENTER MODEL"
    if not include_recording_length:
        label += " (no rec. length)"
    
    print("\n" + "="*60)
    print(label)
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
    
    print(f"Combined: {n_subjects} subjects")
    
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
            X_all[train_mask], y_all[train_mask], 
            [sid for sid, m in zip(sids_all, train_mask) if m], 
            p_train, feat_names,
            include_recording_length=include_recording_length)
        
        X_subj_val, _, _, sids_val = build_subject_features(
            X_all[val_mask], y_all[val_mask], 
            [sid for sid, m in zip(sids_all, val_mask) if m], 
            p_val, feat_names,
            include_recording_length=include_recording_length)
        
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
    
    try:
        ihbas_auc = roc_auc_score(y_subj_true[ihbas_mask], cv_preds[ihbas_mask])
        max_auc = roc_auc_score(y_subj_true[max_mask], cv_preds[max_mask])
    except:
        ihbas_auc = max_auc = float("nan")
    
    print(f"  IHBAS-only AUC: {ihbas_auc:.3f}")
    print(f"  MAX-only AUC: {max_auc:.3f}")
    
    fname_suffix = f"_{suffix}" if suffix else ""
    plot_roc(y_subj_true, cv_preds, f"Pooled ROC (CV){' - no rec. length' if not include_recording_length else ''}", 
             output_dir / f"pooled{fname_suffix}_roc.png")
    
    return {
        "overall_auc": metrics["auc"], "overall_ci_low": ci_low, "overall_ci_high": ci_high,
        "ihbas_auc": ihbas_auc, "max_auc": max_auc,
    }


# ===========================================================================================
# NEW: RECORDING LENGTH CONTROL ANALYSES
# ===========================================================================================


def run_recording_length_analysis(ihbas_root, max_root, seg_sec, k_folds, n_boot, output_dir, rng):
    """
    Comprehensive analysis of recording-length effects.
    
    This function addresses the key reviewer concern by:
    1. Re-running all analyses WITHOUT recording length as a feature
    2. Testing fixed-duration truncation (10, 20, 30 min)
    3. Testing fixed number of epochs per subject
    4. Comparing within-center and cross-center performance across conditions
    """
    print("\n" + "="*80)
    print("RECORDING LENGTH CONTROL ANALYSES")
    print("Addressing reviewer concern: 'Are we detecting epilepsy, or clinical workflow?'")
    print("="*80)
    
    # Load records
    ihbas_records = find_all_subjects(ihbas_root, "IHBAS")
    max_records = find_all_subjects(max_root, "MAX")
    
    results_summary = {
        "with_recording_length": {},
        "without_recording_length": {},
        "fixed_duration_10min": {},
        "fixed_duration_20min": {},
        "fixed_duration_30min": {},
    }
    
    # =========================================================================
    # 1. BASELINE: With recording length (original analysis)
    # =========================================================================
    print("\n" + "-"*60)
    print("CONDITION 1: WITH recording length (baseline)")
    print("-"*60)
    
    ihbas_res, X_ihbas, y_ihbas, feat_names, sids_ihbas = run_center_cv(
        "IHBAS", ihbas_records, seg_sec, k_folds, n_boot, None, rng, output_dir,
        include_recording_length=True, suffix="with_reclength")
    
    max_res, X_max, y_max, _, sids_max = run_center_cv(
        "MAX", max_records, seg_sec, k_folds, n_boot, "balanced", rng, output_dir,
        include_recording_length=True, suffix="with_reclength")
    
    cross_res = cross_center_generalization_corrected(
        X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max, feat_names,
        include_recording_length=True)
    
    results_summary["with_recording_length"] = {
        "IHBAS_AUC": ihbas_res.auc,
        "MAX_AUC": max_res.auc,
        "MAX→IHBAS": cross_res["train_MAX_test_IHBAS"],
        "IHBAS→MAX": cross_res["train_IHBAS_test_MAX"],
    }
    
    print(f"\n  WITH recording length:")
    print(f"    IHBAS within-center AUC: {ihbas_res.auc:.3f}")
    print(f"    MAX within-center AUC: {max_res.auc:.3f}")
    print(f"    MAX→IHBAS cross-center AUC: {cross_res['train_MAX_test_IHBAS']:.3f}")
    print(f"    IHBAS→MAX cross-center AUC: {cross_res['train_IHBAS_test_MAX']:.3f}")
    
    # =========================================================================
    # 2. WITHOUT recording length as a feature
    # =========================================================================
    print("\n" + "-"*60)
    print("CONDITION 2: WITHOUT recording length (critical test)")
    print("-"*60)
    
    ihbas_res_nrl, X_ihbas_nrl, y_ihbas_nrl, feat_names_nrl, sids_ihbas_nrl = run_center_cv(
        "IHBAS", ihbas_records, seg_sec, k_folds, n_boot, None, rng, output_dir,
        include_recording_length=False, suffix="no_reclength")
    
    max_res_nrl, X_max_nrl, y_max_nrl, _, sids_max_nrl = run_center_cv(
        "MAX", max_records, seg_sec, k_folds, n_boot, "balanced", rng, output_dir,
        include_recording_length=False, suffix="no_reclength")
    
    cross_res_nrl = cross_center_generalization_corrected(
        X_ihbas_nrl, y_ihbas_nrl, sids_ihbas_nrl, X_max_nrl, y_max_nrl, sids_max_nrl, feat_names_nrl,
        include_recording_length=False)
    
    results_summary["without_recording_length"] = {
        "IHBAS_AUC": ihbas_res_nrl.auc,
        "MAX_AUC": max_res_nrl.auc,
        "MAX→IHBAS": cross_res_nrl["train_MAX_test_IHBAS"],
        "IHBAS→MAX": cross_res_nrl["train_IHBAS_test_MAX"],
    }
    
    print(f"\n  WITHOUT recording length:")
    print(f"    IHBAS within-center AUC: {ihbas_res_nrl.auc:.3f}")
    print(f"    MAX within-center AUC: {max_res_nrl.auc:.3f}")
    print(f"    MAX→IHBAS cross-center AUC: {cross_res_nrl['train_MAX_test_IHBAS']:.3f}")
    print(f"    IHBAS→MAX cross-center AUC: {cross_res_nrl['train_IHBAS_test_MAX']:.3f}")
    
    # =========================================================================
    # 3. Fixed duration truncation analyses
    # =========================================================================
    for duration_min in [10, 20, 30]:
        print(f"\n" + "-"*60)
        print(f"CONDITION: Fixed {duration_min}-minute truncation (WITHOUT recording length)")
        print("-"*60)
        
        ihbas_res_trunc, X_ihbas_trunc, y_ihbas_trunc, feat_names_trunc, sids_ihbas_trunc = run_center_cv(
            "IHBAS", ihbas_records, seg_sec, k_folds, n_boot, None, rng, output_dir,
            include_recording_length=False,  # Always exclude when truncating
            max_duration_min=duration_min,
            suffix=f"trunc_{duration_min}min")
        
        max_res_trunc, X_max_trunc, y_max_trunc, _, sids_max_trunc = run_center_cv(
            "MAX", max_records, seg_sec, k_folds, n_boot, "balanced", rng, output_dir,
            include_recording_length=False,
            max_duration_min=duration_min,
            suffix=f"trunc_{duration_min}min")
        
        cross_res_trunc = cross_center_generalization_corrected(
            X_ihbas_trunc, y_ihbas_trunc, sids_ihbas_trunc, 
            X_max_trunc, y_max_trunc, sids_max_trunc, feat_names_trunc,
            include_recording_length=False)
        
        key = f"fixed_duration_{duration_min}min"
        results_summary[key] = {
            "IHBAS_AUC": ihbas_res_trunc.auc,
            "MAX_AUC": max_res_trunc.auc,
            "MAX→IHBAS": cross_res_trunc["train_MAX_test_IHBAS"],
            "IHBAS→MAX": cross_res_trunc["train_IHBAS_test_MAX"],
        }
        
        print(f"\n  {duration_min}-minute truncation:")
        print(f"    IHBAS within-center AUC: {ihbas_res_trunc.auc:.3f}")
        print(f"    MAX within-center AUC: {max_res_trunc.auc:.3f}")
        print(f"    MAX→IHBAS cross-center AUC: {cross_res_trunc['train_MAX_test_IHBAS']:.3f}")
        print(f"    IHBAS→MAX cross-center AUC: {cross_res_trunc['train_IHBAS_test_MAX']:.3f}")
    
    # =========================================================================
    # Summary comparison plot
    # =========================================================================
    create_recording_length_comparison_plot(results_summary, output_dir)
    
    # =========================================================================
    # Print comprehensive summary
    # =========================================================================
    print("\n" + "="*80)
    print("RECORDING LENGTH CONTROL: SUMMARY")
    print("="*80)
    
    print("\n┌─────────────────────────────────┬────────────┬───────────┬──────────────┬──────────────┐")
    print("│           Condition             │ IHBAS AUC  │  MAX AUC  │ MAX→IHBAS    │ IHBAS→MAX    │")
    print("├─────────────────────────────────┼────────────┼───────────┼──────────────┼──────────────┤")
    
    for cond, res in results_summary.items():
        if res:
            cond_name = cond.replace("_", " ").title()[:31]
            print(f"│ {cond_name:<31} │   {res['IHBAS_AUC']:6.3f}   │  {res['MAX_AUC']:6.3f}  │    {res['MAX→IHBAS']:6.3f}     │    {res['IHBAS→MAX']:6.3f}     │")
    
    print("└─────────────────────────────────┴────────────┴───────────┴──────────────┴──────────────┘")
    
    # Key interpretation
    print("\n" + "-"*60)
    print("KEY INTERPRETATION:")
    print("-"*60)
    
    auc_with = results_summary["with_recording_length"]["IHBAS_AUC"]
    auc_without = results_summary["without_recording_length"]["IHBAS_AUC"]
    cross_with = results_summary["with_recording_length"]["IHBAS→MAX"]
    cross_without = results_summary["without_recording_length"]["IHBAS→MAX"]
    
    delta_within = auc_with - auc_without
    delta_cross = cross_without - cross_with
    
    print(f"\n  Within-center AUC change (IHBAS): {delta_within:+.3f}")
    print(f"  Cross-center AUC change (IHBAS→MAX): {delta_cross:+.3f}")
    
    if delta_within < 0.05 and auc_without > 0.70:
        print("\n  ✓ GOOD: Removing recording length has minimal impact on within-center AUC")
        print("    This suggests the biomarker captures genuine neurophysiological differences.")
    else:
        print("\n  ⚠ CAUTION: Significant AUC drop when removing recording length")
        print("    Recording length may be contributing substantially to discrimination.")
    
    if cross_without > cross_with:
        print(f"\n  ✓ GOOD: Cross-center generalization IMPROVED without recording length")
        print("    This strongly suggests recording length was a center-specific confound.")
    elif cross_without > 0.5:
        print(f"\n  ✓ Reasonable: Cross-center AUC > 0.5 without recording length")
        print("    Model maintains directional predictions across centers.")
    else:
        print(f"\n  ⚠ CONCERN: Cross-center AUC still inverted even without recording length")
        print("    Other center-specific confounds may be present.")
    
    return results_summary


def create_recording_length_comparison_plot(results_summary, output_dir):
    """Create a visual comparison of results across recording-length conditions."""
    
    conditions = list(results_summary.keys())
    metrics = ["IHBAS_AUC", "MAX_AUC", "MAX→IHBAS", "IHBAS→MAX"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Within-center comparison
    ax1 = axes[0]
    x = np.arange(len(conditions))
    width = 0.35
    
    ihbas_vals = [results_summary[c].get("IHBAS_AUC", 0) for c in conditions]
    max_vals = [results_summary[c].get("MAX_AUC", 0) for c in conditions]
    
    bars1 = ax1.bar(x - width/2, ihbas_vals, width, label='IHBAS', color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, max_vals, width, label='MAX', color='#d62728', alpha=0.8)
    
    ax1.set_ylabel('AUC')
    ax1.set_title('Within-Center Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n").replace("with recording length", "with rec.\nlength").replace("without recording length", "without rec.\nlength")[:20] for c in conditions], fontsize=8)
    ax1.legend()
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Target AUC')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Cross-center comparison
    ax2 = axes[1]
    
    max_ihbas_vals = [results_summary[c].get("MAX→IHBAS", 0) for c in conditions]
    ihbas_max_vals = [results_summary[c].get("IHBAS→MAX", 0) for c in conditions]
    
    bars3 = ax2.bar(x - width/2, max_ihbas_vals, width, label='MAX→IHBAS', color='#2ca02c', alpha=0.8)
    bars4 = ax2.bar(x + width/2, ihbas_max_vals, width, label='IHBAS→MAX', color='#ff7f0e', alpha=0.8)
    
    ax2.set_ylabel('AUC')
    ax2.set_title('Cross-Center Generalization')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n").replace("with recording length", "with rec.\nlength").replace("without recording length", "without rec.\nlength")[:20] for c in conditions], fontsize=8)
    ax2.legend()
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "recording_length_control_comparison.png", dpi=150)
    plt.close('all')
    print(f"\nSaved: {output_dir / 'recording_length_control_comparison.png'}")


# ----------------------------- Main Analysis -----------------------------


def run_full_analysis(ihbas_root, max_root, seg_sec, k_folds, n_boot, output_dir, rng,
                      run_length_controls=False):
    print("\n" + "="*60)
    print("ICS BIOMARKER ANALYSIS (WITH RECORDING LENGTH CONTROLS)")
    print("="*60)
    
    ihbas_records = find_all_subjects(ihbas_root, "IHBAS")
    max_records = find_all_subjects(max_root, "MAX")
    
    print(f"\nIHBAS: {len(ihbas_records)} subjects")
    print(f"MAX: {len(max_records)} subjects")
    
    # Recording length statistics
    print("\n" + "-"*60)
    print("RECORDING LENGTH STATISTICS")
    print("-"*60)
    
    for name, records in [("IHBAS", ihbas_records), ("MAX", max_records)]:
        epochs_epi, epochs_mim = [], []
        for rec in records:
            try:
                arr = np.load(rec.path)
                n_epochs = arr.shape[0]
                if rec.label == 1:
                    epochs_epi.append(n_epochs)
                else:
                    epochs_mim.append(n_epochs)
            except:
                continue
        
        print(f"\n  {name}:")
        print(f"    Epilepsy: {np.mean(epochs_epi):.1f} ± {np.std(epochs_epi):.1f} epochs (n={len(epochs_epi)})")
        print(f"    Mimickers: {np.mean(epochs_mim):.1f} ± {np.std(epochs_mim):.1f} epochs (n={len(epochs_mim)})")
        print(f"    Recording duration (approx): Epi={np.mean(epochs_epi)*seg_sec/60:.1f}min, Mim={np.mean(epochs_mim)*seg_sec/60:.1f}min")
    
    # Run recording length control analyses
    if run_length_controls:
        results = run_recording_length_analysis(
            ihbas_root, max_root, seg_sec, k_folds, n_boot, output_dir, rng)
        return results
    else:
        # Standard analysis without controls
        ihbas_results, X_ihbas, y_ihbas, feat_names, sids_ihbas = run_center_cv(
            "IHBAS", ihbas_records, seg_sec, k_folds, n_boot, None, rng, output_dir)
        max_results, X_max, y_max, _, sids_max = run_center_cv(
            "MAX", max_records, seg_sec, k_folds, n_boot, "balanced", rng, output_dir)
        
        cross_results = cross_center_generalization_corrected(
            X_ihbas, y_ihbas, sids_ihbas, X_max, y_max, sids_max, feat_names)
        
        print(f"\nCross-center generalization:")
        print(f"  Train MAX → Test IHBAS: AUC = {cross_results['train_MAX_test_IHBAS']:.3f}")
        print(f"  Train IHBAS → Test MAX: AUC = {cross_results['train_IHBAS_test_MAX']:.3f}")
        
        return {
            "ihbas": ihbas_results, "max": max_results,
            "cross_center": cross_results,
        }


# ----------------------------- Main -----------------------------


def main():
    ap = argparse.ArgumentParser(description="ICS biomarker analysis with recording length controls")
    ap.add_argument("--ihbas_root", type=str, default="preprocessed_ihbas")
    ap.add_argument("--max_root", type=str, default="preprocessed_max")
    ap.add_argument("--seg_sec", type=float, default=10.0)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=str, default="ics_results_length_controls")
    ap.add_argument("--run_length_controls", action="store_true",
                    help="Run comprehensive recording length control analyses")
    args = ap.parse_args()
    
    rng = np.random.RandomState(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_full_analysis(
        Path(args.ihbas_root).expanduser().resolve(),
        Path(args.max_root).expanduser().resolve(),
        args.seg_sec, args.k_folds, args.n_boot, output_dir, rng,
        run_length_controls=args.run_length_controls
    )
    
    print(f"\nAll results saved to: {output_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
