import os
import numpy as np
import scipy.stats as stats
from scipy.signal import welch
from collections import defaultdict
import pandas as pd
import random

FS = 125
CHANNELS = 19
N_SUBJECTS_PER_CENTER = 50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_subject(file_path):
    x = np.load(file_path)
    assert x.ndim == 3 and x.shape[1] == CHANNELS
    return x

def signal_stats(x):
    flat = x.reshape(-1)
    return {
        "mean": np.mean(flat),
        "std": np.std(flat),
        "rms": np.sqrt(np.mean(flat**2)),
        "ptp": np.ptp(flat),
    }

def bandpower(x):
    bp = defaultdict(list)
    for seg in x:
        for ch in seg:
            f, Pxx = welch(ch, fs=FS, nperseg=FS*2)
            for band, (lo, hi) in BANDS.items():
                mask = (f >= lo) & (f <= hi)
                bp[band].append(np.trapz(Pxx[mask], f[mask]))
    return {k: np.mean(v) for k, v in bp.items()}

# --------------------------------------------------
# Subject sampling (CRITICAL SPEEDUP)
# --------------------------------------------------

def sample_subjects(root, n_total):
    """
    Tries to balance Epileptic/Mimickers if possible
    """
    all_files = []

    for split in ["Train", "Test"]:
        for cls in ["Epileptic", "Mimickers"]:
            path = os.path.join(root, split, cls)
            if not os.path.exists(path):
                continue

            files = [
                (split, cls, os.path.join(path, f))
                for f in os.listdir(path)
                if f.endswith(".npy")
            ]
            all_files.extend(files)

    random.shuffle(all_files)

    return all_files[:n_total]

# --------------------------------------------------
# Dataset scan (FAST)
# --------------------------------------------------

def scan_dataset_fast(root, n_subjects):
    records = []

    sampled = sample_subjects(root, n_subjects)

    print(f"[INFO] {os.path.basename(root)}: using {len(sampled)} subjects")

    for split, cls, fpath in sampled:
        x = load_subject(fpath)

        rec = {
            "center": os.path.basename(root),
            "split": split,
            "class": cls,
            "subject": os.path.basename(fpath),
            "n_segments": x.shape[0],
        }

        rec.update(signal_stats(x))
        rec.update(bandpower(x))

        records.append(rec)

    return pd.DataFrame(records)

# --------------------------------------------------
# Statistics
# --------------------------------------------------

def compare_centers(df, feature):
    max_vals = df[df.center == "preprocessed_max"][feature]
    ihb_vals = df[df.center == "preprocessed_ihbas"][feature]

    ks = stats.ks_2samp(max_vals, ihb_vals)
    d = (max_vals.mean() - ihb_vals.mean()) / np.sqrt(
        (max_vals.var() + ihb_vals.var()) / 2
    )

    return {
        "feature": feature,
        "max_mean": max_vals.mean(),
        "ihbas_mean": ihb_vals.mean(),
        "ks_stat": ks.statistic,
        "ks_p": ks.pvalue,
        "cohens_d": d
    }

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    df_max = scan_dataset_fast("preprocessed_max", N_SUBJECTS_PER_CENTER)
    df_ihb = scan_dataset_fast("preprocessed_ihbas", N_SUBJECTS_PER_CENTER)

    df = pd.concat([df_max, df_ihb], ignore_index=True)

    print("\n===== SUBJECT COUNTS =====")
    print(df.groupby(["center", "class"]).size())

    print("\n===== SEGMENTS PER SUBJECT =====")
    print(df.groupby("center")["n_segments"].describe())

    features = ["mean", "std", "rms", "ptp"] + list(BANDS.keys())

    results = [compare_centers(df, f) for f in features]
    res_df = pd.DataFrame(results)

    print("\n===== CENTER SHIFT SUMMARY =====")
    print(res_df.sort_values("ks_stat", ascending=False))

    res_df.to_csv("center_statistical_comparison_fast.csv", index=False)
