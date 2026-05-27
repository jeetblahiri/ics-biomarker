#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess IHBAS EDF files for Epilepsy vs Mimickers analysis.

Input directory structure (as given):

    data_all/
        data_ihbas/
            Train/
                Epileptic/*.edf
                Mimickers/*.edf
            Test/
                Epileptic/*.edf
                Mimickers/*.edf

Output directory structure:

    preprocessed_ihbas/
        Train/
            Epileptic/*.npy
            Mimickers/*.npy
        Test/
            Epileptic/*.npy
            Mimickers/*.npy

Each .npy file will be a float32 array of shape:

    (N_segments, N_channels, T)

where:
    - N_segments: number of non-overlapping 10 s windows
    - N_channels: number of retained EEG channels
    - T: samples per segment (seg_sec * target_sfreq)

Preprocessing steps:
    1. Harmonize channel names (legacy 10–20 labels, suffix stripping).
    2. Drop non-EEG / polygraph channels (Snore, Flow, DC, PPG, etc.).
    3. Pick a fixed set/order of EEG channels if available.
    4. Set montage to standard_1020 (best-effort).
    5. Set average reference.
    6. Notch filter at 50 Hz.
    7. Band-pass 0.5–40 Hz.
    8. Resample to target sampling frequency (default 256 Hz).
    9. Segment into non-overlapping windows of seg_sec seconds.
   10. Save as .npy.
"""

import argparse
from pathlib import Path
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import mne


# -------------------------- Channel name normalization --------------------------


def strip_suffixes(name: str) -> str:
    """Strip common reference suffixes like '-REF'."""
    return re.sub(r"-(REF|Ref|ref)$", "", name).strip()


LEGACY_MAP = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
}


def legacy_to_modern(name: str) -> str:
    """Map legacy 10–20 labels to modern ones."""
    return LEGACY_MAP.get(name.upper(), name)


def canon(name: str) -> str:
    """
    Canonicalize a channel name:
    - strip suffixes
    - apply legacy mapping
    - fix capitalization for 10–20-like names
    """
    n = strip_suffixes(name)
    n = legacy_to_modern(n)

    # Special handling for multi-letter prefixes
    for pre in ["Fp", "AF", "FT", "FC", "CP", "PO", "TP"]:
        if n.upper().startswith(pre.upper()):
            return pre + n[len(pre):]

    # Generic: first letter uppercase, rest as-is
    if len(n) >= 1 and n[0].isalpha():
        return n[0].upper() + n[1:]
    return n


# Common EEG 10–20 set + fixed order (19 channels)
COMMON_ORDER = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]
COMMON_SET = set(COMMON_ORDER)


# -------------------------- IHBAS-specific channel filtering --------------------------


# All the polygraph/aux channels we want to drop by prefix
IHBAS_DROP_PREFIXES = {
    "ECG", "EMG", "RLEG", "LLEG", "FLOW", "PRESSURE", "SNORE", "ABDOMEN",
    "CHEST", "PHASE", "RMI", "RR", "X", "POSITION", "ELEVATION", "ACTIVITY",
    "PPG", "PTT", "PLETH", "DC", "TRIG", "SPO2", "PR", "PULSEQUALITY",
}


def should_keep_ihbas(ch_name: str) -> bool:
    """
    Decide whether to keep a channel from an IHBAS EDF.
    We:
      - Drop obvious non-EEG/polygraph channels by prefix.
      - Drop A1/A2 explicitly (we'll use average reference).
      - Keep only channels in COMMON_SET (19-channel 10–20 subset).
    """
    base = ch_name.split()[0]
    up = base.upper()

    # Drop polygraph / aux channels
    if any(up.startswith(pfx) for pfx in IHBAS_DROP_PREFIXES):
        return False

    # Drop mastoids; we'll use average reference
    if base in ("A1", "A2"):
        return False

    # Keep only if it's in our canonical EEG set
    return base in COMMON_SET


# -------------------------- MNE IO helpers --------------------------


def read_raw_compat(edf_path: Path) -> mne.io.BaseRaw:
    """Backward-compatible EDF reader for different MNE versions."""
    try:
        return mne.io.read_raw_edf(
            str(edf_path),
            preload=True,
            infer_types=True,
            verbose="ERROR",
        )
    except TypeError:
        return mne.io.read_raw_edf(
            str(edf_path),
            preload=True,
            verbose="ERROR",
        )


# -------------------------- Core preprocessing --------------------------


def harmonize_raw_ihbas(
    raw: mne.io.BaseRaw,
    target_sfreq: float = 125.0,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch: Optional[float] = 50.0,
) -> mne.io.BaseRaw:
    """
    Minimal preprocessing that preserves discriminative features:

    - Channel name harmonization (legacy 10–20 -> modern).
    - Drop non-EEG/polygraph channels.
    - Set standard_1020 montage (best-effort).
    - Average reference.
    - Notch filter at 50 Hz (optional).
    - Band-pass 0.5–40 Hz.
    - Resample to target_sfreq.
    - Reorder channels to COMMON_ORDER subset present in data.
    """

    # 1. Channel name harmonization
    rename_map = {ch: canon(ch) for ch in raw.ch_names}
    raw.rename_channels(rename_map)

    # 2. Drop non-EEG / polygraph channels
    keep = [ch for ch in raw.ch_names if should_keep_ihbas(ch)]
    if not keep:
        raise RuntimeError("No common EEG channels found after filtering.")
    raw.pick(keep)

    # 3. Set montage (best-effort)
    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore", verbose="ERROR")
    except Exception:
        pass

    # 4. Average reference
    raw.set_eeg_reference("average", verbose="ERROR")

    # 5. Notch filter
    if notch is not None and notch > 0:
        try:
            raw.notch_filter(freqs=[float(notch)], verbose="ERROR")
        except Exception:
            # If notch fails for whatever reason, continue
            pass

    # 6. Band-pass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    # 7. Resample
    if abs(raw.info["sfreq"] - target_sfreq) > 1e-6:
        raw.resample(target_sfreq, verbose="ERROR")

    # 8. Reorder channels into COMMON_ORDER subset
    order = [ch for ch in COMMON_ORDER if ch in raw.ch_names]
    raw.reorder_channels(order)

    return raw


# -------------------------- Segmentation --------------------------


def segment_raw_to_array(
    raw: mne.io.BaseRaw,
    seg_sec: int = 10,
) -> Tuple[np.ndarray, list]:
    """
    Segment raw data into fixed-length non-overlapping windows.

    Returns:
        arr: np.ndarray of shape (N_segments, N_channels, T)
        ch_names: list of channel names
    """
    chs = raw.ch_names
    raw_use = raw.copy().pick(chs)

    sf = float(raw_use.info["sfreq"])
    seg_len = int(round(seg_sec * sf))

    data = raw_use.get_data()  # (C, T)
    n_times = data.shape[1]
    n_full = n_times // seg_len
    if n_full == 0:
        return np.zeros((0, data.shape[0], seg_len), dtype=np.float32), raw_use.ch_names

    data = data[:, : n_full * seg_len]
    arr = data.reshape(data.shape[0], n_full, seg_len)  # (C, N, T)
    arr = np.transpose(arr, (1, 0, 2)).astype(np.float32)  # (N, C, T)

    return arr, raw_use.ch_names


# -------------------------- Path helpers --------------------------


def subject_id_from_path(edf_path: Path) -> str:
    """Simple subject id from filename stem."""
    return edf_path.stem


def mirror_out_path(in_root: Path, out_root: Path, edf_path: Path, subject_id: str) -> Path:
    """
    Mirror the Train/Test/Epileptic/Mimickers structure from in_root
    into out_root, preserving split + class directories.
    """
    rel = edf_path.relative_to(in_root)  # e.g. "Train/Epileptic/12679.edf"
    out_dir = out_root / rel.parent     # e.g. "preprocessed_ihbas/Train/Epileptic"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{subject_id}.npy"


def find_all_edfs(in_root: Path) -> List[Path]:
    """Find all EDF files under the IHBAS tree: Train/Test, Epileptic/Mimickers."""
    splits = ["Train", "Test"]
    classes = ["Epileptic", "Mimickers"]
    edfs: List[Path] = []
    for s in splits:
        for cl in classes:
            base = in_root / s / cl
            if base.exists():
                edfs.extend(base.glob("*.edf"))
    return edfs


# -------------------------- Main --------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess IHBAS EDF files into standardized 10 s segments."
    )
    ap.add_argument(
        "--in_root",
        type=str,
        default="data/data_ihbas",
        help="Input root directory (contains Train/Test/Epileptic/Mimickers subdirs).",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="preprocessed_ihbas",
        help="Output root directory for .npy files.",
    )
    ap.add_argument(
        "--target_sfreq",
        type=float,
        default=125.0,
        help="Target sampling rate (Hz) after resampling.",
    )
    ap.add_argument(
        "--seg_sec",
        type=int,
        default=10,
        help="Segment length in seconds.",
    )
    ap.add_argument(
        "--notch",
        type=float,
        default=50.0,
        help="Notch frequency (Hz); set <=0 to disable.",
    )
    ap.add_argument(
        "--l_freq",
        type=float,
        default=0.5,
        help="High-pass (Hz).",
    )
    ap.add_argument(
        "--h_freq",
        type=float,
        default=40.0,
        help="Low-pass (Hz).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-file logging.",
    )
    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    notch = None if args.notch is not None and float(args.notch) <= 0 else args.notch

    print("\n" + "=" * 70)
    print("IHBAS EEG PREPROCESSING")
    print("=" * 70)
    print(f"Input root     : {in_root}")
    print(f"Output root    : {out_root}")
    print(f"Sampling rate  : {args.target_sfreq} Hz")
    print(f"Segment length : {args.seg_sec} s")
    print(f"Band-pass      : {args.l_freq}–{args.h_freq} Hz")
    print(f"Notch          : {notch if notch is not None else 'DISABLED'}")
    print(f"Reference      : AVERAGE")
    print("=" * 70 + "\n")

    edfs = find_all_edfs(in_root)
    if not edfs:
        print(f"No EDF files found under {in_root}/Train|Test/Epileptic|Mimickers", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(edfs)} EDF files. Processing...\n")
    total_ok = 0
    total_segments = 0

    for p in sorted(edfs):
        try:
            raw = read_raw_compat(p)
            raw = harmonize_raw_ihbas(
                raw,
                target_sfreq=args.target_sfreq,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                notch=notch,
            )
            arr, ch_names = segment_raw_to_array(raw, seg_sec=args.seg_sec)
            out_path = mirror_out_path(in_root, out_root, p, subject_id_from_path(p))
            np.save(out_path, arr)

            if not args.quiet:
                print(
                    f"[OK] {p.name:30s} -> {out_path} | "
                    f"segs={arr.shape[0]:3d} ch={arr.shape[1]:2d} len={arr.shape[2]:4d} "
                    f"({raw.info['sfreq']:.1f} Hz)"
                )

            total_ok += 1
            total_segments += int(arr.shape[0])
        except Exception as e:
            print(f"[FAIL] {p} :: {e}", file=sys.stderr)

    print(f"\nDone. Processed OK: {total_ok}/{len(edfs)} files.")
    print(f"Total {args.seg_sec}s segments saved: {total_segments}")
    print(f"Output root: {out_root}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
