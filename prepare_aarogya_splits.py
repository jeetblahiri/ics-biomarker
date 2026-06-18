#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare Aarogya EDF data for the ICS biomarker pipeline.

This script is deliberately non-destructive:
  1. `split` scans raw EDFs and writes a subject-level Train/Eval/Test manifest.
  2. `preprocess` reads that manifest and writes preprocessed .npy arrays.

Raw data is never moved or modified.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from preprocess_max import (
    harmonize_raw_generic,
    read_raw_compat,
    segment_raw_to_array,
)


DEFAULT_RAW_ROOT = Path("/share/data/lakshya/data-aarogya-raw-final")
DEFAULT_SPLIT_MANIFEST = Path("splits/aarogya_ics_subject_splits_seed0.csv")

CLASS_TO_LABEL = {
    "epileptic": ("Epileptic", 1),
    "mimicker": ("Mimickers", 0),
}

SPLITS = ("Train", "Eval", "Test")


@dataclass(frozen=True)
class EdfRecord:
    raw_path: Path
    relative_path: str
    class_raw: str
    class_dir: str
    label: int
    site: str
    subject_id: str
    subject_key: str
    collection: str


def parse_raw_edf_path(raw_root: Path, path: Path, include_ieds: bool) -> Optional[EdfRecord]:
    try:
        parts = path.relative_to(raw_root).parts
    except ValueError:
        return None

    if not parts or parts[0] == "anomalies":
        return None

    class_raw = parts[0]
    if class_raw not in CLASS_TO_LABEL:
        return None

    if len(parts) < 4:
        return None

    site = parts[1]
    collection = "regular"
    subject_idx = 2

    if len(parts) > 2 and parts[2] == "IEDs":
        if not include_ieds:
            return None
        collection = "IEDs"
        subject_idx = 3

    if len(parts) <= subject_idx:
        return None

    subject_id = parts[subject_idx]
    class_dir, label = CLASS_TO_LABEL[class_raw]

    # Use site + subject for leakage control so regular and IED recordings from
    # the same patient cannot land in different splits.
    subject_key = f"{site}::{subject_id}"

    return EdfRecord(
        raw_path=path,
        relative_path=str(path.relative_to(raw_root)),
        class_raw=class_raw,
        class_dir=class_dir,
        label=label,
        site=site,
        subject_id=subject_id,
        subject_key=subject_key,
        collection=collection,
    )


def iter_edf_records(raw_root: Path, include_ieds: bool) -> Iterable[EdfRecord]:
    for path in sorted(raw_root.rglob("*.edf")):
        record = parse_raw_edf_path(raw_root, path, include_ieds=include_ieds)
        if record is not None:
            yield record


def split_subjects(
    records: list[EdfRecord],
    train_frac: float,
    eval_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    total = train_frac + eval_frac + test_frac
    if total <= 0:
        raise ValueError("Split fractions must sum to a positive value.")
    train_frac, eval_frac, test_frac = (train_frac / total, eval_frac / total, test_frac / total)

    by_subject: dict[str, list[EdfRecord]] = defaultdict(list)
    for record in records:
        by_subject[record.subject_key].append(record)

    subject_group: dict[str, tuple[str, str, str]] = {}
    conflicts: list[str] = []
    for subject_key, subject_records in by_subject.items():
        labels = {r.class_dir for r in subject_records}
        sites = {r.site for r in subject_records}
        if len(labels) != 1 or len(sites) != 1:
            conflicts.append(subject_key)
            continue
        label = next(iter(labels))
        site = next(iter(sites))
        has_ied = any(r.collection == "IEDs" for r in subject_records)
        collection = "IEDs" if has_ied else "regular"
        subject_group[subject_key] = (label, site, collection)

    if conflicts:
        examples = ", ".join(sorted(conflicts)[:10])
        raise RuntimeError(f"Found cross-label or cross-site subject conflicts: {examples}")

    rng = random.Random(seed)
    grouped_subjects: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for subject_key, group in subject_group.items():
        grouped_subjects[group].append(subject_key)

    assignments: dict[str, str] = {}
    for group, subject_keys in sorted(grouped_subjects.items()):
        keys = sorted(subject_keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = round(n * train_frac)
        n_eval = round(n * eval_frac)

        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_eval = max(1, min(n_eval, n - n_train - 1))
        elif n == 2:
            n_train, n_eval = 1, 0
        elif n == 1:
            n_train, n_eval = 1, 0

        for subject_key in keys[:n_train]:
            assignments[subject_key] = "Train"
        for subject_key in keys[n_train:n_train + n_eval]:
            assignments[subject_key] = "Eval"
        for subject_key in keys[n_train + n_eval:]:
            assignments[subject_key] = "Test"

    return assignments


def write_split_manifest(
    records: list[EdfRecord],
    assignments: dict[str, str],
    raw_root: Path,
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "label",
        "class_dir",
        "class_raw",
        "site",
        "subject_id",
        "subject_key",
        "collection",
        "recording_name",
        "raw_path",
        "relative_path",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            split = assignments[record.subject_key]
            writer.writerow({
                "split": split,
                "label": record.label,
                "class_dir": record.class_dir,
                "class_raw": record.class_raw,
                "site": record.site,
                "subject_id": record.subject_id,
                "subject_key": record.subject_key,
                "collection": record.collection,
                "recording_name": record.raw_path.name,
                "raw_path": str(record.raw_path),
                "relative_path": str(record.raw_path.relative_to(raw_root)),
            })


def summarize_manifest_rows(rows: list[dict[str, str]]) -> str:
    file_counts = Counter((r["split"], r["class_dir"], r["site"], r["collection"]) for r in rows)
    subject_counts = defaultdict(set)
    for row in rows:
        subject_counts[(row["split"], row["class_dir"], row["site"], row["collection"])].add(row["subject_key"])

    lines = ["split,class,site,collection,subjects,files"]
    keys = sorted(file_counts)
    for key in keys:
        split, class_dir, site, collection = key
        lines.append(
            f"{split},{class_dir},{site},{collection},"
            f"{len(subject_counts[key])},{file_counts[key]}"
        )
    return "\n".join(lines)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def safe_output_stem(row: dict[str, str]) -> str:
    stem = Path(row["recording_name"]).stem
    subject = row["subject_id"]
    if stem == subject:
        return subject
    return f"{subject}__{stem}"


def preprocess_manifest(
    manifest_csv: Path,
    out_root: Path,
    target_sfreq: float,
    seg_sec: int,
    l_freq: float,
    h_freq: float,
    notch: Optional[float],
    limit: Optional[int],
    overwrite: bool,
) -> tuple[int, int, int]:
    rows = read_manifest(manifest_csv)
    out_root.mkdir(parents=True, exist_ok=True)
    total = len(rows) if limit is None else min(limit, len(rows))
    ok = 0
    skipped = 0
    failed = 0

    for idx, row in enumerate(rows[:total], start=1):
        raw_path = Path(row["raw_path"])
        out_dir = out_root / row["split"] / row["class_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_output_stem(row)}.npy"

        if out_path.exists() and not overwrite:
            skipped += 1
            print(f"[SKIP] {idx}/{total} exists {out_path}")
            continue

        try:
            raw = read_raw_compat(raw_path)
            raw = harmonize_raw_generic(
                raw,
                target_sfreq=target_sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                notch=notch,
            )
            arr, ch_names = segment_raw_to_array(raw, seg_sec=seg_sec)
            if arr.shape[0] == 0:
                raise RuntimeError("recording shorter than one full segment")
            np.save(out_path, arr)
            ok += 1
            print(
                f"[OK] {idx}/{total} {raw_path.name} -> {out_path} "
                f"segs={arr.shape[0]} ch={arr.shape[1]} len={arr.shape[2]}"
            )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {idx}/{total} {raw_path} :: {exc}", file=sys.stderr)

    return ok, skipped, failed


def cmd_split(args: argparse.Namespace) -> int:
    raw_root = Path(args.raw_root).expanduser().resolve()
    records = list(iter_edf_records(raw_root, include_ieds=args.include_ieds))
    if not records:
        raise RuntimeError(f"No usable EDF records found under {raw_root}")

    assignments = split_subjects(
        records,
        train_frac=args.train_frac,
        eval_frac=args.eval_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    out_csv = Path(args.out_csv).expanduser().resolve()
    write_split_manifest(records, assignments, raw_root=raw_root, out_csv=out_csv)

    rows = read_manifest(out_csv)
    print(f"Wrote split manifest: {out_csv}")
    print(f"Raw root: {raw_root}")
    print(f"Files: {len(rows)}")
    print(f"Subjects: {len({r['subject_key'] for r in rows})}")
    print()
    print(summarize_manifest_rows(rows))
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    notch = None if args.notch is not None and float(args.notch) <= 0 else float(args.notch)
    ok, skipped, failed = preprocess_manifest(
        manifest_csv=Path(args.manifest).expanduser().resolve(),
        out_root=Path(args.out_root).expanduser().resolve(),
        target_sfreq=args.target_sfreq,
        seg_sec=args.seg_sec,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch=notch,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"\nDone. OK={ok} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Aarogya data for ICS experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="Create a subject-level Train/Eval/Test manifest.")
    split.add_argument("--raw_root", type=str, default=str(DEFAULT_RAW_ROOT))
    split.add_argument("--out_csv", type=str, default=str(DEFAULT_SPLIT_MANIFEST))
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--train_frac", type=float, default=0.70)
    split.add_argument("--eval_frac", type=float, default=0.15)
    split.add_argument("--test_frac", type=float, default=0.15)
    split.add_argument("--include_ieds", action="store_true", help="Include epileptic/max/IEDs as epileptic.")
    split.set_defaults(func=cmd_split)

    prep = sub.add_parser("preprocess", help="Preprocess EDFs listed in a split manifest.")
    prep.add_argument("--manifest", type=str, default=str(DEFAULT_SPLIT_MANIFEST))
    prep.add_argument("--out_root", type=str, default="preprocessed_aarogya_ics")
    prep.add_argument("--target_sfreq", type=float, default=125.0)
    prep.add_argument("--seg_sec", type=int, default=10)
    prep.add_argument("--notch", type=float, default=50.0)
    prep.add_argument("--l_freq", type=float, default=0.5)
    prep.add_argument("--h_freq", type=float, default=40.0)
    prep.add_argument("--limit", type=int, default=None, help="Preprocess only the first N manifest rows.")
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=cmd_preprocess)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
