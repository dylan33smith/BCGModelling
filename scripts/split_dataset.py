#!/usr/bin/env python3
"""Split MIBiG JSONL records into train/val/test sets, stratified by COMPOUND_CLASS.

Filters out records whose sequence exceeds the Evo2 context window (default 262,144 bp).
Reports statistics per split and per class.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/mibig_train_records.jsonl"),
        help="JSONL from mibig_to_jsonl.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Directory to write train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-frac", type=float, default=0.80, help="Training fraction"
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.10, help="Validation fraction"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=262_144,
        help="Max sequence length in bp (Evo2 7B context window). Records longer are dropped.",
    )
    args = parser.parse_args()

    test_frac = round(1.0 - args.train_frac - args.val_frac, 4)
    assert test_frac > 0, "train + val fracs must be < 1.0"

    # Load all records
    records: list[dict] = []
    skipped_long = 0
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["sequence"]) > args.max_seq_len:
                skipped_long += 1
                continue
            records.append(rec)

    print(f"Loaded {len(records)} records (skipped {skipped_long} exceeding {args.max_seq_len:,} bp)", file=sys.stderr)

    # Group by compound class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_class[rec["compound_class"]].append(rec)

    rng = random.Random(args.seed)

    train_recs: list[dict] = []
    val_recs: list[dict] = []
    test_recs: list[dict] = []

    # Stratified split per class
    for cls in sorted(by_class):
        group = by_class[cls]
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, round(n * args.val_frac))
        n_test = max(1, round(n * test_frac))
        n_train = n - n_val - n_test
        if n_train < 1:
            # Tiny class: put all in train, skip val/test
            train_recs.extend(group)
            print(f"  WARNING: class {cls} has only {n} records — all assigned to train", file=sys.stderr)
            continue
        train_recs.extend(group[:n_train])
        val_recs.extend(group[n_train : n_train + n_val])
        test_recs.extend(group[n_train + n_val :])

    # Shuffle each split (remove class ordering)
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)
    rng.shuffle(test_recs)

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train_recs, "val": val_recs, "test": test_recs}
    for name, recs in splits.items():
        outpath = args.output_dir / f"{name}.jsonl"
        with outpath.open("w", encoding="utf-8") as out:
            for rec in recs:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Report
    print(f"\n{'Split':<8} {'Count':>6}  Class distribution", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    for name, recs in splits.items():
        cls_counts = Counter(r["compound_class"] for r in recs)
        dist = ", ".join(f"{c}:{n}" for c, n in cls_counts.most_common())
        print(f"{name:<8} {len(recs):>6}  {dist}", file=sys.stderr)

    # Save held-out accessions for deduplication (Section 4.4)
    heldout_accessions = {r["accession"] for r in val_recs + test_recs}
    heldout_path = args.output_dir / "heldout_accessions.txt"
    with heldout_path.open("w") as f:
        for acc in sorted(heldout_accessions):
            f.write(acc + "\n")
    print(f"\nSaved {len(heldout_accessions)} held-out accessions to {heldout_path}", file=sys.stderr)
    print(f"Output directory: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
