#!/usr/bin/env python3
"""Post-hoc annotation of asdb5_train_records.jsonl with contig_edge field.

Reads the existing JSONL (which lacks contig_edge) and the original tar,
does a single streaming tar pass to build an accession → contig_edge lookup,
then rewrites the JSONL with the new field injected.

This avoids reprocessing sequences, taxonomy, etc. — only region feature
qualifiers are read from the tar.

Usage
-----
    conda activate bgcmodel
    python scripts/annotate_contig_edge.py

    # Custom paths:
    python scripts/annotate_contig_edge.py \\
        --tar     data/antismash_db/asdb5_gbks.tar \\
        --input   data/processed/asdb5_train_records.jsonl \\
        --output  data/processed/asdb5_train_records.annotated.jsonl

After verifying the output, replace the original:
    mv data/processed/asdb5_train_records.annotated.jsonl \\
       data/processed/asdb5_train_records.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Bio import SeqIO


# ---------------------------------------------------------------------------
# Step 1: build contig_edge lookup from tar
# ---------------------------------------------------------------------------

def build_edge_map(tar_path: Path) -> dict[str, bool]:
    """Stream tar once, return {accession: contig_edge} for every region.

    accession format: "{genome_acc}.region{region_number}"
    contig_edge is True if the region feature has contig_edge="True".
    """
    edge_map: dict[str, bool] = {}
    n_genomes = 0

    try:
        tf = tarfile.open(tar_path, "r:*")
    except Exception as exc:
        print(f"ERROR: cannot open {tar_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        for member in tf:
            name = member.name
            if not (name.endswith(".gbk.gz") or name.endswith(".gbk")):
                continue

            stem = Path(name).name
            for suffix in (".gbk.gz", ".gbk"):
                if stem.endswith(suffix):
                    genome_acc = stem[: -len(suffix)]
                    break
            else:
                genome_acc = stem

            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            raw = fobj.read()

            try:
                data = gzip.decompress(raw) if name.endswith(".gz") else raw
                handle = io.StringIO(data.decode("ascii", errors="replace"))
                records = list(SeqIO.parse(handle, "genbank"))
            except Exception as exc:
                print(f"  WARNING: failed to parse {name}: {exc}", file=sys.stderr)
                continue

            for record in records:
                for feature in record.features:
                    if feature.type != "region":
                        continue
                    if feature.qualifiers.get("tool") != ["antismash"]:
                        continue
                    region_number = feature.qualifiers.get("region_number", ["?"])[0]
                    contig_edge = (
                        feature.qualifiers.get("contig_edge", ["False"])[0].strip().lower()
                        == "true"
                    )
                    acc = f"{genome_acc}.region{region_number}"
                    edge_map[acc] = contig_edge

            n_genomes += 1
            if n_genomes % 5000 == 0:
                print(
                    f"  ... {n_genomes:,} genomes scanned, {len(edge_map):,} regions",
                    file=sys.stderr,
                    flush=True,
                )

    except (EOFError, tarfile.TarError) as exc:
        print(
            f"  INFO: end of tar after {n_genomes:,} genomes ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
    finally:
        try:
            tf.close()
        except Exception:
            pass

    print(
        f"Tar scan complete: {n_genomes:,} genomes, {len(edge_map):,} regions",
        file=sys.stderr,
    )
    return edge_map


# ---------------------------------------------------------------------------
# Step 2: annotate JSONL
# ---------------------------------------------------------------------------

def annotate_jsonl(
    input_path: Path,
    output_path: Path,
    edge_map: dict[str, bool],
) -> None:
    """Stream input JSONL, inject contig_edge, write to output."""
    n_in = 0
    n_found = 0
    n_missing = 0

    with (
        input_path.open("r", encoding="utf-8") as in_fh,
        output_path.open("w", encoding="utf-8") as out_fh,
    ):
        for line in in_fh:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARNING: skipping malformed line {n_in + 1}: {exc}", file=sys.stderr)
                n_in += 1
                continue

            acc = rec.get("accession", "")
            if acc in edge_map:
                rec["contig_edge"] = edge_map[acc]
                n_found += 1
            else:
                # Default False — either not in tar (very rare) or already has the field
                rec.setdefault("contig_edge", False)
                n_missing += 1

            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_in += 1

            if n_in % 50_000 == 0:
                print(f"  ... {n_in:,} records annotated", file=sys.stderr, flush=True)

    print(
        f"\nAnnotated {n_in:,} records: "
        f"{n_found:,} matched in edge_map, {n_missing:,} defaulted to False.",
        file=sys.stderr,
    )
    if n_missing > 0:
        pct = 100.0 * n_missing / max(n_in, 1)
        print(
            f"  ({pct:.1f}% defaulted — expected ~0% if tar and JSONL are in sync)",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tar",
        type=Path,
        default=ROOT / "data" / "antismash_db" / "asdb5_gbks.tar",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=ROOT / "data" / "processed" / "asdb5_train_records.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=ROOT / "data" / "processed" / "asdb5_train_records.annotated.jsonl",
    )
    args = parser.parse_args()

    for p, label in [(args.tar, "tar"), (args.input, "input JSONL")]:
        if not p.is_file():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: scanning {args.tar.name} for contig_edge flags ...", file=sys.stderr, flush=True)
    edge_map = build_edge_map(args.tar)

    print(f"\nStep 2: annotating {args.input.name} → {args.output.name} ...", file=sys.stderr, flush=True)
    annotate_jsonl(args.input, args.output, edge_map)

    print(
        f"\nDone. Verify output, then:\n"
        f"  mv {args.output} {args.input}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
