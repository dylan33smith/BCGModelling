#!/usr/bin/env python3
"""Emit MIBiG training records as JSONL (metadata + conditioning string)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bgc_pipeline.class_map import load_class_map
from bgc_pipeline.mibig_record import iter_mibig_records, record_to_json_dict
from bgc_pipeline.taxonomy import load_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mibig-json-dir",
        type=Path,
        default=ROOT / "data" / "mibig" / "mibig_json_4.0",
    )
    parser.add_argument(
        "--mibig-gbk",
        type=Path,
        default=ROOT / "data" / "mibig" / "mibig_gbk_4.0",
        help="Directory of per-cluster .gbk files (default) or mibig_gbk_4.0.tar.gz",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=ROOT / "config" / "compound_class_map.yaml",
    )
    parser.add_argument(
        "--taxonomy-dir",
        type=Path,
        default=ROOT / "data" / "ncbi_taxonomy",
        help="Directory containing NCBI taxdump (names.dmp, nodes.dmp)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max records to emit")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "mibig_train_records.jsonl",
    )
    args = parser.parse_args()

    mapping, default = load_class_map(args.class_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load NCBI taxonomy for proper Evo2-format rank resolution
    taxonomy = None
    if args.taxonomy_dir.is_dir() and (args.taxonomy_dir / "names.dmp").exists():
        print("Loading NCBI taxonomy dump ...", file=sys.stderr, flush=True)
        taxonomy = load_taxonomy(args.taxonomy_dir)
        print(
            f"  Loaded {len(taxonomy.nodes):,} nodes, {len(taxonomy.names):,} names",
            file=sys.stderr,
        )
    else:
        print(
            f"WARNING: NCBI taxonomy not found at {args.taxonomy_dir}; "
            "falling back to GenBank ORGANISM parsing (less accurate for eukaryotes).",
            file=sys.stderr,
        )

    n = 0
    with args.output.open("w", encoding="utf-8") as out:
        for rec in iter_mibig_records(
            args.mibig_json_dir,
            args.mibig_gbk,
            mapping,
            default,
            limit=args.limit,
            taxonomy=taxonomy,
        ):
            out.write(json.dumps(record_to_json_dict(rec), ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
