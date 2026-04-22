#!/usr/bin/env python3
"""Convert antiSMASH DB v5 GenBank archives to training JSONL.

Each per-genome .gbk.gz file inside the tar is parsed.  Every antiSMASH
``region`` feature becomes one training record using COMPOUND_CLASS-only
conditioning (no COMPOUND token — antiSMASH DB has class-level supervision
only, per Section 3.2 of the research plan):

    |COMPOUND_CLASS:{cls}|{tax_tag}{region_seq}

Input layout
------------
data/antismash_db/asdb5_gbks.tar        — outer tar, one .gbk.gz per genome assembly
  └─ asdb5_gbks/GCF_*.gbk.gz           — whole-genome GBK annotated by antiSMASH 8.1
data/antismash_db/asdb5_taxa.json.gz    — pre-computed lineage for every taxid
                                          structure: {"mappings": {taxid: {lineage...}}}

Taxonomy resolution (in priority order)
-----------------------------------------
1. asdb5_taxa.json.gz fast-path:  taxid from GBK /db_xref → pre-computed lineage
2. NCBI taxdump fallback:         full names.dmp/nodes.dmp tree walk
3. GenBank ORGANISM block parser: last resort, least accurate for eukaryotes

AntiSMASH region extraction
-----------------------------
Each genome may contain multiple BGC regions.  For each we:
  1. Find ``region`` features with ``/tool="antismash"``.
  2. Slice the genome sequence at region coordinates.
  3. Map ``/product`` qualifier(s) → harmonised COMPOUND_CLASS.
  4. Emit one JSONL record per region.

Usage
-----
    conda activate bgcmodel
    python scripts/antismash_db_to_jsonl.py

    # All flags:
    python scripts/antismash_db_to_jsonl.py \\
        --tar          data/antismash_db/asdb5_gbks.tar \\
        --taxa         data/antismash_db/asdb5_taxa.json.gz \\
        --output       data/processed/asdb5_train_records.jsonl \\
        --class-map    config/compound_class_map.yaml \\
        --taxonomy-dir data/ncbi_taxonomy \\
        --heldout      data/processed/splits/heldout_accessions.txt \\
        --max-length   262144 \\
        --min-length   100 \\
        --limit        1000
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from bgc_pipeline.class_map import load_class_map
from bgc_pipeline.taxonomy import (
    NCBITaxonomy,
    build_taxonomic_tag,
    load_taxonomy,
)


# ---------------------------------------------------------------------------
# Product → harmonised class
# ---------------------------------------------------------------------------

def map_region_products(
    products: list[str],
    mapping: dict[str, str],
    default: str,
) -> str:
    """Map a list of antiSMASH product types to a single harmonised class label.

    Handles:
    - Single product:    T1PKS → PKS
    - PKS+NRPS hybrid:  [T1PKS, NRPS] → PKS_NRPS_HYBRID
    - Unknown:           falls back to ``default``
    """
    if not products:
        return default

    mapped: list[str] = []
    for prod in products:
        # Products may be space-separated in some antiSMASH versions
        for p in prod.replace(",", " ").split():
            p = p.strip()
            if not p:
                continue
            cls = mapping.get(p.lower(), None)
            if cls:
                mapped.append(cls)

    # Deduplicate, preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in mapped:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if not unique:
        return default
    if len(unique) == 1:
        return unique[0]

    # Known hybrid combinations
    cls_set = set(unique)
    if {"PKS", "NRPS"}.issubset(cls_set):
        return "PKS_NRPS_HYBRID"

    # Return first non-OTHER class for other hybrid combos
    for c in unique:
        if c != "OTHER":
            return c
    return unique[0]


# ---------------------------------------------------------------------------
# Taxa JSON fast-path taxonomy
# ---------------------------------------------------------------------------

# Maps asdb5_taxa.json.gz field names → Evo2 rank prefix
_TAXA_JSON_RANK_ORDER = [
    ("superkingdom", "D"),
    ("phylum",       "P"),
    ("class",        "C"),
    ("order",        "O"),
    ("family",       "F"),
    ("genus",        "G"),
    ("species",      "S"),
]


def load_taxa_json(taxa_path: Path) -> dict[int, dict]:
    """Load asdb5_taxa.json.gz → {taxid(int): lineage_dict}.

    Expected structure::

        {
          "deprecated_ids": {"old_taxid": new_taxid, ...},
          "mappings": {
            "taxid": {
              "tax_id": ..., "name": ...,
              "superkingdom": ..., "phylum": ..., "class": ...,
              "order": ..., "family": ..., "genus": ..., "species": ...
            },
            ...
          }
        }

    Returns empty dict if file is missing or malformed.
    """
    if not taxa_path.is_file():
        return {}
    try:
        with gzip.open(taxa_path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        mappings = raw.get("mappings", {})
        result: dict[int, dict] = {}
        for k, v in mappings.items():
            try:
                result[int(k)] = v
            except (ValueError, TypeError):
                pass
        # Also store deprecated-id redirects
        deprecated = raw.get("deprecated_ids", {})
        _DEPRECATED_IDS: dict[int, int] = {}
        for old, new in deprecated.items():
            try:
                _DEPRECATED_IDS[int(old)] = int(new)
            except (ValueError, TypeError):
                pass
        return result, _DEPRECATED_IDS
    except Exception as exc:
        print(f"WARNING: could not load taxa JSON {taxa_path}: {exc}", file=sys.stderr)
        return {}, {}


def build_tag_from_taxa_entry(entry: dict) -> str | None:
    """Build an Evo2-style taxonomy tag from one asdb5_taxa mapping entry.

    Returns None if the entry has too few ranks to be useful.
    """
    segments: list[str] = []
    for field, prefix in _TAXA_JSON_RANK_ORDER:
        name = entry.get(field, "") or ""
        name = name.strip()
        if not name or name.lower() in ("unknown", "unclassified", ""):
            continue
        safe = re.sub(r"[^A-Z0-9_]", "_", name.upper())
        safe = re.sub(r"_+", "_", safe).strip("_")
        if safe:
            segments.append(f"{prefix}__{safe}")

    if not segments:
        return None
    return f"|{';'.join(segments)}|"


def taxa_tag_for_gbk(
    gbk_text: str,
    taxa_map: dict[int, dict],
    deprecated_ids: dict[int, int],
) -> str | None:
    """Resolve taxonomy tag using the taxa JSON fast-path.

    Extracts taxon ID from ``/db_xref="taxon:XXXX"`` in the GBK text,
    looks up pre-computed lineage in the taxa map, and returns the tag.
    Returns None if any lookup step fails.
    """
    m = re.search(r'/db_xref="taxon:(\d+)"', gbk_text)
    if not m:
        return None
    taxid = int(m.group(1))
    # Resolve deprecated taxids
    taxid = deprecated_ids.get(taxid, taxid)
    entry = taxa_map.get(taxid)
    if entry is None:
        return None
    return build_tag_from_taxa_entry(entry)


# ---------------------------------------------------------------------------
# GenBank parsing
# ---------------------------------------------------------------------------

def _parse_gbk_bytes(raw_bytes: bytes, member_name: str) -> list[SeqRecord]:
    """Decompress (if .gz) and parse ALL records from a GenBank byte string.

    antiSMASH GBKs for fragmented assemblies contain one record per contig.
    We must iterate all records to find every BGC region.
    Returns an empty list on parse failure.
    """
    try:
        data = gzip.decompress(raw_bytes) if member_name.endswith(".gz") else raw_bytes
        handle = io.StringIO(data.decode("ascii", errors="replace"))
        return list(SeqIO.parse(handle, "genbank"))
    except Exception as exc:
        print(f"  WARNING: failed to parse {member_name}: {exc}", file=sys.stderr)
        return []


def _iter_antismash_regions(
    records: list[SeqRecord],
    genome_acc: str,
    gbk_text: str,
    class_mapping: dict[str, str],
    class_default: str,
    taxa_map: dict[int, dict],
    deprecated_ids: dict[int, int],
    taxonomy: NCBITaxonomy | None,
    max_length: int,
    min_length: int,
) -> Iterator[dict]:
    """Yield one training-record dict per antiSMASH ``region`` feature.

    Iterates ALL records in the GBK (handles fragmented multi-contig assemblies).
    Taxonomy tag is resolved once per genome from the full GBK text.
    """

    # --- Resolve taxonomy tag once per genome ---
    # Priority: taxa JSON fast-path → NCBI taxdump → GenBank fallback
    tax_tag: str | None = None
    if taxa_map:
        tax_tag = taxa_tag_for_gbk(gbk_text, taxa_map, deprecated_ids)
    if tax_tag is None:
        tax_tag = build_taxonomic_tag(gbk_text, taxonomy=taxonomy)

    for record in records:
        full_seq = str(record.seq).upper()

        for feature in record.features:
            if feature.type != "region":
                continue
            if feature.qualifiers.get("tool") != ["antismash"]:
                continue

            products: list[str] = feature.qualifiers.get("product", [])
            region_number: str = feature.qualifiers.get("region_number", ["?"])[0]
            compound_class = map_region_products(products, class_mapping, class_default)
            contig_edge: bool = (
                feature.qualifiers.get("contig_edge", ["False"])[0].strip().lower() == "true"
            )

            start = int(feature.location.start)
            end = int(feature.location.end)
            region_seq = full_seq[start:end]

            if len(region_seq) < min_length:
                continue

            if len(region_seq) > max_length:
                # Centre-truncate to fit Evo2's context window
                mid = (start + end) // 2
                half = max_length // 2
                trunc_start = max(0, mid - half)
                region_seq = full_seq[trunc_start: trunc_start + max_length]

            accession = f"{genome_acc}.region{region_number}"
            training_text = f"|COMPOUND_CLASS:{compound_class}|{tax_tag}{region_seq}"

            yield {
                "accession": accession,
                "genome_accession": genome_acc,
                "region_number": int(region_number) if region_number.isdigit() else region_number,
                "compound_class": compound_class,
                "antismash_products": products,
                "contig_edge": contig_edge,
                "region_start": start,
                "region_end": end,
                "taxonomic_tag": tax_tag,
                "sequence": region_seq,
                "training_text": training_text,
                "gbk_member": genome_acc,
            }


# ---------------------------------------------------------------------------
# Main streaming iterator
# ---------------------------------------------------------------------------

def iter_asdb5_records(
    tar_path: Path,
    class_mapping: dict[str, str],
    class_default: str,
    taxa_map: dict[int, dict],
    deprecated_ids: dict[int, int],
    taxonomy: NCBITaxonomy | None,
    heldout: set[str],
    max_length: int,
    min_length: int,
    limit: int | None,
    resume_after: str | None = None,
    only_genomes: set[str] | None = None,
) -> Iterator[dict]:
    """Stream ``tar_path`` and yield one dict per antiSMASH BGC region.

    Handles truncated / still-downloading tars gracefully.
    If ``resume_after`` is set, skip all genomes up to and including that accession.
    If ``only_genomes`` is set, only process genomes in that set.
    """
    yielded = 0
    # Resume support: skip genomes until we pass resume_after
    skipping = resume_after is not None
    if skipping:
        print(
            f"  Resume mode: skipping genomes up to and including {resume_after}",
            file=sys.stderr,
        )
    if only_genomes is not None:
        print(
            f"  Patch mode: processing {len(only_genomes):,} specific genomes only",
            file=sys.stderr,
        )

    try:
        tf = tarfile.open(tar_path, "r:*")
    except Exception as exc:
        print(f"ERROR: cannot open {tar_path}: {exc}", file=sys.stderr)
        return

    try:
        for member in tf:
            if limit is not None and yielded >= limit:
                break

            name = member.name
            if not (name.endswith(".gbk.gz") or name.endswith(".gbk")):
                continue

            # Derive genome accession from filename
            stem = Path(name).name
            for suffix in (".gbk.gz", ".gbk"):
                if stem.endswith(suffix):
                    genome_acc = stem[: -len(suffix)]
                    break
            else:
                genome_acc = stem

            # Resume: skip until we've passed the resume_after genome
            if skipping:
                if genome_acc == resume_after:
                    skipping = False  # next genome will be processed
                continue

            # Patch mode: only process specific genomes
            if only_genomes is not None and genome_acc not in only_genomes:
                continue

            if genome_acc in heldout:
                continue

            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            raw = fobj.read()

            records = _parse_gbk_bytes(raw, name)
            if not records:
                continue

            try:
                gbk_text = (
                    gzip.decompress(raw).decode("ascii", errors="replace")
                    if name.endswith(".gz")
                    else raw.decode("ascii", errors="replace")
                )
            except Exception:
                gbk_text = ""

            for rec_dict in _iter_antismash_regions(
                records, genome_acc, gbk_text,
                class_mapping, class_default,
                taxa_map, deprecated_ids, taxonomy,
                max_length, min_length,
            ):
                yield rec_dict
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

    except (EOFError, tarfile.TarError) as exc:
        print(
            f"  INFO: reached end of available data in {tar_path.name} "
            f"after {yielded:,} records ({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
    finally:
        try:
            tf.close()
        except Exception:
            pass


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
        help="Path to asdb5_gbks.tar",
    )
    parser.add_argument(
        "--taxa",
        type=Path,
        default=ROOT / "data" / "antismash_db" / "asdb5_taxa.json.gz",
        help="asdb5_taxa.json.gz for fast taxonomy lookups",
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
        help="NCBI taxdump directory (names.dmp + nodes.dmp) — fallback only",
    )
    parser.add_argument(
        "--heldout",
        type=Path,
        default=ROOT / "data" / "processed" / "splits" / "heldout_accessions.txt",
        help="One accession per line to exclude from output",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=ROOT / "data" / "processed" / "asdb5_train_records.jsonl",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=262144,
        help="Max region bp; longer regions centre-truncated (default: 262144)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Min region bp; shorter skipped (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N records (testing only)",
    )
    parser.add_argument(
        "--resume-after",
        type=str,
        default=None,
        help="Genome accession (e.g. GCF_902832935.1) to resume after; "
             "skips all genomes up to and including this one. "
             "Use --append to append to an existing output file.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Open output in append mode (use with --resume-after)",
    )
    parser.add_argument(
        "--only-genomes-file",
        type=Path,
        default=None,
        help="File with one genome accession per line; only those genomes are processed. "
             "Used to patch a previous run that missed certain genomes.",
    )
    args = parser.parse_args()

    # Class map
    class_mapping, class_default = load_class_map(args.class_map)
    print(
        f"Loaded {len(class_mapping)} class-map entries (default: {class_default})",
        file=sys.stderr,
    )

    # Taxa JSON fast-path
    taxa_map: dict[int, dict] = {}
    deprecated_ids: dict[int, int] = {}
    if args.taxa.is_file():
        print(f"Loading taxa JSON: {args.taxa} ...", file=sys.stderr, flush=True)
        result = load_taxa_json(args.taxa)
        if isinstance(result, tuple):
            taxa_map, deprecated_ids = result
        else:
            taxa_map = result
        print(
            f"  Loaded {len(taxa_map):,} taxid entries "
            f"({len(deprecated_ids):,} deprecated-id redirects)",
            file=sys.stderr,
        )
    else:
        print(
            f"WARNING: taxa JSON not found at {args.taxa}; "
            "will fall back to NCBI taxdump or GenBank parsing.",
            file=sys.stderr,
        )

    # NCBI taxonomy (fallback only — skipped if taxa JSON covers everything)
    taxonomy: NCBITaxonomy | None = None
    if args.taxonomy_dir.is_dir() and (args.taxonomy_dir / "names.dmp").exists():
        print("Loading NCBI taxonomy dump (fallback) ...", file=sys.stderr, flush=True)
        taxonomy = load_taxonomy(args.taxonomy_dir)
        print(
            f"  Loaded {len(taxonomy.nodes):,} nodes",
            file=sys.stderr,
        )
    else:
        print(
            f"NOTE: NCBI taxdump not found at {args.taxonomy_dir}; "
            "GenBank ORGANISM parser will be last-resort fallback.",
            file=sys.stderr,
        )

    # Heldout set
    heldout: set[str] = set()
    if args.heldout.is_file():
        heldout = {
            ln.strip()
            for ln in args.heldout.read_text("utf-8").splitlines()
            if ln.strip()
        }
        print(f"Loaded {len(heldout):,} heldout accessions.", file=sys.stderr)

    # Only-genomes filter
    only_genomes: set[str] | None = None
    if args.only_genomes_file is not None:
        if not args.only_genomes_file.is_file():
            print(f"ERROR: --only-genomes-file not found: {args.only_genomes_file}", file=sys.stderr)
            sys.exit(1)
        only_genomes = {
            ln.strip()
            for ln in args.only_genomes_file.read_text("utf-8").splitlines()
            if ln.strip()
        }
        print(f"Patch mode: {len(only_genomes):,} target genomes loaded.", file=sys.stderr)

    # Validate tar
    if not args.tar.is_file():
        print(f"ERROR: tar not found: {args.tar}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    class_counts: dict[str, int] = {}

    open_mode = "a" if args.append else "w"
    if args.append:
        print(
            f"\nAppending to {args.output} (resume after {args.resume_after}) ...",
            file=sys.stderr, flush=True,
        )
    else:
        print(f"\nStreaming {args.tar.name} → {args.output} ...", file=sys.stderr, flush=True)

    with args.output.open(open_mode, encoding="utf-8") as out_fh:
        for rec in iter_asdb5_records(
            tar_path=args.tar,
            class_mapping=class_mapping,
            class_default=class_default,
            taxa_map=taxa_map,
            deprecated_ids=deprecated_ids,
            taxonomy=taxonomy,
            heldout=heldout,
            max_length=args.max_length,
            min_length=args.min_length,
            limit=args.limit,
            resume_after=args.resume_after,
            only_genomes=only_genomes,
        ):
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            class_counts[rec["compound_class"]] = (
                class_counts.get(rec["compound_class"], 0) + 1
            )
            if n_written % 5000 == 0:
                print(f"  ... {n_written:,} records", file=sys.stderr, flush=True)

    print(f"\nWrote {n_written:,} records to {args.output}", file=sys.stderr)
    print("\nClass distribution:", file=sys.stderr)
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / max(n_written, 1)
        print(f"  {cls:25s} {cnt:7,d}  ({pct:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
