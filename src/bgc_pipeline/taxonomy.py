"""Build Evo2-style taxonomic prefix strings using NCBI Taxonomy.

Evo2 expects ALL-UPPERCASE tags with 7 Linnaean ranks:
    |D__BACTERIA;P__PSEUDOMONADOTA;C__GAMMAPROTEOBACTERIA;O__ENTEROBACTERALES;
     F__ENTEROBACTERIACEAE;G__ESCHERICHIA;S__ESCHERICHIA_COLI|

This module:
  1. Parses NCBI taxdump (names.dmp + nodes.dmp) into a local lookup.
  2. Resolves organisms to taxon IDs via db_xref in GenBank or name lookup.
  3. Walks the taxonomy tree to extract the 7 standard Linnaean ranks.
  4. Formats the result in Evo2's pipe-delimited, uppercase format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# NCBI Taxonomy local index
# ---------------------------------------------------------------------------

# Evo2 rank prefixes in order.
# NCBI changed "superkingdom" → "domain" in recent taxdumps; accept both.
_EVO2_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
_DOMAIN_ALIASES = {"superkingdom", "domain"}  # both map to D__
_EVO2_PREFIXES = ["D", "P", "C", "O", "F", "G", "S"]


class NCBITaxonomy:
    """In-memory NCBI taxonomy tree for rank lookups."""

    def __init__(self) -> None:
        # taxid -> (parent_taxid, rank)
        self.nodes: dict[int, tuple[int, str]] = {}
        # taxid -> scientific name
        self.names: dict[int, str] = {}
        # scientific name (lowercase) -> taxid (first match)
        self.name_to_id: dict[str, int] = {}

    @classmethod
    def from_dump(cls, dump_dir: Path) -> "NCBITaxonomy":
        """Load from NCBI taxdump directory containing names.dmp and nodes.dmp."""
        tax = cls()
        nodes_path = dump_dir / "nodes.dmp"
        names_path = dump_dir / "names.dmp"

        # Parse nodes.dmp: taxid | parent_taxid | rank | ...
        with nodes_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t|\t")
                if len(parts) < 3:
                    continue
                taxid = int(parts[0].strip())
                parent = int(parts[1].strip())
                rank = parts[2].strip()
                tax.nodes[taxid] = (parent, rank)

        # Parse names.dmp: taxid | name | unique_name | name_class | ...
        # We only keep scientific names for the primary lookup.
        with names_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t|\t")
                if len(parts) < 4:
                    continue
                name_class = parts[3].strip().rstrip("\t|").strip()
                if name_class != "scientific name":
                    continue
                taxid = int(parts[0].strip())
                name = parts[1].strip()
                tax.names[taxid] = name
                key = name.lower()
                if key not in tax.name_to_id:
                    tax.name_to_id[key] = taxid

        return tax

    def lineage_ranks(self, taxid: int) -> dict[str, str]:
        """Walk tree to root, returning {rank: scientific_name} for Evo2 ranks."""
        result: dict[str, str] = {}
        evo2_rank_set = set(_EVO2_RANKS) | _DOMAIN_ALIASES
        current = taxid
        visited: set[int] = set()

        while current in self.nodes and current not in visited:
            visited.add(current)
            parent, rank = self.nodes[current]
            if rank in evo2_rank_set:
                # Normalise superkingdom → domain
                canonical = "domain" if rank in _DOMAIN_ALIASES else rank
                name = self.names.get(current, "")
                if name:
                    result[canonical] = name
            if current == 1 or current == parent:
                break
            current = parent

        return result

    def lookup_organism(self, organism_name: str) -> Optional[int]:
        """Find taxon ID by scientific name (case-insensitive, exact match)."""
        key = organism_name.strip().lower()
        # Try exact match first
        if key in self.name_to_id:
            return self.name_to_id[key]
        # Try stripping strain info (everything after last space if > 2 words)
        # e.g. "Micromonospora maris AB-18-032" -> "Micromonospora maris"
        parts = key.split()
        if len(parts) > 2:
            binomial = " ".join(parts[:2])
            if binomial in self.name_to_id:
                return self.name_to_id[binomial]
        # Try genus only
        if len(parts) >= 1:
            genus = parts[0]
            if genus in self.name_to_id:
                return self.name_to_id[genus]
        return None


# ---------------------------------------------------------------------------
# Singleton loader (avoid re-parsing the 400 MB dump on every call)
# ---------------------------------------------------------------------------

_TAXONOMY: Optional[NCBITaxonomy] = None


def load_taxonomy(dump_dir: Path) -> NCBITaxonomy:
    """Load or return cached NCBI taxonomy index."""
    global _TAXONOMY  # noqa: PLW0603
    if _TAXONOMY is None:
        _TAXONOMY = NCBITaxonomy.from_dump(dump_dir)
    return _TAXONOMY


# ---------------------------------------------------------------------------
# GenBank helpers
# ---------------------------------------------------------------------------


def extract_taxon_id(gbk_text: str) -> Optional[int]:
    """Extract NCBI taxon ID from /db_xref="taxon:XXXX" in GenBank text."""
    m = re.search(r'/db_xref="taxon:(\d+)"', gbk_text)
    if m:
        return int(m.group(1))
    return None


def extract_organism_name(gbk_text: str) -> Optional[str]:
    """Extract species name from ORGANISM line in GenBank text."""
    lines = gbk_text.splitlines()
    for line in lines:
        if line.strip().startswith("ORGANISM"):
            org = line.split("ORGANISM", 1)[1].strip()
            if org and org != ".":
                return org
    return None


# ---------------------------------------------------------------------------
# Legacy fallback: parse GenBank ORGANISM block directly (no NCBI lookup)
# ---------------------------------------------------------------------------


def _extract_organism_section(gbk_text: str) -> Optional[str]:
    """Return raw ORGANISM subsection text (species line + indented lineage)."""
    lines = gbk_text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("  ORGANISM"):
            chunk: list[str] = [line.split("ORGANISM", 1)[1].strip()]
            k = idx + 1
            while k < len(lines):
                ln = lines[k]
                if ln and not ln.startswith(" "):
                    break
                if ln.strip():
                    chunk.append(ln.strip())
                k += 1
            return "\n".join(chunk)
    return None


def _lineage_tokens(organism_section: str) -> tuple[str, list[str]]:
    """Species from first line; semicolon-separated lineage from the rest."""
    parts = organism_section.splitlines()
    if not parts:
        return "unknown", []
    species = parts[0].strip()
    rest = " ".join(p.strip() for p in parts[1:] if p.strip())
    if not rest:
        return species, []
    tokens = [t.strip().rstrip(".") for t in rest.split(";") if t.strip()]
    return species, tokens


def _build_tag_from_genbank_fallback(gbk_text: str) -> str:
    """Fallback tag builder that parses GenBank ORGANISM block directly.

    Less accurate for eukaryotes (intermediate ranks confuse positional mapping)
    but works when NCBI taxonomy is unavailable.
    """
    section = _extract_organism_section(gbk_text)
    if not section:
        return "|TAX:UNKNOWN|"

    species, lineage = _lineage_tokens(section)
    prefixes = ["D", "P", "C", "O", "F", "G"]
    segments: list[str] = []
    for i, tok in enumerate(lineage[: len(prefixes)]):
        safe = re.sub(r"[^A-Z0-9_]", "_", tok.strip().rstrip(".").upper())
        safe = re.sub(r"_+", "_", safe).strip("_")
        segments.append(f"{prefixes[i]}__{safe}")
    sp_safe = re.sub(r"[^A-Z0-9_]", "_", species.strip().rstrip(".").upper())
    sp_safe = re.sub(r"_+", "_", sp_safe).strip("_")
    segments.append(f"S__{sp_safe}")
    inner = ";".join(segments)
    return f"|{inner}|"


# ---------------------------------------------------------------------------
# Primary API
# ---------------------------------------------------------------------------


def build_taxonomic_tag(
    gbk_text: str,
    taxonomy: Optional[NCBITaxonomy] = None,
) -> str:
    """Build an Evo2-style pipe-delimited taxonomy tag.

    Uses NCBI taxonomy tree if available; falls back to GenBank parsing.
    Output is ALL UPPERCASE to match Evo2 pretraining format.
    """
    if taxonomy is not None:
        # Try taxon ID from GenBank first
        taxid = extract_taxon_id(gbk_text)
        if taxid is None:
            # Fall back to organism name lookup
            org = extract_organism_name(gbk_text)
            if org:
                taxid = taxonomy.lookup_organism(org)

        if taxid is not None:
            rank_map = taxonomy.lineage_ranks(taxid)
            if rank_map:
                segments: list[str] = []
                for rank, prefix in zip(_EVO2_RANKS, _EVO2_PREFIXES):
                    name = rank_map.get(rank, "")
                    if name:
                        safe = re.sub(r"[^A-Z0-9_]", "_", name.upper())
                        safe = re.sub(r"_+", "_", safe).strip("_")
                        segments.append(f"{prefix}__{safe}")
                if segments:
                    return f"|{';'.join(segments)}|"

    # Fallback: parse GenBank directly
    return _build_tag_from_genbank_fallback(gbk_text)


# ---------------------------------------------------------------------------
# Compound token normalisation (unchanged)
# ---------------------------------------------------------------------------


def normalize_compound_token(name: str) -> str:
    """MIBiG compound name -> token body (no outer pipes)."""
    n = (name or "").strip().lower()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^a-z0-9_+-]", "", n)
    return n or "unknown"
