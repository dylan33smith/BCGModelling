"""Build MIBiG training records (JSON metadata + conditioning text)."""

from __future__ import annotations

import io
import json
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterator

from Bio import SeqIO

from bgc_pipeline.class_map import map_mibig_class
from bgc_pipeline.taxonomy import (
    NCBITaxonomy,
    build_taxonomic_tag,
    normalize_compound_token,
)


@dataclass
class MibigTrainingRecord:
    accession: str
    compound_class: str
    compound_token: str
    compound_names_all: list[str]
    mibig_biosynthesis_classes: list[str]
    taxonomic_tag: str
    sequence: str
    training_text: str
    gbk_member: str


def _is_gbk_tarball(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith(".tar.gz") or name.endswith(".tgz"))


def _read_gbk_from_directory(gbk_dir: Path, accession: str) -> tuple[str, str] | None:
    """Return (raw_gbk_text, provenance label) or None if missing."""
    path = gbk_dir / f"{accession}.gbk"
    if not path.is_file():
        return None
    text = path.read_text(encoding="ascii", errors="replace")
    return text, path.name


def _read_gbk_from_open_tar(tf: tarfile.TarFile, accession: str) -> tuple[str, str] | None:
    """Return (raw_gbk_text, member_name) or None if missing."""
    inner = f"mibig_gbk_4.0/{accession}.gbk"
    try:
        member = tf.getmember(inner)
    except KeyError:
        return None
    raw = tf.extractfile(member)
    if raw is None:
        return None
    text = raw.read().decode("ascii", errors="replace")
    return text, inner


@contextmanager
def _gbk_read_fn(gbk_source: Path) -> Iterator[Callable[[str], tuple[str, str] | None]]:
    """
    Yield a callable(accession) -> (gbk_text, provenance) | None.
    gbk_source is either a directory of BGC*.gbk files or a MIBiG mibig_gbk_4.0.tar.gz.
    """
    if gbk_source.is_dir():
        yield lambda acc: _read_gbk_from_directory(gbk_source, acc)
    elif _is_gbk_tarball(gbk_source):
        with tarfile.open(gbk_source, "r:*") as tf:
            yield lambda acc: _read_gbk_from_open_tar(tf, acc)
    else:
        raise FileNotFoundError(
            f"GenBank source must be a directory of .gbk files or a .tar.gz archive; "
            f"not found or unsupported: {gbk_source}"
        )


def _sequence_from_gbk_text(gbk_text: str) -> str:
    handle = io.StringIO(gbk_text)
    record = next(SeqIO.parse(handle, "genbank"))
    return str(record.seq).upper()


def iter_mibig_records(
    json_dir: Path,
    gbk_source: Path,
    class_mapping: dict[str, str],
    class_default: str,
    limit: int | None = None,
    taxonomy: NCBITaxonomy | None = None,
) -> Iterator[MibigTrainingRecord]:
    paths = sorted(json_dir.glob("*.json"))
    yielded = 0
    with _gbk_read_fn(gbk_source) as read_gbk:
        for jpath in paths:
            if limit is not None and yielded >= limit:
                break
            data: dict[str, Any] = json.loads(jpath.read_text(encoding="utf-8"))
            accession = str(data.get("accession", jpath.stem))
            bios = data.get("biosynthesis") or {}
            classes = bios.get("classes") or []
            mibig_class_names = [
                str(c.get("class", "")).strip() for c in classes if c.get("class")
            ]
            if not mibig_class_names:
                continue
            primary_mibig_class = mibig_class_names[0]
            compound_class = map_mibig_class(primary_mibig_class, class_mapping, class_default)

            compounds = data.get("compounds") or []
            names = [str(c.get("name", "")).strip() for c in compounds if c.get("name")]
            if not names:
                continue
            primary_compound = names[0]
            compound_tok = normalize_compound_token(primary_compound)

            gbk_pair = read_gbk(accession)
            if gbk_pair is None:
                continue
            gbk_text, member = gbk_pair
            try:
                seq = _sequence_from_gbk_text(gbk_text)
            except StopIteration:
                continue
            if len(seq) < 30:
                continue

            tax_tag = build_taxonomic_tag(gbk_text, taxonomy=taxonomy)
            training_text = (
                f"|COMPOUND_CLASS:{compound_class}|"
                f"|COMPOUND:{compound_tok}|"
                f"{tax_tag}"
                f"{seq}"
            )

            yield MibigTrainingRecord(
                accession=accession,
                compound_class=compound_class,
                compound_token=compound_tok,
                compound_names_all=names,
                mibig_biosynthesis_classes=mibig_class_names,
                taxonomic_tag=tax_tag,
                sequence=seq,
                training_text=training_text,
                gbk_member=member,
            )
            yielded += 1


def record_to_json_dict(rec: MibigTrainingRecord) -> dict[str, Any]:
    return asdict(rec)
