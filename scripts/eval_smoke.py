#!/usr/bin/env python3
"""
Lightweight evaluation smoke test: sequence stats + optional DNA Chisel + optional antiSMASH.

Compare a real BGC sequence against a length-matched shuffled control. Expect similar GC
for the control; max homopolymer often differs. DNA Chisel is run on a short 5' slice
of each sequence by default so smoke tests stay fast on large BGCs.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if seq else 0.0


def max_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    best = cur = 1
    prev = seq[0].upper()
    for ch in seq[1:].upper():
        if ch == prev:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
            prev = ch
    return best


def shuffle_dna(seq: str, rng: random.Random) -> str:
    chars = list(seq.upper())
    rng.shuffle(chars)
    return "".join(chars)


def sequence_metrics(seq: str) -> dict[str, Any]:
    return {
        "length_bp": len(seq),
        "gc_fraction": round(gc_content(seq), 4),
        "max_homopolymer": max_homopolymer(seq),
    }


def dnachisel_summary(seq: str, max_slice: int) -> dict[str, Any] | None:
    try:
        from dnachisel import DnaOptimizationProblem
        from dnachisel.builtin_specifications import EnforceGCContent
    except ImportError:
        return None

    seq = seq.upper()
    if not re.fullmatch(r"[ACGT]+", seq):
        return {"error": "non-ACGT characters; skipped DNA Chisel"}

    if len(seq) > max_slice:
        work = seq[:max_slice]
        truncated = True
    else:
        work = seq
        truncated = False

    problem = DnaOptimizationProblem(
        sequence=work,
        constraints=[EnforceGCContent(mini=0.25, maxi=0.65)],
    )
    try:
        passes = problem.all_constraints_pass()
        summary = problem.constraints_text_summary()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {
        "gc_constraint_passes": passes,
        "constraints_text_summary": summary,
        "slice_bp": len(work),
        "truncated": truncated,
    }


def run_antismash(fasta_path: Path, out_dir: Path, timeout: int) -> dict[str, Any]:
    cmd = [
        "antismash",
        str(fasta_path),
        "--output-dir",
        str(out_dir),
        "--genefinding-tool",
        "prodigal",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"skipped": True, "reason": "antismash executable not found"}
    except subprocess.TimeoutExpired:
        return {"skipped": True, "reason": f"timeout after {timeout}s"}
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }


def load_sequences_from_jsonl(path: Path, max_records: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= max_records:
                break
            obj = json.loads(line)
            acc = obj.get("accession", "unknown")
            seq = obj.get("sequence", "")
            if seq:
                rows.append((str(acc), str(seq)))
    return rows


def load_sequences_from_fasta(path: Path, max_records: int) -> list[tuple[str, str]]:
    from Bio import SeqIO

    rows: list[tuple[str, str]] = []
    for record in SeqIO.parse(path.open(), "fasta"):
        if len(rows) >= max_records:
            break
        rows.append((record.id, str(record.seq)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path, help="JSONL from mibig_to_jsonl.py")
    src.add_argument("--fasta", type=Path, help="FASTA of nucleotide sequences")
    parser.add_argument("--max-sequences", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dnachisel-max-slice",
        type=int,
        default=8000,
        metavar="BP",
        help="Max length passed to DNA Chisel (5' slice; keeps smoke test fast)",
    )
    parser.add_argument(
        "--run-antismash",
        action="store_true",
        help="Run antiSMASH CLI if installed (slow; needs prodigal)",
    )
    parser.add_argument("--antismash-timeout", type=int, default=600)
    parser.add_argument(
        "--skip-dnachisel",
        action="store_true",
        help="Do not attempt DNA Chisel import",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.jsonl:
        pairs = load_sequences_from_jsonl(args.jsonl, args.max_sequences)
    else:
        pairs = load_sequences_from_fasta(args.fasta, args.max_sequences)

    if not pairs:
        print("No sequences loaded.", file=sys.stderr)
        sys.exit(1)

    report: dict[str, Any] = {"sequences": []}

    for acc, seq in pairs:
        shuf = shuffle_dna(seq, rng)
        entry: dict[str, Any] = {
            "accession": acc,
            "positive": sequence_metrics(seq),
            "negative_shuffled": sequence_metrics(shuf),
        }
        if not args.skip_dnachisel:
            entry["dnachisel_positive"] = dnachisel_summary(seq, args.dnachisel_max_slice)
            entry["dnachisel_negative"] = dnachisel_summary(shuf, args.dnachisel_max_slice)

        if args.run_antismash:
            with tempfile.TemporaryDirectory() as tmp:
                fasta = Path(tmp) / f"{acc}.fasta"
                fasta.write_text(f">{acc}\n{seq}\n", encoding="utf-8")
                outd = Path(tmp) / "as_out"
                entry["antismash"] = run_antismash(fasta, outd, args.antismash_timeout)

        report["sequences"].append(entry)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
