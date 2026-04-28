#!/usr/bin/env python3
"""Quick data/eval readiness report for gputee production runs.

Checks the presence of key datasets/artifacts and selected external commands.
This does not execute expensive jobs; it only reports readiness state.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def status_for_path(path: Path, required: bool = True) -> dict[str, str]:
    exists = path.exists()
    return {
        "path": str(path),
        "required": "yes" if required else "optional",
        "status": "ready" if exists else "missing",
    }


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root path",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text",
    )
    args = p.parse_args()

    root = args.repo_root

    file_checks = {
        "combined_train": status_for_path(
            root / "data/processed/splits_combined/train.jsonl"
        ),
        "combined_val": status_for_path(
            root / "data/processed/splits_combined/val.jsonl"
        ),
        "combined_test": status_for_path(
            root / "data/processed/splits_combined/test.jsonl"
        ),
        "pfam_hmm": status_for_path(root / "data/pfam/Pfam-A.hmm"),
        "npatlas_json": status_for_path(
            root / "data/npatlas/NPAtlas_download.json", required=False
        ),
        "uniref50_db_dir": status_for_path(
            root / "data/uniref50/uniref50", required=False
        ),
        "antismash_db_v5_tar": status_for_path(
            root / "data/antismash_db/asdb5_gbks.tar", required=False
        ),
        "antismash_taxa_json": status_for_path(
            root / "data/antismash_db/asdb5_taxa.json.gz"
        ),
    }

    cmd_checks = {
        "download-antismash-databases": bool(
            shutil.which("download-antismash-databases")
        ),
        "mmseqs": bool(shutil.which("mmseqs")),
        "deepspeed": bool(shutil.which("deepspeed")),
        "python": bool(shutil.which("python")),
    }

    result = {
        "repo_root": str(root),
        "files": file_checks,
        "commands": cmd_checks,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("Data/Eval Readiness Report")
    print(f"repo_root: {root}")
    print("")
    print("Files:")
    for name, info in file_checks.items():
        print(
            f"  - {name:18s} [{info['status']:7s}]"
            f" required={info['required']:3s}  {info['path']}"
        )
    print("")
    print("Commands:")
    for name, ok in cmd_checks.items():
        print(f"  - {name:28s} {'ready' if ok else 'missing'}")


if __name__ == "__main__":
    main()
