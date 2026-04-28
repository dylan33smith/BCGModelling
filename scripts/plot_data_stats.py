#!/usr/bin/env python3
"""Data analysis visualisations for the combined BGC training splits.

Streams `data/processed/splits_combined/{train,val,test}.jsonl` once to build
a per-record summary CSV at `data/processed/stats/record_summary.csv`, then
renders the requested plots as PNGs under the project-level `figures/`
directory.

See `docs/gputee/FINETUNE_GUIDE.md` §3 and `docs/gputee/PROJECT_GUIDE.md` §5
for the data schema and upstream preprocessing.

Usage
-----
    # Build cache (first run) and render every plot:
    python scripts/plot_data_stats.py --rebuild-cache

    # Re-render from the existing cache, a subset of plots:
    python scripts/plot_data_stats.py --plots len_hist,len_cdf,class_counts

    # Send figures somewhere else:
    python scripts/plot_data_stats.py --outdir /tmp/bgc_figs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "processed" / "splits_combined"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "processed" / "stats"
DEFAULT_FIGURES_DIR = REPO_ROOT / "figures"
CACHE_FILENAME = "record_summary.csv"

SPLITS: tuple[str, ...] = ("train", "val", "test")

CANDIDATE_L: list[int] = [
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]

LENGTH_BUCKET_EDGES: list[int] = [0, 5_000, 20_000, 50_000, 100_000, 262_144 + 1]
LENGTH_BUCKET_LABELS: list[str] = [
    "< 5 kb",
    "5–20 kb",
    "20–50 kb",
    "50–100 kb",
    "100–262 kb",
]

TAX_PREFIX_TO_COL: dict[str, str] = {
    "D__": "tax_domain",
    "P__": "tax_phylum",
    "C__": "tax_class",
    "O__": "tax_order",
    "F__": "tax_family",
    "G__": "tax_genus",
    "S__": "tax_species",
}
TAX_COLUMNS: list[str] = list(TAX_PREFIX_TO_COL.values())

CACHE_COLUMNS: list[str] = [
    "split",
    "source",
    "accession",
    "compound_class",
    "length",
    "gc",
    "n_frac",
    "contig_edge",
    *TAX_COLUMNS,
]


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield one parsed record per line from a JSONL file."""
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_taxonomy(tag: str | None) -> dict[str, str]:
    """Parse an Evo2-style taxonomic tag into per-rank columns.

    The tag looks like
        |D__BACTERIA;P__ACTINOMYCETOTA;C__ACTINOMYCETES;...|
    Returns a dict keyed by `tax_domain`, `tax_phylum`, ... with any missing
    rank stored as the empty string.
    """
    result: dict[str, str] = {col: "" for col in TAX_COLUMNS}
    if not tag:
        return result
    inner = tag.strip().strip("|")
    for token in inner.split(";"):
        token = token.strip()
        for prefix, col in TAX_PREFIX_TO_COL.items():
            if token.startswith(prefix):
                result[col] = token[len(prefix) :]
                break
    return result


def sequence_stats(sequence: str) -> tuple[int, float, float]:
    """Return (length, gc_fraction, non_acgt_fraction) for a DNA sequence.

    `gc_fraction` is computed over (A+C+G+T) only — i.e. N and other
    non-standard characters are excluded from the denominator so that the
    GC number is comparable across records with different N content.
    `non_acgt_fraction` is the complementary non-ACGT fraction over the
    full sequence length.
    """
    length = len(sequence)
    if length == 0:
        return 0, 0.0, 0.0
    counts = Counter(sequence.upper())
    a = counts.get("A", 0)
    c = counts.get("C", 0)
    g = counts.get("G", 0)
    t = counts.get("T", 0)
    acgt = a + c + g + t
    gc = (g + c) / acgt if acgt > 0 else 0.0
    non_acgt = (length - acgt) / length
    return length, gc, non_acgt


def detect_source(accession: str, record: dict) -> str:
    """Return 'mibig' or 'asdb' for a combined-split record.

    MIBiG accessions start with `BGC`; antiSMASH records have a
    `{genome_accession}.region{N}` shape. The absence of `contig_edge` is a
    secondary signal — it is not written by the MIBiG preprocessing step.
    """
    if accession.startswith("BGC"):
        return "mibig"
    if "contig_edge" not in record:
        return "mibig"
    return "asdb"


def record_to_row(record: dict, split: str) -> dict:
    """Extract the slim per-record summary we cache to CSV."""
    accession = record.get("accession", "")
    source = detect_source(accession, record)
    length, gc, n_frac = sequence_stats(record.get("sequence", ""))
    taxonomy = parse_taxonomy(record.get("taxonomic_tag"))

    contig_edge_raw = record.get("contig_edge", None)
    if contig_edge_raw is None:
        contig_edge_str = ""
    else:
        contig_edge_str = "1" if contig_edge_raw else "0"

    row = {
        "split": split,
        "source": source,
        "accession": accession,
        "compound_class": record.get("compound_class", ""),
        "length": length,
        "gc": f"{gc:.6f}",
        "n_frac": f"{n_frac:.6f}",
        "contig_edge": contig_edge_str,
    }
    row.update(taxonomy)
    return row


def build_cache(splits_dir: Path, cache_path: Path) -> None:
    """Stream all three split files once and write the summary CSV."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cache] writing {cache_path}")

    with cache_path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CACHE_COLUMNS)
        writer.writeheader()

        for split in SPLITS:
            split_path = splits_dir / f"{split}.jsonl"
            if not split_path.exists():
                print(f"[cache]   WARNING: {split_path} not found, skipping")
                continue
            print(f"[cache]   reading {split_path}")
            n = 0
            for rec in iter_jsonl(split_path):
                writer.writerow(record_to_row(rec, split))
                n += 1
                if n % 50_000 == 0:
                    print(f"[cache]     {split}: {n:,} records")
            print(f"[cache]   {split}: {n:,} records total")

    print("[cache] done")


def load_cache(cache_path: Path) -> pd.DataFrame:
    """Read the cached CSV into a DataFrame with sensible dtypes."""
    print(f"[cache] loading {cache_path}")
    df = pd.read_csv(
        cache_path,
        dtype={
            "split": "category",
            "source": "category",
            "accession": "string",
            "compound_class": "category",
            "length": "int64",
            "gc": "float64",
            "n_frac": "float64",
            **{col: "category" for col in TAX_COLUMNS},
        },
    )
    df["contig_edge"] = df["contig_edge"].map(
        {1: True, 0: False, "1": True, "0": False}
    )
    print(f"[cache] loaded {len(df):,} records")
    return df


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

L_TARGET = 32_768


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def _annotate_l_lines(ax: plt.Axes, ylim: tuple[float, float] | None = None) -> None:
    """Draw vertical guide lines at the candidate L values with a highlight at L=32k."""
    for L in CANDIDATE_L:
        if L == L_TARGET:
            ax.axvline(L, color="red", linewidth=1.2, linestyle="--", zorder=1)
        else:
            ax.axvline(L, color="grey", linewidth=0.6, linestyle=":", alpha=0.6, zorder=1)
    if ylim is not None:
        top = ylim[1]
        for L in CANDIDATE_L:
            label = f"{L // 1024}k" if L >= 1024 else str(L)
            ax.text(
                L, top, label,
                rotation=90, va="top", ha="right",
                fontsize=7, color="red" if L == L_TARGET else "grey",
            )


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_len_hist(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 1: sequence length histogram (log x, log y), pooled + per split."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    bins = np.logspace(np.log10(100), np.log10(300_000), 60)

    panels = [
        ("All records", df),
        ("train", df[df["split"] == "train"]),
        ("val", df[df["split"] == "val"]),
        ("test", df[df["split"] == "test"]),
    ]
    for ax, (title, sub) in zip(axes.flat, panels):
        ax.hist(sub["length"], bins=bins, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{title}  (n = {len(sub):,})")
        ax.set_xlabel("Sequence length (bp)")
        ax.set_ylabel("Records")
        ax.grid(True, which="both", alpha=0.3)
        ylim = ax.get_ylim()
        _annotate_l_lines(ax, ylim=ylim)
        ax.set_ylim(ylim)

    fig.suptitle("Sequence length distribution (log–log)", fontsize=14)
    _save(fig, outdir, "1_length_histogram")


def plot_len_cdf(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 2: sequence length CDF with %-covered annotations at candidate L."""
    lengths = np.sort(df["length"].to_numpy())
    n = len(lengths)
    y = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lengths, y, color="steelblue", linewidth=1.5, label="All records")
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length L (bp)")
    ax.set_ylabel("Fraction of records with full_length ≤ L")
    ax.set_title("Sequence length CDF — % of records fully covered at each candidate L")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.02)

    rows = []
    for L in CANDIDATE_L:
        frac = float(np.searchsorted(lengths, L, side="right")) / n
        rows.append((L, frac))
        color = "red" if L == L_TARGET else "grey"
        linestyle = "--" if L == L_TARGET else ":"
        ax.axvline(L, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.7)
        ax.plot([L], [frac], marker="o", color=color)
        label = f"L={L//1024}k → {frac*100:.1f}%" if L >= 1024 else f"L={L} → {frac*100:.1f}%"
        ax.annotate(
            label, xy=(L, frac), xytext=(5, -10), textcoords="offset points",
            fontsize=8, color=color,
        )

    _save(fig, outdir, "2_length_cdf")


def plot_bp_discarded(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 3: fraction of bp discarded if truncating each record to L."""
    lengths = df["length"].to_numpy()
    total_bp = float(lengths.sum())

    xs = np.array(CANDIDATE_L, dtype=float)
    frac_bp_discarded = np.zeros_like(xs)
    frac_records_cropped = np.zeros_like(xs)
    for i, L in enumerate(CANDIDATE_L):
        overflow = np.clip(lengths - L, 0, None)
        frac_bp_discarded[i] = overflow.sum() / total_bp
        frac_records_cropped[i] = float((lengths > L).sum()) / len(lengths)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, frac_bp_discarded * 100, marker="o", color="crimson", label="% bp discarded")
    ax.plot(xs, frac_records_cropped * 100, marker="s", color="steelblue", label="% records cropped")
    ax.set_xscale("log")
    ax.set_xlabel("Training cap L (bp)")
    ax.set_ylabel("% of data affected by the cap")
    ax.set_title("Cost of capping training at L — how much sequence is thrown away")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(L_TARGET, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

    for L, frac_bp, frac_rec in zip(xs, frac_bp_discarded, frac_records_cropped):
        ax.annotate(
            f"{frac_bp*100:.1f}%", xy=(L, frac_bp * 100), xytext=(0, 6),
            textcoords="offset points", fontsize=8, color="crimson", ha="center",
        )
        ax.annotate(
            f"{frac_rec*100:.1f}%", xy=(L, frac_rec * 100), xytext=(0, -12),
            textcoords="offset points", fontsize=8, color="steelblue", ha="center",
        )
    ax.legend()
    _save(fig, outdir, "3_bp_discarded_by_L")


def plot_len_by_class(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 4: sequence length box plot per compound_class (log y)."""
    classes = sorted(df["compound_class"].dropna().unique().tolist())
    data = [df.loc[df["compound_class"] == cls, "length"].to_numpy() for cls in classes]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(classes) + 4), 6))
    bp = ax.boxplot(data, tick_labels=classes, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")

    ax.set_yscale("log")
    ax.set_ylabel("Sequence length (bp, log)")
    ax.set_xlabel("Compound class")
    ax.set_title("Sequence length by compound class (outliers hidden; log-y)")
    ax.axhline(L_TARGET, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"L = {L_TARGET:,}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    for i, n in enumerate(counts, start=1):
        ax.text(i, ax.get_ylim()[0] * 1.2, f"n={n:,}", ha="center", va="bottom",
                fontsize=7, color="dimgray")

    _save(fig, outdir, "4_length_by_class")


def plot_class_counts(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 6: compound_class record counts, horizontal bar, log x."""
    counts = df["compound_class"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("Record count (log)")
    ax.set_title("Compound class distribution — all records (train+val+test)")
    ax.grid(True, which="both", axis="x", alpha=0.3)
    for i, v in enumerate(counts.values):
        ax.text(v * 1.02, i, f"{v:,}", va="center", fontsize=8)
    _save(fig, outdir, "6_class_counts")


def plot_class_counts_by_source(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 7: compound_class counts stacked by MIBiG vs antiSMASH."""
    pivot = (
        df.groupby(["compound_class", "source"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_values(by=list(df["source"].cat.categories), ascending=True)
    )
    for col in ("mibig", "asdb"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["asdb", "mibig"]]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(pivot))))
    ax.barh(pivot.index.astype(str), pivot["asdb"], color="steelblue", label="antiSMASH")
    ax.barh(
        pivot.index.astype(str), pivot["mibig"], left=pivot["asdb"],
        color="orange", label="MIBiG",
    )
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Record count (symlog)")
    ax.set_title("Compound class distribution by source")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", axis="x", alpha=0.3)
    for i, (asdb_v, mibig_v) in enumerate(zip(pivot["asdb"], pivot["mibig"])):
        total = asdb_v + mibig_v
        ax.text(max(total, 1) * 1.02, i,
                f"{total:,} ({mibig_v:,} MIBiG)",
                va="center", fontsize=8)
    _save(fig, outdir, "7_class_counts_by_source")


def plot_class_length_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 9: class × length-bucket heatmap (log colour)."""
    bucket_index = pd.cut(
        df["length"], bins=LENGTH_BUCKET_EDGES, right=False,
        labels=LENGTH_BUCKET_LABELS, include_lowest=True,
    )
    tab = pd.crosstab(df["compound_class"], bucket_index)
    tab = tab.reindex(columns=LENGTH_BUCKET_LABELS, fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(tab))))
    data = tab.to_numpy(dtype=float)
    data_for_log = np.where(data > 0, data, np.nan)
    im = ax.imshow(
        np.log10(data_for_log), aspect="auto", cmap="viridis", origin="upper",
    )
    ax.set_xticks(range(len(tab.columns)))
    ax.set_xticklabels(tab.columns)
    ax.set_yticks(range(len(tab.index)))
    ax.set_yticklabels(tab.index.astype(str))
    ax.set_title("Compound class × sequence-length bucket (cell = record count)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(records)")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            if v > 0:
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        fontsize=7, color="white")

    _save(fig, outdir, "9_class_length_heatmap")


def plot_top_phyla(df: pd.DataFrame, outdir: Path, n: int = 20) -> None:
    """Plot 10: top-N phyla by record count."""
    counts = (
        df["tax_phylum"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(n)
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color="seagreen")
    ax.set_xscale("log")
    ax.set_xlabel("Record count (log)")
    ax.set_title(f"Top-{n} phyla by record count")
    ax.grid(True, which="both", axis="x", alpha=0.3)
    for i, v in enumerate(counts.values):
        ax.text(v * 1.02, i, f"{v:,}", va="center", fontsize=8)
    _save(fig, outdir, "10_top_phyla")


def plot_class_phylum_heatmap(df: pd.DataFrame, outdir: Path, top_n: int = 10) -> None:
    """Plot 12: compound_class × top-N phyla heatmap (log colour)."""
    top_phyla = (
        df["tax_phylum"].replace("", pd.NA).dropna().value_counts().head(top_n).index.tolist()
    )
    sub = df[df["tax_phylum"].isin(top_phyla)]
    tab = pd.crosstab(sub["compound_class"], sub["tax_phylum"])
    tab = tab.reindex(columns=top_phyla, fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(top_phyla) + 4), max(4, 0.35 * len(tab))))
    data = tab.to_numpy(dtype=float)
    data_for_log = np.where(data > 0, data, np.nan)
    im = ax.imshow(
        np.log10(data_for_log), aspect="auto", cmap="magma", origin="upper",
    )
    ax.set_xticks(range(len(tab.columns)))
    ax.set_xticklabels(tab.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(tab.index)))
    ax.set_yticklabels(tab.index.astype(str))
    ax.set_title(f"Compound class × top-{top_n} phyla (cell = record count)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(records)")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            if v > 0:
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        fontsize=7, color="white")

    _save(fig, outdir, "12_class_phylum_heatmap")


def plot_gc_histogram(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 13: GC content — overall histogram + by-class violins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(df["gc"].to_numpy(), bins=80, color="steelblue",
                 edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("GC content (over ACGT)")
    axes[0].set_ylabel("Records")
    axes[0].set_title("GC content — all records")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].axvline(0.51, color="red", linestyle="--", linewidth=1.0,
                    alpha=0.7, label="E. coli K-12 ≈ 0.51")
    axes[0].legend()

    classes = sorted(df["compound_class"].dropna().unique().tolist())
    data = [df.loc[df["compound_class"] == cls, "gc"].to_numpy() for cls in classes]
    axes[1].boxplot(data, tick_labels=classes, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue"))
    axes[1].set_ylabel("GC content (over ACGT)")
    axes[1].set_xlabel("Compound class")
    axes[1].set_title("GC content by compound class")
    axes[1].axhline(0.51, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.setp(axes[1].get_xticklabels(), rotation=40, ha="right")

    _save(fig, outdir, "13_gc_content")


def plot_contig_edge_by_class(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 15: contig_edge rate by compound_class (antiSMASH-only)."""
    asdb = df[df["source"] == "asdb"].copy()
    asdb = asdb[asdb["contig_edge"].notna()]
    if asdb.empty:
        print("[plot] skipping contig_edge_by_class — no antiSMASH rows with contig_edge")
        return

    grp = asdb.groupby("compound_class", observed=True)["contig_edge"]
    rates = grp.mean()
    counts = grp.size()
    order = rates.sort_values(ascending=True).index

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(rates))))
    ax.barh([str(x) for x in order], rates.loc[order].to_numpy() * 100, color="indianred")
    ax.set_xlabel("contig_edge rate (%) — antiSMASH records only")
    ax.set_title("contig_edge rate by compound class (antiSMASH only)")
    overall = float(asdb["contig_edge"].mean())
    ax.axvline(overall * 100, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"overall = {overall*100:.1f}%")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    for i, cls in enumerate(order):
        rate = rates.loc[cls] * 100
        n = counts.loc[cls]
        ax.text(rate + 0.3, i, f"{rate:.1f}%  (n={n:,})", va="center", fontsize=8)

    _save(fig, outdir, "15_contig_edge_by_class")


def plot_class_across_splits(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 17: class distribution across train/val/test (stratification check)."""
    tab = pd.crosstab(df["compound_class"], df["split"], normalize="columns") * 100
    for s in SPLITS:
        if s not in tab.columns:
            tab[s] = 0.0
    tab = tab[list(SPLITS)]
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]

    x = np.arange(len(tab.index))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(tab) + 4), 6))
    for i, split in enumerate(SPLITS):
        ax.bar(x + (i - 1) * width, tab[split].to_numpy(), width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index.astype(str), rotation=40, ha="right")
    ax.set_ylabel("% of split")
    ax.set_title("Compound class distribution across train / val / test (stratification check)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, outdir, "17_class_across_splits")


def plot_length_cdf_across_splits(df: pd.DataFrame, outdir: Path) -> None:
    """Plot 18: sequence length CDF overlaid across train/val/test."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"train": "steelblue", "val": "darkorange", "test": "seagreen"}
    for split in SPLITS:
        lengths = np.sort(df.loc[df["split"] == split, "length"].to_numpy())
        if len(lengths) == 0:
            continue
        y = np.arange(1, len(lengths) + 1) / len(lengths)
        ax.plot(lengths, y, label=f"{split} (n={len(lengths):,})",
                color=colors.get(split, None), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Sequence length (bp, log)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Sequence length CDF across splits (should overlap tightly)")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(L_TARGET, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"L = {L_TARGET:,}")
    ax.legend()
    _save(fig, outdir, "18_length_cdf_across_splits")


PLOT_REGISTRY: dict[str, tuple[str, Callable[[pd.DataFrame, Path], None]]] = {
    "len_hist":                ("1  Sequence length histogram (log–log, per split)", plot_len_hist),
    "len_cdf":                 ("2  Sequence length CDF with L annotations", plot_len_cdf),
    "bp_discarded":            ("3  Fraction of bp/records discarded at each candidate L", plot_bp_discarded),
    "len_by_class":            ("4  Sequence length boxplot by compound class", plot_len_by_class),
    "class_counts":            ("6  Compound class record counts (log x)", plot_class_counts),
    "class_counts_by_source":  ("7  Compound class counts stacked by MIBiG/antiSMASH", plot_class_counts_by_source),
    "class_length_heatmap":    ("9  Compound class × length-bucket heatmap", plot_class_length_heatmap),
    "top_phyla":               ("10 Top-20 phyla by record count", plot_top_phyla),
    "class_phylum_heatmap":    ("12 Compound class × top-10 phyla heatmap", plot_class_phylum_heatmap),
    "gc_histogram":            ("13 GC content overall + by class", plot_gc_histogram),
    "contig_edge_by_class":    ("15 contig_edge rate by compound class (antiSMASH only)", plot_contig_edge_by_class),
    "class_across_splits":     ("17 Class distribution across train/val/test", plot_class_across_splits),
    "length_cdf_across_splits":("18 Length CDF overlaid across train/val/test", plot_length_cdf_across_splits),
}


def parse_plot_selection(spec: str) -> list[str]:
    if spec == "all":
        return list(PLOT_REGISTRY.keys())
    requested = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [name for name in requested if name not in PLOT_REGISTRY]
    if unknown:
        raise SystemExit(
            f"unknown plot name(s): {unknown}\n"
            f"available: {', '.join(PLOT_REGISTRY.keys())}"
        )
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR,
        help=f"directory containing train/val/test.jsonl (default: {DEFAULT_SPLITS_DIR})",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help=f"directory to read/write record_summary.csv (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_FIGURES_DIR,
        help=f"directory to write PNG figures (default: {DEFAULT_FIGURES_DIR})",
    )
    parser.add_argument(
        "--plots", type=str, default="all",
        help="comma-separated plot names, or 'all' (default). "
             "Use --list to see the full registry.",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="force re-streaming of the JSONL splits even if the cache exists",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list the available plot names and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print("Available plots:")
        for name, (desc, _) in PLOT_REGISTRY.items():
            print(f"  {name:28s}  {desc}")
        return 0

    cache_path = args.cache_dir / CACHE_FILENAME
    if args.rebuild_cache or not cache_path.exists():
        build_cache(args.splits_dir, cache_path)
    else:
        print(f"[cache] re-using existing cache at {cache_path} (pass --rebuild-cache to rebuild)")

    df = load_cache(cache_path)
    if df.empty:
        print("[error] cache is empty, nothing to plot", file=sys.stderr)
        return 1

    selected = parse_plot_selection(args.plots)
    print(f"[plot] rendering {len(selected)} plot(s) to {args.outdir}")
    for name in selected:
        desc, fn = PLOT_REGISTRY[name]
        print(f"[plot] {name}: {desc}")
        fn(df, args.outdir)

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
