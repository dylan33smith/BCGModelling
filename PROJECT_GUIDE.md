# BGC Modelling — Project Guide

*Living document — last updated 2026-04-07*

This document is the single reference for understanding, running, and extending the
de novo BGC generation pipeline built on Evo2. It covers what has been built, how
to reproduce every step, and what remains to be done.

---

## 1  Project Goal

Fine-tune **Evo2 7B** (StripedHyena 2, 262 k context window) to generate
synthesis-ready biosynthetic gene cluster (BGC) nucleotide sequences conditioned on:


| Token            | Example                             | Source                                            |
| ---------------- | ----------------------------------- | ------------------------------------------------- |
| `COMPOUND_CLASS` | `PKS`, `NRPS`, `TERPENE`            | Harmonised vocabulary shared by MIBiG + antiSMASH |
| `COMPOUND`       | `indigoidine`, `violacein`          | MIBiG compound name (normalised)                  |
| Taxonomic tag    | `|D__BACTERIA;P__…;S__ESCHERICHIA|` | Evo2 pretraining format, ALL UPPERCASE            |


The pipeline validates generated sequences with an **eight-metric computational
evaluation suite** before any wet-lab work.

---

## 2  Repository Layout

```
BCGModelling/
├── BGC_Research_Plan.md            # Full research plan (v6, 11 sections)
├── PROJECT_GUIDE.md                # ← you are here
├── environment.yml                 # Conda env definition
├── requirements.txt                # pip-only fallback
├── LICENSE                         # MIT
│
├── config/
│   └── compound_class_map.yaml     # 60+ antiSMASH/MIBiG product types → harmonised tokens
│
├── src/bgc_pipeline/
│   ├── __init__.py                 # Package (v0.1.0)
│   ├── class_map.py                # Load & apply YAML class map
│   ├── taxonomy.py                 # NCBI taxdump → Evo2 taxonomic tags
│   ├── mibig_record.py             # MIBiG JSON+GBK → training records
│   └── evaluation.py               # Eight-metric evaluation suite
│
├── scripts/
│   ├── mibig_to_jsonl.py           # Step 1 — MIBiG → JSONL
│   ├── split_dataset.py            # Step 2 — stratified train/val/test
│   ├── evaluate_bgc.py             # Step 3 — full evaluation CLI
│   └── eval_smoke.py               # Quick sanity checks
│
└── data/
    ├── mibig/
    │   ├── mibig_json_4.0/         # 3,013 JSON metadata files
    │   ├── mibig_gbk_4.0/          # ~2,900 GenBank files
    │   ├── mibig_json_4.0.tar.gz   # 9.6 MB archive
    │   ├── mibig_gbk_4.0.tar.gz    # 80 MB archive
    │   └── mibig_prot_seqs_4.0.fasta
    ├── ncbi_taxonomy/
    │   ├── names.dmp               # 266 MB — taxon ID ↔ names
    │   ├── nodes.dmp               # 198 MB — tree structure + ranks
    │   └── taxdump.tar.gz
    ├── npatlas/
    │   └── NPAtlas_download.json   # 36,454 compounds (454 MB)
    ├── pfam/
    │   └── Pfam-A.hmm              # Pfam 37.0 — 21,979 families (1.6 GB)
    ├── antismash_db/               # Populated by download-antismash-databases
    └── processed/
        ├── mibig_train_records.jsonl   # 2,636 records (full preprocessed)
        └── splits/
            ├── train.jsonl             # 2,099 records
            ├── val.jsonl               #   263 records
            ├── test.jsonl              #   263 records
            └── heldout_accessions.txt  #   526 accessions (val + test)
```

---

## 3  Environment Setup

### 3.1  Create the conda environment

```bash
conda env create -f environment.yml
conda activate bgcmodel
```

The conda env includes: antiSMASH 8.0.4, pyhmmer, Biopython, PyYAML,
DNA Chisel, BiG-SCAPE 2.0, MMseqs2, Foldseek, and Prodigal.
Python version is solver-determined (currently **3.12.13**).

### 3.2  Download antiSMASH reference databases

```bash
download-antismash-databases    # ~15 GB, takes 10–30 min
```

### 3.3  GPU stack

The GPU tools are installed via pip **after** conda env creation. Version pins
are critical — see `requirements.txt` for the full rationale.

```bash
conda activate bgcmodel

# 1. PyTorch 2.5.1 with CUDA 12.4
#    (2.5.1 is required — PyTorch 2.6 changed the c10::Error ABI,
#     which breaks flash-attn compilation)
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 2. flash-attn (build from source against the installed torch)
pip install flash-attn==2.7.4.post1 --no-build-isolation

# 3. HuggingFace transformers 4.46.3
#    (4.46.x is required — transformers 5.x added a torch>=2.6 guard that
#     blocks loading .pt ESMFold weights on PyTorch 2.5)
pip install transformers==4.46.3 accelerate==1.13.0

# 4. Evo2 (Arc Institute PyPI package, loads evo2_7b_262k from HuggingFace)
pip install evo2==0.5.5
```

**Verified GPU setup on this server:**

- 4× NVIDIA A40 (46 GB VRAM each); GPUs 2 and 3 are free
- `CUDA_VISIBLE_DEVICES=2` recommended for single-GPU inference
- Evo2 7B checkpoint (~~14 GB) downloads automatically to `~~/.cache/huggingface/`
- ESMFold 3B checkpoint (~3 GB) downloads automatically on first use

### 3.4  UniRef50 for MMseqs2 (Metric 8)

```bash
# Already downloaded — 29 GB at data/uniref50/
mmseqs databases UniRef50 data/uniref50/uniref50 tmp/   # only needed to re-download
```

> **Important:** All scripts must be run with `conda activate bgcmodel` and
> `PYTHONPATH=src` (or from the repo root with `python -m`).

---

## 4  Data Acquisition

All data is stored under `data/` and excluded from git (`.gitignore`).

### 4.1  What was downloaded


| Dataset             | Version  | Files                              | Size   | Status                                    |
| ------------------- | -------- | ---------------------------------- | ------ | ----------------------------------------- |
| MIBiG JSON          | 4.0      | 3,013 JSON files                   | 9.6 MB | ✅ Downloaded & extracted                  |
| MIBiG GBK           | 4.0      | ~2,900 GenBank files               | 80 MB  | ✅ Downloaded & extracted                  |
| MIBiG protein seqs  | 4.0      | 1 FASTA                            | 31 MB  | ✅ Downloaded                              |
| NPAtlas             | 3.0      | 1 JSON (36,454 compounds)          | 454 MB | ✅ Downloaded                              |
| Pfam-A.hmm          | 37.0     | 1 HMM file (21,979 families)       | 1.6 GB | ✅ Downloaded                              |
| NCBI Taxonomy       | Apr 2026 | names.dmp + nodes.dmp              | 464 MB | ✅ Downloaded & extracted                  |
| antiSMASH ref DBs   | —        | via `download-antismash-databases` | ~15 GB | ✅ Downloaded                              |
| **antiSMASH DB v4** | **v4**   | **231,534 BGCs**                   | **—**  | **⏳ Blocked** — bulk download unavailable |
| UniRef50            | —        | MMseqs2 DB                         | ~30 GB | ❌ Not yet downloaded                      |


### 4.2  Download sources

```bash
# MIBiG 4.0 (from Zenodo/MIBiG)
wget -O data/mibig/mibig_json_4.0.tar.gz \
  "https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz"
wget -O data/mibig/mibig_gbk_4.0.tar.gz \
  "https://dl.secondarymetabolites.org/mibig/mibig_gbk_4.0.tar.gz"
wget -O data/mibig/mibig_prot_seqs_4.0.fasta \
  "https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_4.0.fasta"

# NPAtlas
wget -O data/npatlas/NPAtlas_download.json \
  "https://www.npatlas.org/api/v1/compounds/full"

# Pfam 37.0
wget -O data/pfam/Pfam-A.hmm.gz \
  "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam-A.hmm.gz"
gunzip data/pfam/Pfam-A.hmm.gz

# NCBI Taxonomy
wget -O data/ncbi_taxonomy/taxdump.tar.gz \
  "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
tar -xzf data/ncbi_taxonomy/taxdump.tar.gz -C data/ncbi_taxonomy/

# antiSMASH DB v5
wget -c https://dl.secondarymetabolites.org/database/5.0/asdb5_gbks.tar

wget -c https://dl.secondarymetabolites.org/database/5.0/asdb5_taxa.json.gz
```



### 4.3  antiSMASH DB v4 — blocked

Bulk download files are missing from the antiSMASH download server (confirmed
via GitHub discussion). The compound class map (`config/compound_class_map.yaml`)
already has 60+ antiSMASH product type mappings ready. When data becomes available:

1. Download GenBank/FASTA bulk exports
2. Run through the same preprocessing as MIBiG (but with `COMPOUND_CLASS` only, no `COMPOUND` token)
3. Deduplicate against heldout MIBiG val/test accessions (see Section 4.4 of research plan)

---

## 5  Data Pipeline — Step by Step

All commands assume you are in the repo root with `bgcmodel` activated.

### Step 1: Convert MIBiG to JSONL

```bash
PYTHONPATH=src python scripts/mibig_to_jsonl.py \
  --mibig-json-dir data/mibig/mibig_json_4.0 \
  --mibig-gbk      data/mibig/mibig_gbk_4.0 \
  --class-map       config/compound_class_map.yaml \
  --taxonomy-dir    data/ncbi_taxonomy \
  -o data/processed/mibig_train_records.jsonl
```

**What it does:**

- Reads 3,013 MIBiG JSON files for metadata (compound names, biosynthesis class)
- Matches each to its GenBank file for the nucleotide sequence
- Looks up organism taxonomy in NCBI taxdump → ALL UPPERCASE Evo2 tag
- Maps compound class through `compound_class_map.yaml`
- Normalises compound name (lowercase, underscores, alphanumeric)

**Output:** 2,636 JSONL records (377 filtered: 27 no compound name, 350 no matching GBK)

**Record format (one JSON object per line):**

```json
{
  "accession": "BGC0001386",
  "compound_class": "PKS",
  "compound_token": "jbir-76",
  "compound_names_all": ["JBIR-76", "JBIR-77"],
  "mibig_biosynthesis_classes": ["PKS"],
  "taxonomic_tag": "|D__BACTERIA;P__ACTINOMYCETOTA;...;S__STREPTOMYCES_SP_RI_77|",
  "sequence": "GCGTCGGCCAGG...",
  "training_text": "|COMPOUND_CLASS:PKS||COMPOUND:jbir-76||D__BACTERIA;...|GCGTCGGCCAGG...",
  "gbk_member": "mibig_gbk_4.0/BGC0001386.gbk"
}
```

The `training_text` field is the exact string fed to Evo2 during fine-tuning.

### Step 2: Split into train / val / test

```bash
PYTHONPATH=src python scripts/split_dataset.py \
  --input     data/processed/mibig_train_records.jsonl \
  --output-dir data/processed/splits \
  --seed 42 \
  --max-seq-len 262144
```

**What it does:**

- Filters out 11 sequences exceeding the 262,144 bp Evo2 context window
- Stratifies by `compound_class` (80% train / 10% val / 10% test)
- Writes `heldout_accessions.txt` listing val + test accessions for antiSMASH deduplication

**Output:**


| Split   | Records | File                                           |
| ------- | ------- | ---------------------------------------------- |
| Train   | 2,099   | `data/processed/splits/train.jsonl`            |
| Val     | 263     | `data/processed/splits/val.jsonl`              |
| Test    | 263     | `data/processed/splits/test.jsonl`             |
| Heldout | 526     | `data/processed/splits/heldout_accessions.txt` |


**Class distribution (identical proportions across splits):**


| Class      | Train | Val | Test |
| ---------- | ----- | --- | ---- |
| NRPS       | 700   | 87  | 87   |
| PKS        | 589   | 74  | 74   |
| RIPP       | 286   | 36  | 36   |
| OTHER      | 279   | 35  | 35   |
| TERPENE    | 134   | 17  | 17   |
| SACCHARIDE | 111   | 14  | 14   |


**Verified properties:**

- Zero accession overlap between any pair of splits
- `heldout_accessions.txt` exactly equals val ∪ test accessions

### Step 3: Evaluate sequences

```bash
# Quick: metrics 2, 4, 7 only (no GPU, no external tools)
PYTHONPATH=src python scripts/evaluate_bgc.py \
  --jsonl data/processed/splits/test.jsonl \
  --max-sequences 3 \
  --skip-metrics 1 3 5 6 8 \
  --pfam-hmm data/pfam/Pfam-A.hmm

# With shuffled negative controls
PYTHONPATH=src python scripts/evaluate_bgc.py \
  --jsonl data/processed/splits/test.jsonl \
  --max-sequences 3 \
  --include-negative-control \
  --skip-metrics 1 3 5 6 8 \
  --pfam-hmm data/pfam/Pfam-A.hmm

# Full evaluation on a generated FASTA
PYTHONPATH=src python scripts/evaluate_bgc.py \
  --fasta generated_bgcs.fasta \
  --expected-class PKS \
  --pfam-hmm data/pfam/Pfam-A.hmm \
  --mibig-gbk-dir data/mibig/mibig_gbk_4.0 \
  -o eval_results.json
```

See Section 7 for metric details.

---

## 6  Conditioning Token Format

The exact token format for Evo2 training and inference:

### MIBiG rows (compound-specific)

```
|COMPOUND_CLASS:NRPS||COMPOUND:indigoidine||D__BACTERIA;P__PSEUDOMONADOTA;C__GAMMAPROTEOBACTERIA;O__ENTEROBACTERALES;F__ENTEROBACTERIACEAE;G__ESCHERICHIA;S__ESCHERICHIA|ATGCGATCG...
```

### antiSMASH rows (class-level only — no COMPOUND token)

```
|COMPOUND_CLASS:PKS||D__BACTERIA;P__ACTINOMYCETOTA;C__ACTINOMYCETES;O__KITASATOSPORALES;F__STREPTOMYCETACEAE;G__STREPTOMYCES;S__STREPTOMYCES_COELICOLOR|ATGCGATCG...
```

### Inference (swap taxonomic tag to chassis organism)

```
|COMPOUND_CLASS:NRPS||COMPOUND:indigoidine||D__BACTERIA;P__PSEUDOMONADOTA;C__GAMMAPROTEOBACTERIA;O__ENTEROBACTERALES;F__ENTEROBACTERIACEAE;G__ESCHERICHIA;S__ESCHERICHIA|
```

Then let the model generate the sequence autoregressively.

### Taxonomic tag rules

- ALL UPPERCASE
- 7 Linnaean ranks: D (domain), P (phylum), C (class), O (order), F (family), G (genus), S (species)
- Semicolon-delimited, pipe-enclosed
- Spaces → underscores, non-alphanumeric stripped
- Built from NCBI Taxonomy tree walk (not naive GenBank parsing)

---

## 7  Evaluation Metrics

### Overview


| #   | Metric                       | Tool                       | Tier        | GPU?    | Status                           |
| --- | ---------------------------- | -------------------------- | ----------- | ------- | -------------------------------- |
| 1   | BGC class identification     | antiSMASH                  | Primary     | No      | ✅ Implemented & tested           |
| 2   | Domain architecture recovery | pyhmmer + Pfam 37.0        | Primary     | No      | ✅ Implemented & tested           |
| 3   | Protein foldability          | ESMFold + Foldseek         | Primary     | **Yes** | ✅ Implemented & tested (A40 GPU) |
| 4   | Synthesis feasibility        | DNA Chisel                 | Primary     | No      | ✅ Implemented & tested           |
| 5   | Sequence naturalness         | Evo2 base model perplexity | Secondary   | **Yes** | ✅ Implemented & tested (A40 GPU) |
| 6   | Structural novelty           | BiG-SCAPE 2.0              | Secondary   | No      | ✅ Implemented, needs testing     |
| 7   | Organism compatibility       | CAI + GC% + dinucleotide   | Secondary   | No      | ✅ Implemented & tested           |
| 8   | Protein homology             | MMseqs2 vs UniRef50        | Descriptive | No      | ✅ Implemented, needs UniRef50 DB |


### Metric details

**M1 — antiSMASH BGC Class Identification**

- Writes sequence to temp FASTA, runs `antismash --genefinding-tool prodigal`
- Parses output JSON for predicted product types
- Maps predictions through `compound_class_map.yaml` to harmonised vocabulary
- **Pass:** predicted class matches conditioned `COMPOUND_CLASS`

**M2 — Functional Domain Recovery**

- Six-frame ORF finder (min 50 aa)
- Translates ORFs, scans against Pfam-A.hmm via pyhmmer (E < 1e-10)
- Checks obligate domains per class:


| Class                              | Required Pfam Domains                     | Logic                                 |
| ---------------------------------- | ----------------------------------------- | ------------------------------------- |
| PKS                                | PF00109 (KS), PF00698 (AT), PF00550 (ACP) | All required                          |
| NRPS                               | PF00668 (C), PF00501 (A), PF00550 (T/PCP) | All required                          |
| TERPENE                            | PF03936, PF19086, PF01397                 | Any one of                            |
| SACCHARIDE                         | PF00534 (glycosyltransferase)             | Required                              |
| SIDEROPHORE                        | PF04183 (IucA/IucC)                       | Required                              |
| PKS_NRPS_HYBRID                    | PF00109 (KS) + PF00668 (C)                | All required                          |
| RIPP, OTHER, ALKALOID, BETALACTONE | —                                         | No obligate domains (pass by default) |


- **Pass:** all obligate domains found (or class has none defined)

**M3 — Protein Foldability (GPU)**

- ESMFold predicts structure for each ORF
- pLDDT > 70 for majority of residues = confidently folded
- Foldseek structural search against PDB/AlphaFold DB

**M4 — Synthesis Feasibility**

- Global GC: 25–65%
- Local GC (50 bp window): 35–65%
- No homopolymer runs ≥ 10 bp
- No direct/inverted repeats > 20 bp
- DNA Chisel constraint checking against Twist Bioscience specs
- **Pass:** all constraints satisfied

**M5 — Sequence Naturalness (GPU)**

- Loads pretrained Evo2 7B (no fine-tuning)
- Computes per-nucleotide negative log-likelihood
- **Pass:** perplexity within MIBiG reference distribution

**M6 — Structural Novelty**

- BiG-SCAPE 2.0 pairwise distance to MIBiG training corpus
- **Pass:** distance 0.3–0.7 (novel but architecturally coherent)

**M7 — Organism Compatibility**

- CAI vs E. coli K-12 codon usage (target > 0.7)
- GC content vs E. coli ~51%
- Dinucleotide frequency RMSD vs E. coli reference
- **Pass:** composite ≥ 2/3 sub-checks pass

**M8 — Protein Homology (descriptive)**

- MMseqs2 easy-search of ORFs against UniRef50
- Reports max percent identity, number of hits
- Flags > 95% identity as memorisation
- Flags zero hits as suspicious
- **Not a pass/fail metric** — descriptive analysis

### Validated discrimination

Tested on 3 real MIBiG BGCs + shuffled negative controls:

```
Accession            Control      M1 M2 M3 M4 M5 M6 M7 M8
BGC0001537           positive      ✓   ✓   -   ✗   -   -   ✗   -
BGC0001537_shuffled  negative      ✗   ✗   -   ✗   -   -   ✗   -
BGC0000982           positive      ✓   ✓   -   ✗   -   -   ✗   -
BGC0000982_shuffled  negative      ✗   ✗   -   ✗   -   -   ✗   -
BGC0002786           positive      ✓   ✓   -   ✗   -   -   ✗   -
BGC0002786_shuffled  negative      ✗   ✗   -   ✗   -   -   ✗   -
```

M1 and M2 show **perfect discrimination** between real and shuffled BGCs. M4 and
M7 correctly fail on native BGCs (they are from native producers, not E. coli-optimised).
GPU metrics (M3, M5) and external DB metrics (M6, M8) marked `-` (skipped — not
available in local testing).

---

## 8  Key Source Modules

### 8.1  `src/bgc_pipeline/taxonomy.py`

The taxonomy module was rewritten to use NCBI Taxonomy tree walks instead of
naive GenBank ORGANISM parsing. This was critical because:

- GenBank ORGANISM blocks list lineage elements without rank labels
- Naive positional mapping breaks for eukaryotes (Kingdom, Subkingdom, etc. shift all ranks)
- NCBI renamed "superkingdom" → "domain" — handled via `_DOMAIN_ALIASES`

**Key API:**

```python
from bgc_pipeline.taxonomy import load_taxonomy, build_taxonomic_tag

# Load taxonomy once (cached singleton, ~30 sec for 2.7M nodes)
taxonomy = load_taxonomy(Path("data/ncbi_taxonomy"))

# Build tag from a GenBank record's text
tag = build_taxonomic_tag(gbk_text, taxonomy)
# → "|D__BACTERIA;P__PSEUDOMONADOTA;C__GAMMAPROTEOBACTERIA;...|"

# Normalise compound names
from bgc_pipeline.taxonomy import normalize_compound_token
normalize_compound_token("JBIR-76")  # → "jbir-76"
```

Fallback: if NCBI lookup fails (42 records: uncultured/metagenomic), falls
back to GenBank parsing with best-effort rank assignment.

### 8.2  `src/bgc_pipeline/class_map.py`

```python
from bgc_pipeline.class_map import load_class_map, map_mibig_class

mapping, default = load_class_map(Path("config/compound_class_map.yaml"))
cls = map_mibig_class("T1PKS", mapping, default)  # → "PKS"
cls = map_mibig_class("lanthipeptide-class-ii", mapping, default)  # → "RIPP"
```

### 8.3  `src/bgc_pipeline/mibig_record.py`

```python
from bgc_pipeline.mibig_record import iter_mibig_records, record_to_json_dict

for rec in iter_mibig_records(
    json_dir=Path("data/mibig/mibig_json_4.0"),
    gbk_source=Path("data/mibig/mibig_gbk_4.0"),
    mapping=mapping,
    default_class=default,
    taxonomy=taxonomy,
):
    d = record_to_json_dict(rec)
    # d["training_text"] is the full Evo2 input string
```

### 8.4  `src/bgc_pipeline/evaluation.py`

```python
from bgc_pipeline.evaluation import evaluate_bgc, EvalConfig

config = EvalConfig(
    pfam_hmm=Path("data/pfam/Pfam-A.hmm"),
    mibig_gbk_dir=Path("data/mibig/mibig_gbk_4.0"),
    skip_metrics={3, 5, 6, 8},  # skip GPU / external DB metrics
)

result = evaluate_bgc(
    sequence="ATGCGATCG...",
    accession="generated_001",
    expected_class="PKS",
    config=config,
)
# result["metric_1"]["pass"], result["metric_2"]["pass"], etc.
```

---

## 9  Compound Class Map

The harmonised vocabulary in `config/compound_class_map.yaml` maps 60+ raw
labels from MIBiG and antiSMASH into a shared set of tokens:


| Harmonised Token  | Source Labels                                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `PKS`             | PKS, T1PKS, T2PKS, T3PKS, transAT-PKS, hglE-KS                                                                                     |
| `NRPS`            | NRPS, NRPS-like, thioamide-NRP, isocyanide-nrp, NAPAA                                                                              |
| `TERPENE`         | terpene                                                                                                                            |
| `RIPP`            | ribosomal, lanthipeptide (class i–v), thiopeptide, lassopeptide, sactipeptide, cyanobactin, bottromycin, LAP, RiPP-like, + 15 more |
| `SACCHARIDE`      | saccharide, oligosaccharide, amglyccycl, aminocoumarin                                                                             |
| `OTHER`           | other, tropodithietic-acid, NAGGN, acyl_amino_acids                                                                                |
| `PKS_NRPS_HYBRID` | PKS-NRPS_Hybrids, NRPS-PKS_Hybrids                                                                                                 |
| `SIDEROPHORE`     | siderophore, NRP-metallophore, NRPS-independent-siderophore                                                                        |
| `ALKALOID`        | alkaloid, indole                                                                                                                   |
| + 11 more         | BETALACTONE, MELANIN, NUCLEOSIDE, PHOSPHONATE, etc.                                                                                |


Default for unmapped labels: `OTHER`.

---

## 10  Target Compounds for Wet Lab Validation


| Compound                | Class               | Detection                  | Genes          | MIBiG Refs            |
| ----------------------- | ------------------- | -------------------------- | -------------- | --------------------- |
| Violacein               | Shikimate/oxidative | Blue-purple, HPLC 575 nm   | vioABCDE (5)   | Multiple entries      |
| Carotenoid (zeaxanthin) | TERPENE             | Yellow-orange, HPLC 450 nm | crtEBIYZ (~6)  | BGC0000633–BGC0000650 |
| Indigoidine             | NRPS                | Bright blue, HPLC 612 nm   | BpsA + sfp (2) | Multiple entries      |


All are non-hazardous (BSL-1), expressible in E. coli BL21(DE3), and visually
detectable before HPLC.

---

## 11  Known Issues and Decisions

### Resolved issues


| Issue                                                      | Resolution                                                      |
| ---------------------------------------------------------- | --------------------------------------------------------------- |
| MIBiG JSON tarball not gzipped despite `.tar.gz` extension | Use `tar -xf` (no z flag) for JSON; `tar -xzf` for GBK          |
| Eukaryotic taxonomy ranks misassigned                      | Rewrote taxonomy.py to use NCBI taxdump tree walk               |
| Taxonomic tags were mixed case                             | Applied `.upper()` and character sanitisation throughout        |
| NCBI "superkingdom" vs "domain"                            | Added `_DOMAIN_ALIASES = {"superkingdom", "domain"}`            |
| pyhmmer API differences (v0.12)                            | Fixed `.query.accession`, `isinstance` bytes check, `.i_evalue` |
| Terpene obligate domain not found (PF03936)                | Added alternative Pfam families with any-one-of logic           |
| antiSMASH conda conflicts with Python 3.12                 | Removed Python pin from environment.yml; solver picks 3.12.13   |
| `class_match` vs `pass` key in M1                          | Renamed to `pass` for consistency                               |


### Design decisions


| Decision                                                 | Rationale                                                   |
| -------------------------------------------------------- | ----------------------------------------------------------- |
| Omit `COMPOUND` on antiSMASH rows (not null/placeholder) | Truthfully reflects "class only"; avoids sentinel pollution |
| `COMPOUND_CLASS` first in token order                    | MIBiG and antiSMASH share same prefix for class backbone    |
| Full fine-tuning (not LoRA)                              | LoRA incompatible with StripedHyena 2 (BioNeMo issue #884)  |
| OSTIR/RBS metric removed                                 | Removed from plan and code; 8 metrics instead of 9          |
| Lycopene → Carotenoid (zeaxanthin)                       | Lycopene absent from MIBiG; 17 carotenoid entries available |
| 262,144 bp max sequence length                           | Evo2 7B context window; 11 MIBiG records filtered           |


---

## 12  What Remains To Do

### Blocked


| Task                        | Blocker                                 |
| --------------------------- | --------------------------------------- |
| antiSMASH DB v4 integration | Bulk download files missing from server |


### Ready to start


| Task                                            | Prerequisites                  | Notes                                                        |
| ----------------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| Test BiG-SCAPE metric (M6) end-to-end           | antiSMASH DBs ✅                | Needs GenBank output from M1                                 |
| Fine-tune Evo2 7B                               | GPUs 2 & 3 free (46 GB each) ✅ | Phase 1: antiSMASH class-only; Phase 2: MIBiG compound+class |
| Generate BGC sequences                          | Fine-tuned model               | Condition on target compound + E. coli tag                   |
| Full 8-metric evaluation of generated sequences | All metrics now operational ✅  | The main deliverable                                         |
| Identify wet lab collaborator                   | —                              | Parallel to computational work                               |


### Completed since last update (2026-04-07)


| Task                               | Status | Notes                                                                        |
| ---------------------------------- | ------ | ---------------------------------------------------------------------------- |
| UniRef50 database for MMseqs2 (M8) | ✅ Done | 29 GB at `data/uniref50/`                                                    |
| ESMFold (M3) GPU setup             | ✅ Done | `transformers==4.46.3` + A40 GPU; pLDDT 86 on test ORF                       |
| Evo2 7B (M5) GPU setup             | ✅ Done | `evo2==0.5.5` + `flash-attn==2.7.4.post1`; log-likelihood −0.998 on test seq |
| PyTorch version lock               | ✅ Done | Pinned to 2.5.1+cu124 (see requirements.txt for rationale)                   |


### Future enhancements

- 40B Evo2 checkpoint (requires multi-H100 setup)
- LoRA support if BioNeMo adds it for StripedHyena 2
- Additional compound classes when antiSMASH data available
- Codon-optimised MIBiG training variants (if M7 shows poor E. coli compatibility)

---

## 13  Quick Reference Commands

```bash
# Always activate first
conda activate bgcmodel

# Regenerate training data from scratch
PYTHONPATH=src python scripts/mibig_to_jsonl.py \
  --mibig-json-dir data/mibig/mibig_json_4.0 \
  --mibig-gbk data/mibig/mibig_gbk_4.0 \
  --class-map config/compound_class_map.yaml \
  --taxonomy-dir data/ncbi_taxonomy \
  -o data/processed/mibig_train_records.jsonl

# Re-split
PYTHONPATH=src python scripts/split_dataset.py \
  --input data/processed/mibig_train_records.jsonl \
  --output-dir data/processed/splits

# Fast evaluation (no GPU, no external DBs)
PYTHONPATH=src python scripts/evaluate_bgc.py \
  --jsonl data/processed/splits/test.jsonl \
  --max-sequences 3 \
  --skip-metrics 1 3 5 6 8 \
  --pfam-hmm data/pfam/Pfam-A.hmm

# Full evaluation with antiSMASH + negative controls
PYTHONPATH=src python scripts/evaluate_bgc.py \
  --jsonl data/processed/splits/test.jsonl \
  --max-sequences 3 \
  --include-negative-control \
  --skip-metrics 3 5 6 8 \
  --pfam-hmm data/pfam/Pfam-A.hmm

# Smoke test
PYTHONPATH=src python scripts/eval_smoke.py \
  --jsonl data/processed/splits/test.jsonl \
  --max-sequences 5

# Evaluate a generated FASTA
PYTHONPATH=src python scripts/evaluate_bgc.py \
  --fasta my_generated.fasta \
  --expected-class NRPS \
  --pfam-hmm data/pfam/Pfam-A.hmm \
  -o eval_output.json
```

---

*This document should be updated as the project progresses — especially
Sections 4, 7, and 12 as new data is acquired and metrics are validated on GPU.*