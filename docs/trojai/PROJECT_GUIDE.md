# BGC Modelling — Project Guide

*Living document — last updated 2026-04-15*

This document is the single reference for understanding, running, and extending the
de novo BGC generation pipeline built on Evo2. It covers what has been built, how
to reproduce every step, and what remains to be done.

---

## 1  Project Goal

Fine-tune **Evo2 7B** (StripedHyena 2, 262 k context window) to generate
synthesis-ready biosynthetic gene cluster (BGC) nucleotide sequences conditioned on:


| Token            | Example                    | Source                                            |
| ---------------- | -------------------------- | ------------------------------------------------- |
| `COMPOUND_CLASS` | `PKS`, `NRPS`, `TERPENE`   | Harmonised vocabulary shared by MIBiG + antiSMASH |
| `COMPOUND`       | `indigoidine`, `violacein` | MIBiG compound name (normalised)                  |
| Taxonomic tag    | `                          | D__BACTERIA;P__…;S__ESCHERICHIA                   |


The pipeline validates generated sequences with an **eight-metric computational
evaluation suite** before any wet-lab work.

---

## 2  Repository Layout

```
BCGModelling/
├── BGC_Research_Plan.md            # Full research plan (v6, 11 sections)
├── PROJECT_GUIDE.md                # ← you are here
├── FINETUNE_GUIDE.md               # Evo2 fine-tuning: hardware, hyperparameters, logging, checkpointing
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
│   ├── mibig_to_jsonl.py           # Step 1a — MIBiG → JSONL
│   ├── antismash_db_to_jsonl.py    # Step 1b — antiSMASH DB v5 → JSONL
│   ├── annotate_contig_edge.py     # Post-hoc contig_edge annotation (single tar pass)
│   ├── split_dataset.py            # Step 2  — stratified train/val/test
│   ├── finetune_evo2.py            # Step 3a — Evo2 full fine-tune (smoke-tested; OOMs at optimizer.step — use LoRA)
│   ├── finetune_evo2_lora.py       # Step 3b — Evo2 LoRA fine-tune (smoke-test passed ✅ — use this)
│   ├── evaluate_bgc.py             # Step 4  — full evaluation CLI
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
    ├── antismash_db/
    │   ├── asdb5_gbks.tar          # 173 GB — 56,846 genomes annotated by antiSMASH 8.1
    │   └── asdb5_taxa.json.gz      # 946 KB — pre-computed lineage for 29,016 taxids
    ├── uniref50/                   # 29 GB — MMseqs2 UniRef50 DB (for M8)
    └── processed/
        ├── mibig_train_records.jsonl     # 2,636 records (MIBiG only)
        ├── asdb5_train_records.jsonl     # 343,923 records (antiSMASH DB v5, incl. contig_edge) ✅
        └── splits/
            ├── train.jsonl               # 2,099 records (MIBiG only — will be re-split)
            ├── val.jsonl                 #   263 records
            ├── test.jsonl                #   263 records
            └── heldout_accessions.txt    #   526 accessions (val + test)
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

- 4× NVIDIA A40 (48 GB VRAM each, ~46,068 MiB per nvidia-smi); all 4 GPUs available for training
- `CUDA_VISIBLE_DEVICES=0,1,2,3` for fine-tuning (all 4 required — see FINETUNE_GUIDE.md)
- Any single GPU for inference / evaluation
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


| Dataset             | Version  | Files                               | Size       | Status                                |
| ------------------- | -------- | ----------------------------------- | ---------- | ------------------------------------- |
| MIBiG JSON          | 4.0      | 3,013 JSON files                    | 9.6 MB     | ✅ Downloaded & extracted              |
| MIBiG GBK           | 4.0      | ~2,900 GenBank files                | 80 MB      | ✅ Downloaded & extracted              |
| MIBiG protein seqs  | 4.0      | 1 FASTA                             | 31 MB      | ✅ Downloaded                          |
| NPAtlas             | 3.0      | 1 JSON (36,454 compounds)           | 454 MB     | ✅ Downloaded                          |
| Pfam-A.hmm          | 37.0     | 1 HMM file (21,979 families)        | 1.6 GB     | ✅ Downloaded                          |
| NCBI Taxonomy       | Apr 2026 | names.dmp + nodes.dmp               | 464 MB     | ✅ Downloaded & extracted              |
| antiSMASH ref DBs   | —        | via `download-antismash-databases`  | ~15 GB     | ✅ Downloaded                          |
| **antiSMASH DB v5** | **v5**   | **56,846 genomes / ~497K BGCs**     | **173 GB** | ✅ Downloaded — processing in progress |
| antiSMASH taxa JSON | v5       | Pre-computed lineage for 29K taxids | 946 KB     | ✅ Downloaded                          |
| UniRef50            | —        | MMseqs2 DB                          | 29 GB      | ✅ Downloaded — `data/uniref50/`       |


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

### 4.3  antiSMASH DB v5 — downloaded and processed ✅

antiSMASH DB v5 was released January 2026 with 497,429 BGCs from 56,846 genome
assemblies annotated by antiSMASH 8.1 (2× the size of v4).

```bash
# Already downloaded to data/antismash_db/
# asdb5_gbks.tar         — 173 GB, one .gbk.gz per genome assembly
# asdb5_taxa.json.gz     — 946 KB, pre-computed lineage for all 29K taxids

# To regenerate the JSONL from scratch:
conda activate bgcmodel
python scripts/antismash_db_to_jsonl.py \
    --tar          data/antismash_db/asdb5_gbks.tar \
    --taxa         data/antismash_db/asdb5_taxa.json.gz \
    --output       data/processed/asdb5_train_records.jsonl \
    --class-map    config/compound_class_map.yaml \
    --taxonomy-dir data/ncbi_taxonomy \
    --heldout      data/processed/splits/heldout_accessions.txt

# To resume a crashed/interrupted run:
python scripts/antismash_db_to_jsonl.py \
    --resume-after GCF_XXXXXXXXX.X \
    --append \
    --output data/processed/asdb5_train_records.jsonl

# To patch missed genomes (e.g. after a bug fix):
python scripts/antismash_db_to_jsonl.py \
    --only-genomes-file /tmp/missed_genomes.txt \
    --append \
    --output data/processed/asdb5_train_records.jsonl
```

**Key design notes:**

- Emits `COMPOUND_CLASS`-only records — no `COMPOUND` token (antiSMASH has no compound-level labels)
- Each genome GBK may contain multiple BGC regions; all are extracted (one record per region)
- Handles fragmented/draft assemblies: iterates all contigs per GBK (`list(SeqIO.parse(...))`)
- Emits `contig_edge: bool` — True if the BGC region touches a contig boundary (potentially truncated); 11.9% of records
- Taxonomy fast-path: taxon ID from GBK `/db_xref` → pre-computed lineage in `asdb5_taxa.json.gz`
- Fallback chain: taxa JSON → NCBI taxdump → GenBank ORGANISM parser
- Regions > 262,144 bp are centre-truncated; < 100 bp are skipped
- Output is append-compatible with `mibig_train_records.jsonl` (same JSON schema)

**antiSMASH DB v5 record format:**

```json
{
  "accession": "GCF_000007185.1.region1",
  "genome_accession": "GCF_000007185.1",
  "region_number": 1,
  "compound_class": "RIPP",
  "antismash_products": ["RiPP-like"],
  "contig_edge": false,
  "region_start": 105027,
  "region_end": 116160,
  "taxonomic_tag": "|D__ARCHAEA;P__METHANOBACTERIOTA;...|",
  "sequence": "ATGCG...",
  "training_text": "|COMPOUND_CLASS:RIPP||D__ARCHAEA;...|ATGCG...",
  "gbk_member": "GCF_000007185.1"
}
```

---

## 5  Training Data Summary

All figures as of 2026-04-15.  The combined dataset is the union of
`mibig_train_records.jsonl` (2,636 records) and
`asdb5_train_records.jsonl` (343,923 records).

---

### 5.1  Record counts


| Dataset            | Records     | Source genomes / entries | Notes                                  |
| ------------------ | ----------- | ------------------------ | -------------------------------------- |
| MIBiG 4.0          | 2,636       | 2,636                    | One BGC entry per record; 377 filtered |
| antiSMASH DB v5    | 343,923     | 55,950                   | Multiple BGC regions per genome        |
| **Combined total** | **346,559** | —                        | Ready to merge + split                 |


MIBiG filtered: 27 no compound name, 350 no matching GBK, 11 exceed 262,144 bp context window.

---

### 5.2  Compound class distribution

#### MIBiG (2,636 records)


| Class      | Records | % of MIBiG |
| ---------- | ------- | ---------- |
| NRPS       | 875     | 33.2%      |
| PKS        | 738     | 28.0%      |
| RIPP       | 358     | 13.6%      |
| OTHER      | 354     | 13.4%      |
| TERPENE    | 171     | 6.5%       |
| SACCHARIDE | 140     | 5.3%       |


#### antiSMASH DB v5 (343,923 records)


| Class             | Records | % of ASDB |
| ----------------- | ------- | --------- |
| TERPENE           | 86,245  | 25.1%     |
| RIPP              | 84,628  | 24.6%     |
| NRPS              | 40,302  | 11.7%     |
| OTHER             | 28,023  | 8.1%      |
| PKS               | 27,641  | 8.0%      |
| PKS_NRPS_HYBRID   | 14,316  | 4.2%      |
| SIDEROPHORE       | 14,087  | 4.1%      |
| BETALACTONE       | 11,891  | 3.5%      |
| ARYLPOLYENE       | 10,833  | 3.1%      |
| HSERLACTONE       | 7,160   | 2.1%      |
| ECTOINE           | 6,193   | 1.8%      |
| BUTYROLACTONE     | 3,358   | 1.0%      |
| PHOSPHONATE       | 1,860   | 0.5%      |
| MELANIN           | 1,779   | 0.5%      |
| CDPS              | 1,221   | 0.4%      |
| RESORCINOL        | 1,204   | 0.3%      |
| PHENAZINE         | 877     | 0.3%      |
| ALKALOID          | 796     | 0.2%      |
| SACCHARIDE        | 499     | 0.1%      |
| FURAN             | 424     | 0.1%      |
| BETALACTAM        | 192     | 0.1%      |
| PUFA              | 189     | 0.1%      |
| NUCLEOSIDE        | 165     | 0.0%      |
| PHOSPHOGLYCOLIPID | 20      | 0.0%      |
| LADDERANE         | 18      | 0.0%      |
| PBDE              | 2       | 0.0%      |


#### Combined (346,559 records)


| Class              | Records | % of total |
| ------------------ | ------- | ---------- |
| TERPENE            | 86,416  | 24.9%      |
| RIPP               | 84,986  | 24.5%      |
| NRPS               | 41,177  | 11.9%      |
| PKS                | 28,379  | 8.2%       |
| OTHER              | 28,377  | 8.2%       |
| PKS_NRPS_HYBRID    | 14,316  | 4.1%       |
| SIDEROPHORE        | 14,087  | 4.1%       |
| BETALACTONE        | 11,891  | 3.4%       |
| ARYLPOLYENE        | 10,833  | 3.1%       |
| HSERLACTONE        | 7,160   | 2.1%       |
| ECTOINE            | 6,193   | 1.8%       |
| BUTYROLACTONE      | 3,358   | 1.0%       |
| *(8 classes < 1%)* | 8,196   | 2.4%       |


Note: TERPENE and RIPP together account for ~49% of training data (antiSMASH-driven).
NRPS and PKS are better-represented in MIBiG than antiSMASH (MIBiG: 33%+28%, ASDB: 12%+8%).

---

### 5.3  Sequence length distribution


| Length range | MIBiG records | MIBiG % | ASDB records | ASDB % |
| ------------ | ------------- | ------- | ------------ | ------ |
| < 5 kb       | 187           | 7.1%    | 1,278        | 0.4%   |
| 5 – 20 kb    | 815           | 30.9%   | 59,666       | 17.3%  |
| 20 – 50 kb   | 967           | 36.7%   | 227,760      | 66.2%  |
| 50 – 100 kb  | 530           | 20.1%   | 47,884       | 13.9%  |
| 100 – 262 kb | 137           | 5.2%    | 7,335        | 2.1%   |



| Stat   | MIBiG         | antiSMASH DB v5 |
| ------ | ------------- | --------------- |
| Min    | 188 bp        | 1,001 bp        |
| Median | 28,227 bp     | 22,917 bp       |
| Mean   | 39,624 bp     | 32,859 bp       |
| Max    | 4,150,267 bp¹ | 262,144 bp²     |


¹ MIBiG max is an outlier (one very large cluster); the 95th percentile is ~120 kb.
² antiSMASH records are hard-capped at 262,144 bp (Evo2 context window); longer regions are centre-truncated.

---

### 5.4  Taxonomy breakdown

#### Kingdom-level


| Kingdom   | MIBiG | MIBiG % | ASDB    | ASDB % |
| --------- | ----- | ------- | ------- | ------ |
| BACTERIA  | 2,095 | 79.5%   | 336,638 | 97.9%  |
| EUKARYOTA | 529   | 20.1%   | 3,801   | 1.1%   |
| ARCHAEA   | 3     | 0.1%    | 3,484   | 1.0%   |
| UNKNOWN   | 9     | 0.3%    | 0       | 0.0%   |


#### Top phyla (antiSMASH DB v5)


| Phylum            | Records | %     |
| ----------------- | ------- | ----- |
| PSEUDOMONADOTA    | 135,431 | 39.4% |
| ACTINOMYCETOTA    | 106,569 | 31.0% |
| BACILLOTA         | 66,008  | 19.2% |
| BACTEROIDOTA      | 13,775  | 4.0%  |
| ASCOMYCOTA        | 3,261   | 0.9%  |
| CYANOBACTERIOTA   | 3,185   | 0.9%  |
| METHANOBACTERIOTA | 3,072   | 0.9%  |
| MYXOCOCCOTA       | 2,439   | 0.7%  |
| CAMPYLOBACTEROTA  | 2,398   | 0.7%  |
| *(others)*        | 8,780   | 2.6%  |


#### Top phyla (MIBiG)


| Phylum          | Records | %     |
| --------------- | ------- | ----- |
| ACTINOMYCETOTA  | 1,102   | 41.8% |
| ASCOMYCOTA      | 455     | 17.3% |
| PSEUDOMONADOTA  | 453     | 17.2% |
| BACILLOTA       | 237     | 9.0%  |
| CYANOBACTERIOTA | 131     | 5.0%  |
| MYXOCOCCOTA     | 101     | 3.8%  |
| *(others)*      | 157     | 6.0%  |


#### Top genera (antiSMASH DB v5, top 15)


| Genus          | Records |
| -------------- | ------- |
| STREPTOMYCES   | 56,280  |
| PSEUDOMONAS    | 24,954  |
| BACILLUS       | 19,682  |
| KLEBSIELLA     | 7,262   |
| STREPTOCOCCUS  | 6,826   |
| VIBRIO         | 6,351   |
| ESCHERICHIA    | 6,239   |
| PAENIBACILLUS  | 5,688   |
| ACINETOBACTER  | 5,474   |
| ENTEROBACTER   | 4,399   |
| STAPHYLOCOCCUS | 4,377   |
| RHODOCOCCUS    | 3,892   |
| MICROMONOSPORA | 3,718   |
| BURKHOLDERIA   | 3,368   |
| SERRATIA       | 3,308   |


Total unique genera across antiSMASH DB: **3,492**.
Total unique genera across MIBiG: **410**.

#### Top genera (MIBiG, top 10)


| Genus          | Records |
| -------------- | ------- |
| STREPTOMYCES   | 800     |
| ASPERGILLUS    | 158     |
| PSEUDOMONAS    | 97      |
| BACILLUS       | 61      |
| PENICILLIUM    | 56      |
| FUSARIUM       | 43      |
| MICROMONOSPORA | 39      |
| BURKHOLDERIA   | 39      |
| STREPTOCOCCUS  | 39      |
| AMYCOLATOPSIS  | 35      |


---

### 5.5  MIBiG compound coverage

MIBiG is the only source of compound-name conditioning (`COMPOUND` token).
The antiSMASH DB provides class-level supervision only — no `COMPOUND` token is emitted for those records.

#### Top-level counts

| Stat                                          | Value  |
| --------------------------------------------- | -----: |
| Total MIBiG records                           |  2,636 |
| Records with a compound token                 |  2,636 (100%) |
| **Unique normalised compound tokens**         |  **2,295** |
| Tokens appearing exactly once                 |  2,093 (91.2%) |
| Tokens appearing 2+ times (repeated)         |    202 (8.8%) |
| Records covered by a repeated token           |    543 (20.6%) |
| Unique raw compound names (aliases, case-insensitive) | 4,245 |
| Total raw name entries across all records     |  4,732 |
| antiSMASH DB compound tokens                  |  none — class-level only |

Compound tokens are normalised from `compound_names_all[0]`: lowercased, spaces → underscores,
non-alphanumeric characters stripped. The 4,245 unique raw names include all aliases
(e.g. `aflatoxin B1`, `aflatoxin G1`, `aflatoxin B2` all normalise to their first-listed alias).

#### Repeated compound tokens

202 compound tokens appear in 2 or more records.
The repeats arise because MIBiG contains **multiple independently characterised BGCs**
that produce the same compound (different organisms, strains, or parallel biosynthetic pathways).

| Repeat count | # distinct compound tokens | Example                                            |
| -----------: | -------------------------: | -------------------------------------------------- |
| 30×          |                          1 | `capsular_polysaccharide` (30 bacterial strains)   |
| 17×          |                          1 | `carotenoid`                                       |
| 11×          |                          1 | `o-antigen`                                        |
| 10×          |                          1 | `ectoine`                                          |
|  8×          |                          1 | `lipopolysaccharide`                               |
|  7×          |                          1 | `ochratoxin_a`                                     |
|  5–6×        |                          9 | `exopolysaccharide`, `mycophenolic_acid`, `kanamycin`, `melanin`, `streptothricin_f`, `myxochromide_a/d`, `coformycin`, `glycopeptidolipid` |
|  4×          |                          5 | `prodigiosin`, `eicosapentaenoic_acid`, `aerobactin`, `cylindrospermopsin`, `1-heptadecene` |
|  3×          |                         30 | `bacillibactin`, `enterobactin`, `yersiniabactin`, `violacein`, `valinomycin`, … |
|  2×          |                        152 | majority of repeats; typically same compound from different taxa |

The top repeated compound is `capsular_polysaccharide` (30 records from 30 bacterial species) —
this is a broad functional category rather than a single compound, and is typical of SACCHARIDE class.

#### Cross-class compounds

26 compound tokens appear in records assigned to **more than one compound class**.
This reflects genuine biochemical ambiguity (hybrid PKS-NRPS pathways, or MIBiG annotation revision):

| Compound token      | Classes                   | Note                                          |
| ------------------- | ------------------------- | --------------------------------------------- |
| `ochratoxin_a`      | PKS, NRPS                 | Hybrid biosynthesis; both annotations valid   |
| `coformycin`        | SACCHARIDE, OTHER         | Nucleoside + aminocyclitol hybrid             |
| `glycopeptidolipid` | SACCHARIDE, NRPS          | Peptide backbone with glycan decorations      |
| `indigoidine`       | SACCHARIDE, NRPS          | Different MIBiG entries use different classes |
| `valinomycin`       | SACCHARIDE, NRPS          | Depsipeptide with sugar moiety                |
| `citrinin`          | PKS, OTHER                | Mixed annotation across MIBiG versions        |
| *(20 others)*       | typically PKS/NRPS hybrid | —                                             |

These 26 compounds contribute some class-label noise but are a small fraction (<1%) of the dataset.

#### Notable well-represented compounds (≥ 3 records)

| Compound              | Records | Class     | Significance                                  |
| --------------------- | ------: | --------- | --------------------------------------------- |
| `capsular_polysaccharide` |  30 | SACCHARIDE | Broad category; 30 distinct bacterial CPS loci |
| `carotenoid`          |      17 | TERPENE   | Multiple organisms; target class for generation |
| `o-antigen`           |      11 | SACCHARIDE | Gram-negative LPS O-antigen loci              |
| `ectoine`             |      10 | OTHER     | Widespread stress-protection osmolyte         |
| `lipopolysaccharide`  |       8 | SACCHARIDE | Core/O-antigen LPS clusters                   |
| `ochratoxin_a`        |       7 | PKS/NRPS  | Mycotoxin; hybrid biosynthesis                |
| `streptomycin`        |       5 | SACCHARIDE | Aminoglycoside; `streptothricin_f` token covers F–E variants |
| `mycophenolic_acid`   |       5 | PKS       | Immunosuppressant from *Penicillium*          |
| `kanamycin`           |       5 | SACCHARIDE | Aminoglycoside antibiotic                     |

---

#### SMILES coverage audit (2026-04-15)

SMILES were sourced from two places:
1. **Direct**: `compounds[].structure` field in each MIBiG JSON entry
2. **NPAtlas cross-reference**: `compounds[].databaseIds` entries of the form `npatlas:NPAXXXXXX`,
   looked up in `data/npatlas/NPAtlas_download.json` (36,454 compounds with SMILES)

**MIBiG entry-level coverage (all 3,013 JSON files):**

| Source                              | Entries | % of 3,013 |
| ----------------------------------- | ------: | ---------: |
| Has direct SMILES (`structure`)     |   2,387 |      79.2% |
| Has NPAtlas ID with SMILES          |     965 |      32.0% |
| Has SMILES from **either** source   |   2,390 |      **79.3%** |
| Has both direct + NPAtlas           |     962 |      31.9% |
| **No SMILES at all**                |   **623** |  **20.7%** |

**Compound-level coverage (5,443 total compound entries across all MIBiG JSONs):**

| Stat                                        | Value |
| ------------------------------------------- | ----: |
| Compound entries with SMILES (either source) | 4,410 (81.0%) |
| Compound entries with no SMILES              |   1,033 (19.0%) |

**Training record coverage (2,636 processed records in JSONL):**

| Stat                                             | Value |
| ------------------------------------------------ | ----: |
| Records with ≥ 1 SMILES                          | 2,118 (80.3%) |
| Records with no SMILES                           |   518 (19.7%) |
| Records with multiple SMILES (multi-compound BGC) |   654 (24.8%) |
| Unique SMILES strings across all records         |  3,564 |

**Coverage by compound class (training records):**

| Class      | Total | w/ SMILES | Coverage |
| ---------- | ----: | --------: | -------: |
| PKS        |   738 |       677 |    91.7% |
| TERPENE    |   171 |       153 |    89.5% |
| NRPS       |   875 |       760 |    86.9% |
| OTHER      |   354 |       301 |    85.0% |
| SACCHARIDE |   140 |        74 |    52.9% |
| RIPP       |   358 |       153 |    42.7% |

RIPP and SACCHARIDE have the poorest SMILES coverage. RiPPs are often short peptides whose
structures are not always deposited; saccharides are frequently described as compound classes
(e.g. "capsular polysaccharide") rather than specific structures.

**SMILES string length distribution (first SMILES per record, n=2,118):**

| Length range | Records |    % |
| ------------ | ------: | ---: |
| ≤ 50 chars   |     380 | 17.9% |
| 51–100 chars |     782 | 36.9% |
| 101–200 chars |    681 | 32.2% |
| > 200 chars  |     275 | 13.0% |

Median SMILES length: **92 characters**; mean: 117; max: 1,018 (a very large macrolide).
At median length of 92 chars, a SMILES conditioning prefix would consume <0.04% of Evo2's
262,144 bp context window — negligible overhead.

**Implication for SMILES conditioning:**
- 2,118 of 2,636 training records (80.3%) can be SMILES-conditioned immediately
- Canonicalisation with RDKit is required before training (eliminates representation variance)
- SACCHARIDE and RIPP coverage is low enough (~43–53%) that those classes would need
  a fallback strategy (class-only conditioning, or NPAtlas/PubChem lookup for the gaps)
- The 518 records without SMILES would either be dropped or conditioned on class only

---

### 5.6  Data quality flags


| Flag                            | ASDB records | %     | Notes                                                            |
| ------------------------------- | ------------ | ----- | ---------------------------------------------------------------- |
| `contig_edge=True`              | 41,065       | 11.9% | BGC touches contig boundary; may be truncated at one/both flanks |
| `contig_edge=False`             | 302,858      | 88.1% | Complete BGC within contig                                       |
| Centre-truncated (> 262,144 bp) | ~7,335       | 2.1%  | Max-length cap applied; affects largest clusters only            |


MIBiG records have no `contig_edge` field (full-length curated BGC entries).

---

### 5.7  Current train / val / test splits

These are **MIBiG-only** splits (pre-merge). Combined splits will be regenerated
after merging MIBiG + antiSMASH DB.


| Split | Records | NRPS | PKS | RIPP | OTHER | TERPENE | SACCHARIDE |
| ----- | ------- | ---- | --- | ---- | ----- | ------- | ---------- |
| train | 2,099   | 700  | 589 | 286  | 279   | 134     | 111        |
| val   | 263     | 87   | 74  | 36   | 35    | 17      | 14         |
| test  | 263     | 87   | 74  | 36   | 35    | 17      | 14         |


Stratified by `compound_class`; zero overlap verified.
Heldout set (val + test): 526 accessions in `data/processed/splits/heldout_accessions.txt`.

---

## 6  Data Pipeline — Step by Step

All commands assume you are in the repo root with `bgcmodel` activated.

### Step 1a: Convert MIBiG to JSONL

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

### Step 1b: Convert antiSMASH DB v5 to JSONL

```bash
python scripts/antismash_db_to_jsonl.py \
    --tar          data/antismash_db/asdb5_gbks.tar \
    --taxa         data/antismash_db/asdb5_taxa.json.gz \
    --output       data/processed/asdb5_train_records.jsonl \
    --class-map    config/compound_class_map.yaml \
    --taxonomy-dir data/ncbi_taxonomy \
    --heldout      data/processed/splits/heldout_accessions.txt
```

**What it does:**

- Streams the 173 GB tar without full extraction (memory-efficient)
- Parses ALL contigs per genome (multi-record GBKs for fragmented assemblies)
- Finds every antiSMASH `region` feature and extracts the BGC sub-sequence
- Resolves taxonomy from pre-computed `asdb5_taxa.json.gz` (fast-path) or NCBI taxdump (fallback)
- Filters out MIBiG val/test accessions (heldout set) to prevent data leakage
- Maps antiSMASH product types → harmonised COMPOUND_CLASS vocabulary
- Handles hybrid BGCs (e.g. T1PKS + NRPS → PKS_NRPS_HYBRID)

**Expected output:** ~400K records in `data/processed/asdb5_train_records.jsonl`

**Current status:** ~247K records written; patch run in progress to add ~15K
multi-contig assembly genomes that were initially missed by a single-record parsing bug.

### Step 2: Split into train / val / test

```bash
# MIBiG-only splits (current state — will be re-run after antiSMASH DB processing):
PYTHONPATH=src python scripts/split_dataset.py \
  --input     data/processed/mibig_train_records.jsonl \
  --output-dir data/processed/splits \
  --seed 42 \
  --max-seq-len 262144

# Combined MIBiG + antiSMASH DB splits (run once asdb5_train_records.jsonl is complete):
cat data/processed/mibig_train_records.jsonl \
    data/processed/asdb5_train_records.jsonl \
  | PYTHONPATH=src python scripts/split_dataset.py \
      --input -  \
      --output-dir data/processed/splits_combined \
      --seed 42
```

**What it does:**

- Filters out sequences exceeding the 262,144 bp Evo2 context window
- Stratifies by `compound_class` (80% train / 10% val / 10% test)
- Writes `heldout_accessions.txt` listing val + test accessions

**Current output (MIBiG-only splits):**


| Split   | Records | File                                           |
| ------- | ------- | ---------------------------------------------- |
| Train   | 2,099   | `data/processed/splits/train.jsonl`            |
| Val     | 263     | `data/processed/splits/val.jsonl`              |
| Test    | 263     | `data/processed/splits/test.jsonl`             |
| Heldout | 526     | `data/processed/splits/heldout_accessions.txt` |


**After antiSMASH DB processing completes**, the combined splits will be ~400K train
records, with MIBiG val/test preserved to maintain the evaluation benchmark.

**Verified properties of current splits:**

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

See Section 8 for metric details.

---

## 7  Conditioning Token Format

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

### Generation strategy: transposition vs invention

#### The core distinction

This project is bounded by a fundamental data constraint that determines what the model
can and cannot do. It is important to be explicit about this.

**Invention** — generating a BGC for a novel compound with no close known analogue — requires
the model to learn a mapping from chemical structure or function to biosynthetic sequence.
That requires hundreds to thousands of examples per compound class with known structure-sequence
pairings. We do not have this data: 91% of MIBiG compound tokens appear in exactly one record.
The model cannot learn what the token `violacein` means chemically from a single sequence;
it can at best memorise the one example.

**Transposition** — taking a known BGC architecture and regenerating it for a new chassis
organism — is a different and more tractable problem. The model needs to learn:

1. What biosynthetic gene architecture is associated with a given compound or compound class
   (learned from MIBiG, 2,636 records)
2. What BGC sequences look like in the target chassis — codon usage, GC content, operon
   spacing, regulatory elements (learned from the 6,239 *E. coli* and ~56K total antiSMASH
   genome records)

These two things are learned from **different parts of the training data** and do not require
the model to generalise to new chemistry. The compound or class token acts as an architectural
pointer; the taxonomic tag handles the chassis adaptation. This is a reasonable ask of a
fine-tuned 7B-parameter language model.

**What we are building:** A system that, given a conditioning token for a known compound class
(and optionally a known compound), generates a plausible BGC sequence with the correct
biosynthetic architecture expressed with chassis-appropriate sequence statistics. We are
**not** building a system that designs novel biosynthetic pathways from scratch.

#### Why this matters for evaluation

The M1 metric (antiSMASH class prediction) directly validates transposition: it checks
whether the generated sequence has the correct biosynthetic class annotation. A model that
successfully transposes a PKS cluster to *E. coli* should produce a sequence antiSMASH
annotates as PKS, with the correct Pfam domain complement (M2), reasonable codon adaptation
index for *E. coli* (M7), and foldable protein products (M3). These metrics are sufficient
to validate transposition. They cannot validate invention, because we have no ground truth
for novel compounds.

#### What compound-level conditioning can and cannot do

The `COMPOUND` token is meaningful only to the extent the model has seen enough examples
of a given compound to associate the token with a distinctive biosynthetic architecture.
The practical tiers are:

| Tier | Examples per compound | # tokens | What the model learns |
| ---- | --------------------: | --------: | --------------------- |
| **Strong signal** | ≥ 5 | ~10 | Real architectural signal; model associates token with specific module/domain composition |
| **Weak signal** | 3–4 | ~35 | Some compound-specific bias; output distinguishable from class-random |
| **Memorisation risk** | 1–2 | ~2,250 | Output will closely resemble the single training sequence; model is essentially interpolating the one example |

For the memorisation-risk tier, the `COMPOUND` token still provides value: the model
uses the one training example as an architectural template and adapts it to the target
chassis. This is useful — generating a chassis-adapted variant of a known BGC is a
legitimate goal — but it should be understood as **chassis adaptation of a known sequence**,
not generation of an independent novel sequence.

The best near-term use cases for compound-specific generation are therefore the
well-represented compounds: `carotenoid` (×17), `o-antigen` (×11), `ectoine` (×10),
`lipopolysaccharide` (×8), `mycophenolic_acid` (×5), `kanamycin` (×5), and the ~35
compounds with 3–4 examples. For these, the model has enough signal to produce
architecturally coherent variations, not just memorised copies.

---

### Phased conditioning plan

> **Full fine-tuning detail** (hardware, hyperparameters, logging, checkpointing,
> launch commands) is in **FINETUNE_GUIDE.md**. The summary below covers the data
> conditioning strategy only.

Three phases, each building on the last. Phase 1 is ready to run now.

#### Phase 1 — Class-only conditioning (current plan)

**Format:** `|COMPOUND_CLASS:{cls}|{tax_tag}{sequence}` for **all** records.

The `COMPOUND` token is dropped from MIBiG records. Both MIBiG and antiSMASH DB
records use identical format. This is the lowest-noise, highest-data-efficiency
configuration: 346,559 uniformly formatted records, no sparse compound tokens.

**Validation:** M1 class match rate is the primary signal. If the fine-tuned model
generates sequences where antiSMASH predicts the correct class at > random-baseline
rate (which would be ~8% for TERPENE, the largest class), the conditioning is working.
M2 domain recovery and M7 chassis compatibility provide secondary validation.

**Goal:** Establish that the model can generate class-correct, chassis-appropriate BGC
sequences at all. All subsequent phases depend on this working.

#### Phase 2 — Compound conditioning for well-represented compounds

**Format:** MIBiG records with ≥ 3 examples use `|COMPOUND_CLASS:{cls}||COMPOUND:{tok}|{tax_tag}{sequence}`.
MIBiG records with 1–2 examples and all antiSMASH records remain class-only.

This targets the ~45 compounds with strong or weak signal (≥ 3 examples). The test:
does conditioning on `carotenoid` vs `ectoine` produce architecturally distinct outputs
even when both are TERPENE/OTHER class? If yes, the compound token is learning real
architectural signal beyond what the class token alone provides.

**Data change required:** Filter MIBiG JSONL by `compound_token` count ≥ 3 before
emitting the `COMPOUND` token. Small script change to `mibig_to_jsonl.py`.

#### Phase 3 — SMILES conditioning (future)

**Format:** Replace `COMPOUND` name token with canonical SMILES:
`|COMPOUND_CLASS:{cls}||SMILES:{canonical_smiles}|{tax_tag}{sequence}`

**Coverage:** 2,118 of 2,636 MIBiG training records have usable SMILES (80.3%).
SACCHARIDE (52.9%) and RIPP (42.7%) have poor coverage and would fall back to class-only.

**SMILES characteristics:** Median length 92 characters (< 0.04% of Evo2's 262,144 bp
context window — negligible overhead). Canonicalisation via RDKit required before training.

**What this enables:** Chemical interpolation at inference time. Given a target molecule's
SMILES, the model can be conditioned on a structurally similar compound's sequence grammar
without that exact compound appearing in training. Particularly promising for PKS (tight
structure-module relationship) and less so for NRPS/RiPP.

**Prerequisite:** Phase 1 and 2 must show that conditioning tokens are being used
meaningfully before investing in the SMILES infrastructure.

---

## 8  Evaluation Metrics

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

## 9  Key Source Modules

### 9.1  `src/bgc_pipeline/taxonomy.py`

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

### 9.2  `src/bgc_pipeline/class_map.py`

```python
from bgc_pipeline.class_map import load_class_map, map_mibig_class

mapping, default = load_class_map(Path("config/compound_class_map.yaml"))
cls = map_mibig_class("T1PKS", mapping, default)  # → "PKS"
cls = map_mibig_class("lanthipeptide-class-ii", mapping, default)  # → "RIPP"
```

### 9.3  `src/bgc_pipeline/mibig_record.py`

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

### 9.4  `src/bgc_pipeline/evaluation.py`

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

## 10  Compound Class Map

The harmonised vocabulary in `config/compound_class_map.yaml` maps 60+ raw
labels from MIBiG and antiSMASH into a shared set of tokens:


| Harmonised Token  | Source Labels                                                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `PKS`             | PKS, T1PKS, T2PKS, T3PKS, transAT-PKS, hglE-KS, HR-T2PKS                                                                                                                 |
| `NRPS`            | NRPS, NRPS-like, thioamide-NRP, isocyanide-nrp, NAPAA, t3nrps-iterative                                                                                                  |
| `TERPENE`         | terpene, terpene-precursor, cf_polyprenyl                                                                                                                                |
| `RIPP`            | ribosomal, lanthipeptide (class i–vi), thiopeptide, lassopeptide, sactipeptide, cyanobactin, bottromycin, LAP, RiPP-like, azole-containing-RiPP, thioamitides, + 15 more |
| `SACCHARIDE`      | saccharide, oligosaccharide, amglyccycl, aminocoumarin, cf_saccharide                                                                                                    |
| `OTHER`           | other, tropodithietic-acid, NAGGN, acyl_amino_acids, hydrogen-cyanide, cf_putative, cf_fatty_acid, nitrous-oxide                                                         |
| `PKS_NRPS_HYBRID` | PKS-NRPS_Hybrids, NRPS-PKS_Hybrids (+ auto-detected from mixed regions)                                                                                                  |
| `SIDEROPHORE`     | siderophore, NRP-metallophore, NRPS-independent-siderophore                                                                                                              |
| `ALKALOID`        | alkaloid, indole, prodiginine                                                                                                                                            |
| + 11 more         | BETALACTONE, MELANIN, NUCLEOSIDE, PHOSPHONATE, RESORCINOL, etc.                                                                                                          |


Total: 91 mapped labels. Default for unmapped labels: `OTHER`.

Updated 2026-04-14: added 16 antiSMASH v5/v8 product types
(`azole-containing-RiPP`, `thioamitides`, `terpene-precursor`, `t3nrps-iterative`,
`cyclic-lactone-autoinducer`, `lanthipeptide-class-vi`, `cf_*` ClusterFinder types, etc.).

---

## 11  Target Compounds for Wet Lab Validation


| Compound                | Class               | Detection                  | Genes          | MIBiG Refs            |
| ----------------------- | ------------------- | -------------------------- | -------------- | --------------------- |
| Violacein               | Shikimate/oxidative | Blue-purple, HPLC 575 nm   | vioABCDE (5)   | Multiple entries      |
| Carotenoid (zeaxanthin) | TERPENE             | Yellow-orange, HPLC 450 nm | crtEBIYZ (~6)  | BGC0000633–BGC0000650 |
| Indigoidine             | NRPS                | Bright blue, HPLC 612 nm   | BpsA + sfp (2) | Multiple entries      |


All are non-hazardous (BSL-1), expressible in E. coli BL21(DE3), and visually
detectable before HPLC.

---

## 12  Known Issues and Decisions

### Resolved issues


| Issue                                                                     | Resolution                                                                                                      |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| MIBiG JSON tarball not gzipped despite `.tar.gz` extension                | Use `tar -xf` (no z flag) for JSON; `tar -xzf` for GBK                                                          |
| Eukaryotic taxonomy ranks misassigned                                     | Rewrote taxonomy.py to use NCBI taxdump tree walk                                                               |
| Taxonomic tags were mixed case                                            | Applied `.upper()` and character sanitisation throughout                                                        |
| NCBI "superkingdom" vs "domain"                                           | Added `_DOMAIN_ALIASES = {"superkingdom", "domain"}`                                                            |
| pyhmmer API differences (v0.12)                                           | Fixed `.query.accession`, `isinstance` bytes check, `.i_evalue`                                                 |
| Terpene obligate domain not found (PF03936)                               | Added alternative Pfam families with any-one-of logic                                                           |
| antiSMASH conda conflicts with Python 3.12                                | Removed Python pin from environment.yml; solver picks 3.12.13                                                   |
| `class_match` vs `pass` key in M1                                         | Renamed to `pass` for consistency                                                                               |
| fair-esm/OpenFold dependency chain (cascading failures on Python 3.12)    | Abandoned fair-esm; used `transformers==4.46.3` `EsmForProteinFolding`                                          |
| PyTorch 2.6 c10::Error ABI break (flash-attn undefined symbol at runtime) | Pinned `torch==2.5.1+cu124`; `flash-attn==2.7.4.post1` compiles cleanly                                         |
| transformers 5.x blocks `.pt` loading on torch < 2.6 (CVE-2025-32434)     | Pinned `transformers==4.46.3` (last pre-CVE version)                                                            |
| Evo2 `score_sequences()` unexpected `device` kwarg                        | Removed `device` kwarg; Evo2 0.5.5 sets device at model load time                                               |
| ESMFold pLDDT wrong scale (atom-level mean, not residue-level)            | Use `.mean(dim=-1)` over 37 atoms, then `* 100.0` for 0–100 scale                                               |
| antiSMASH DB processing: disk filled (100%) mid-run                       | Deleted superseded `asdb-beta2-jsons.tar` (106 GB freed); resumed with `--resume-after --append`                |
| antiSMASH DB script: only parsed first contig per genome                  | Fixed `_parse_gbk_bytes` to return all records; 94% of missed genomes were fragmented (multi-record) assemblies |


### Design decisions


| Decision                                                   | Rationale                                                                                                                                                                                                                               |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Omit `COMPOUND` on antiSMASH rows (not null/placeholder)   | Truthfully reflects "class only"; avoids sentinel pollution                                                                                                                                                                             |
| `COMPOUND_CLASS` first in token order                      | MIBiG and antiSMASH share same prefix for class backbone                                                                                                                                                                                |
| LoRA fine-tuning (not full fine-tune)                      | Full fine-tune OOMs on 4× A40: StripedHyena activations + AdamW state exceed 46 GB/rank. LoRA reduces optimizer state 167× (56 GB → 336 MB), peak GPU mem 23.2 GB at L=1024. peft 0.19.0 is compatible after 3 fixes (see FINETUNE_GUIDE.md §12.6). BioNeMo issue #884 was wrong. |
| OSTIR/RBS metric removed                                   | Removed from plan and code; 8 metrics instead of 9                                                                                                                                                                                      |
| Lycopene → Carotenoid (zeaxanthin)                         | Lycopene absent from MIBiG; 17 carotenoid entries available                                                                                                                                                                             |
| 262,144 bp max sequence length                             | Evo2 7B context window; 11 MIBiG records filtered                                                                                                                                                                                       |
| One BGC region = one training record (no contig stitching) | BGCs (20–150 kb) almost always fit within one contig; antiSMASH already emits one sliced region per BGC call, so records are contiguous DNA by construction                                                                             |
| Keep `contig_edge=True` BGCs in v1 training (don't filter) | 11.9% of antiSMASH BGCs touch a contig boundary and may be truncated at one/both flanks. Core biosynthetic logic is typically intact; Evo2 tolerates partial context. Filter only if v1 generations show pathological early-termination |


### `contig_edge` noise characterisation (sampled 2,004 antiSMASH BGCs)


| Metric                             | Value                                                    |
| ---------------------------------- | -------------------------------------------------------- |
| `contig_edge=True` rate            | 41,065 / 343,923 = **11.9%** (sample estimate was 11.2%) |
| Truncated <10 kb (most concerning) | 39 (17% of truncated, 1.9% of all)                       |
| Truncated 10–30 kb                 | 117 (52% of truncated)                                   |
| Truncated 30–60 kb                 | 54 (24% of truncated)                                    |
| Truncated 60 kb+                   | 14 (6% of truncated)                                     |


**Status:** `contig_edge: bool` is present in all 343,923 records (annotated 2026-04-15 via `annotate_contig_edge.py`, 0 unmatched). Leave records in for v1 fine-tune; revisit only if generations show cluster-architecture truncation artifacts. To filter edge BGCs at split time: `jq 'select(.contig_edge == false)' asdb5_train_records.jsonl`.

---

## 13  What Remains To Do

### Ready to start (unblocked)


| Task                                            | Prerequisites                    | Notes                                                                        |
| ----------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------- |
| **⭐ NEXT: Per-block activation checkpointing** | `finetune_evo2_lora.py` ✅       | Required before production launch. Without it, max feasible L is ~2048 bp — covers only ~9% of median BGC (23 kb). With per-block `torch.utils.checkpoint`, estimated peak mem at L=32768 drops from ~112 GB → ~18–22 GB, enabling full-length training on 83% of sequences. Wrap the 32 StripedHyena blocks in `finetune_evo2_lora.py`. See FINETUNE_GUIDE.md §1 for memory analysis. |
| Fine-tune Evo2 7B                               | Combined splits ✅ + activation checkpointing ⬆️ | Do NOT launch at L ≤ 8192 — core biosynthetic domains are past the truncation point for most BGCs. Launch at L=32768 after checkpointing is implemented and smoke-tested. |
| Generate BGC sequences                          | Fine-tuned model                 | Condition on target class + E. coli taxonomy tag                             |
| Full 8-metric evaluation of generated sequences | Generated sequences              | All 8 metrics operational; main project deliverable                          |
| Test BiG-SCAPE metric (M6) end-to-end           | antiSMASH DBs ✅                  | Needs GenBank output from M1; structural novelty scoring                     |
| Identify wet lab collaborator                   | —                                | Parallel track; not blocking computational work                              |


### Completed (cumulative)


| Task                                        | Date       | Notes                                                                                        |
| ------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------- |
| MIBiG 4.0 preprocessing + JSONL             | 2026-04-05 | 2,636 records; taxonomy via NCBI taxdump                                                     |
| Stratified MIBiG train/val/test splits      | 2026-04-05 | 2,099 / 263 / 263; zero overlap verified                                                     |
| 8-metric evaluation suite (M1–M8)           | 2026-04-06 | All implemented; M1+M2 show perfect discrimination on MIBiG test set                         |
| Pfam-A.hmm download (M2)                    | 2026-04-05 | Pfam 37.0, 21,979 families                                                                   |
| NCBI taxonomy download + integration        | 2026-04-05 | 2.7M nodes; Evo2-format 7-rank tags                                                          |
| antiSMASH reference DBs                     | 2026-04-06 | Via `download-antismash-databases` (~15 GB)                                                  |
| UniRef50 for MMseqs2 (M8)                   | 2026-04-07 | 29 GB at `data/uniref50/`                                                                    |
| ESMFold GPU setup (M3)                      | 2026-04-07 | `transformers==4.46.3`; pLDDT 86 on A40 GPU                                                  |
| Evo2 7B GPU setup (M5)                      | 2026-04-07 | `evo2==0.5.5` + `flash-attn==2.7.4.post1`; log-likelihood −0.998 on test seq                 |
| PyTorch version lock + `requirements.txt`   | 2026-04-07 | Pinned `torch==2.5.1+cu124`; rationale documented in requirements.txt                        |
| antiSMASH DB v5 download                    | 2026-04-08 | `asdb5_gbks.tar` (173 GB) + `asdb5_taxa.json.gz` (946 KB)                                    |
| `scripts/antismash_db_to_jsonl.py`          | 2026-04-08 | Streaming tar parser; taxa JSON fast-path; resume/append/patch modes                         |
| Class map expanded for antiSMASH v5         | 2026-04-08 | 16 new product types added; 91 total mappings; OTHER rate dropped to ~4%                     |
| Multi-contig GBK parsing bug fixed          | 2026-04-14 | `next(SeqIO.parse)` → `list(SeqIO.parse)`; was missing 94% of fragmented assemblies          |
| `contig_edge` annotation of antiSMASH JSONL | 2026-04-15 | Single tar pass via `annotate_contig_edge.py`; 41,065/343,923 = 11.9% edge BGCs; 0 unmatched |
| `scripts/finetune_evo2.py` written + smoke-tested | 2026-04-15 | Full fine-tune script; DeepSpeed ZeRO-2 + WandB; 3 Evo2↔DS bugs fixed; OOMs at optimizer.step on 4× A40 — use LoRA instead |
| `scripts/finetune_evo2_lora.py` written + smoke-tested | 2026-04-15 | LoRA script (peft 0.19); r=16, 28.7M trainable params (0.44%); 5 bugs fixed; 23.2 GB peak at L=1024; val loss 1.93→1.89 in 10 steps; pipeline ready to launch |


### Future enhancements

- 40B Evo2 checkpoint (requires multi-H100 setup)
- Activation checkpointing at StripedHyena block level (needed for L > 4096 training)
- Increase LoRA rank to 32–64 if val loss plateaus early in production run
- Additional compound classes when antiSMASH data available
- Codon-optimised MIBiG training variants (if M7 shows poor E. coli compatibility)

---

## 14  Quick Reference Commands

```bash
# Always activate first
conda activate bgcmodel

# Regenerate MIBiG training data from scratch
PYTHONPATH=src python scripts/mibig_to_jsonl.py \
  --mibig-json-dir data/mibig/mibig_json_4.0 \
  --mibig-gbk data/mibig/mibig_gbk_4.0 \
  --class-map config/compound_class_map.yaml \
  --taxonomy-dir data/ncbi_taxonomy \
  -o data/processed/mibig_train_records.jsonl

# Process antiSMASH DB v5 (full run, ~5 hrs)
python scripts/antismash_db_to_jsonl.py \
  --tar          data/antismash_db/asdb5_gbks.tar \
  --taxa         data/antismash_db/asdb5_taxa.json.gz \
  --output       data/processed/asdb5_train_records.jsonl \
  --heldout      data/processed/splits/heldout_accessions.txt

# Check antiSMASH DB processing progress
wc -l data/processed/asdb5_train_records.jsonl
ps aux | grep antismash_db_to_jsonl | grep -v grep

# Annotate existing JSONL with contig_edge (single tar pass, ~same runtime as processing)
# Writes asdb5_train_records.annotated.jsonl; verify then mv over original
python scripts/annotate_contig_edge.py \
  --tar    data/antismash_db/asdb5_gbks.tar \
  --input  data/processed/asdb5_train_records.jsonl \
  --output data/processed/asdb5_train_records.annotated.jsonl
# After verifying: mv data/processed/asdb5_train_records.annotated.jsonl \
#                     data/processed/asdb5_train_records.jsonl

# Check contig_edge rate in annotated JSONL
python -c "
import json, sys
total = edge = 0
for line in open('data/processed/asdb5_train_records.jsonl'):
    r = json.loads(line)
    total += 1
    if r.get('contig_edge'): edge += 1
print(f'{edge}/{total} = {100*edge/total:.1f}% contig_edge=True')
"

# Merge + re-split (run after antiSMASH DB processing completes)
cat data/processed/mibig_train_records.jsonl \
    data/processed/asdb5_train_records.jsonl \
  > /tmp/combined.jsonl
PYTHONPATH=src python scripts/split_dataset.py \
  --input /tmp/combined.jsonl \
  --output-dir data/processed/splits_combined \
  --seed 42

# Re-split MIBiG-only (current splits)
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
Sections 4, 5, 8, and 13 as new data is acquired and metrics are validated on GPU.*