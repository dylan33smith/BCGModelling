**De Novo Generation of Synthetic Biosynthetic Gene Clusters**

**Using Genome Language Models**

*Research Plan  |  Dylan Smith  |  May 2026  |  Version 9*

> **Version 9 note:** Sections §3.2–§4 and §8–§11 were refreshed to match the **implemented** repository (`scripts/finetune_evo2_lora.py`, combined splits, gputee H100). Earlier drafts assumed BioNeMo full fine-tuning on an RTX A6000 and claimed LoRA was incompatible with Evo2—that narrative is **superseded**. Canonical operational detail: `PROJECT_GUIDE.md`, `FINETUNE_GUIDE.md`.

# **1\. Overview and Motivation**

Biosynthetic gene clusters (BGCs) are genomic regions in bacteria and fungi where all the genes required to synthesise a specific natural product are physically co-localised. Historically, discovering and deploying these clusters has relied on a parts-based approach: isolate individual genetic elements, characterise them separately, and reassemble them by hand. This is slow, fails to account for emergent regulatory complexity, and is limited to variants of what nature has already evolved.

The emergence of large-scale genomic language models, particularly Evo2, creates a new possibility: learning the full nucleotide-level grammar of BGCs across all domains of life and using that knowledge to generate synthetic clusters that are coherent, novel, and optimised for expression in a chosen chassis organism. This project proposes to fine-tune Evo2 to generate complete, synthesis-ready BGC nucleotide sequences conditioned on a target compound, its **biosynthetic class**, and a chassis organism, and to validate the pipeline through a multi-layered computational evaluation suite designed to be robust in the absence of experimental data.

# **2\. Novel Contribution and Positioning**

## **2.1  Gap in the Literature**

The most directly relevant prior work is Kawano et al. (2025, PLOS Computational Biology), who trained a RoBERTa-based transformer on Pfam functional domain tokens to predict and design BGC architectures, achieving \>60% top-1 domain prediction accuracy on MIBiG-validated clusters and demonstrating experimental validation with the cyclooctatin pathway. However, Kawano et al. operate exclusively at the domain-token abstraction level and cannot generate a single nucleotide. In their own discussion they explicitly state their intention to integrate 'nucleic acid generation tools such as Evo' as a future direction. A targeted literature search (April 2026\) found no published work that has implemented this integration. No existing tool generates nucleotide-level BGC sequences conditioned on target compound identity **and** an explicit, shared biosynthetic-class signal aligned with large-scale BGC corpora.

## **2.2  This Project's Contribution**

This work proposes the first end-to-end computational pipeline for generating synthesis-ready BGC nucleotide sequences conditioned on a **specific target compound, a harmonised biosynthetic class (`COMPOUND_CLASS`), and a chassis organism**. The novel elements are:

* Fine-tuning Evo2 with **hierarchical conditioning tokens**: a shared `COMPOUND_CLASS` token (aligned across MIBiG and antiSMASH) plus a **`COMPOUND` slot that is always present**—MIBiG rows use a normalised metabolite name; bulk antiSMASH and other non-MIBiG rows use a single reserved **`NO_COMPOUND` sentinel**—so the prefix template is fixed and class-level breadth supervision shares the same token positions as compound-specific supervision.  
* Using the Kawano et al. domain-level model as an independent evaluation metric to verify that generated sequences encode the expected functional domain architecture.  
* Testing raw model output without post-processing, cleanly isolating what the model itself has learned about BGC structure, codon usage, and regulatory grammar.  
* An eight-metric computational evaluation suite spanning sequence naturalness, protein foldability, biosynthetic architecture, organism compatibility, synthesis manufacturability, structural novelty, and protein homology — designed to be fully convincing in the absence of wet lab data.  
* Prospective wet lab validation via heterologous expression of three computationally designed BGCs in E. coli, measuring production of three chemically distinct, visually detectable natural products.

# **3\. Technical Pipeline**

## **3.1  Input Specification**

At inference time, the user provides three inputs: (1) a **biosynthetic class** (`COMPOUND_CLASS`, harmonised vocabulary shared with training), (2) a **target compound** (`COMPOUND`, MIBiG-style name for the initial scope), and (3) a **chassis organism**. Scope is restricted to MIBiG compounds in the initial version, providing experimentally validated ground-truth BGC sequences for evaluation; the class for each target is taken from MIBiG annotations and/or the same harmonisation mapping used to build the training corpus (Section 4.2). The chassis organism is encoded using Evo2's existing taxonomic tag system, which forms part of the model's pretraining vocabulary:

|D\_\_BACTERIA;P\_\_PSEUDOMONADOTA;C\_\_GAMMAPROTEOBACTERIA;O\_\_ENTEROBACTERALES;F\_\_ENTEROBACTERIACEAE;G\_\_ESCHERICHIA;S\_\_ESCHERICHIA|

No separate chassis label is needed. This tag conditions the model toward the target organism's codon preferences, GC content, promoter architecture, and RBS context, drawing on knowledge embedded during pretraining on 9.3 trillion nucleotides.

## **3.2  Fine-Tuning Evo2**

### **Architecture**

Evo2 uses the StripedHyena 2 architecture — a hybrid of local convolutional Hyena operators, selective state-space components, and interspersed attention pockets — supporting a 1-million base pair context window at near-linear compute scaling. The 7-billion parameter checkpoint (arcinstitute/evo2\_7b\_262k on HuggingFace) is the primary fine-tuning target.

### **Fine-tuning approach (implemented)**

Training uses **LoRA** via HuggingFace **PEFT** on Evo2 7B (`scripts/finetune_evo2_lora.py`), not NVIDIA BioNeMo and not full-parameter fine-tuning on consumer GPUs. Adapters target named **`nn.Linear`** layers across StripedHyena blocks (`Wqkv`, `out_proj`, `out_filter_dense`, MLP `l1`/`l2`/`l3`). Convolutional Hyena operators and state-space internals remain **frozen**; ~28.7M trainable parameters (~0.44% of ~6.5B).

Older statements that “LoRA is incompatible with Evo2” referred to **first-party BioNeMo packaging** and integration assumptions (public issue-tracker discussion), not whether low-rank adapters can be attached to linear projections in PyTorch. This project demonstrates they can, with Evo2↔DeepSpeed↔PEFT fixes recorded in `FINETUNE_GUIDE.md` §12.6.

**Why not full fine-tuning:** bf16 weights + gradients + AdamW optimizer states already require **≥84 GB VRAM before activations**, which exceeds a single **80 GB H100** and still tends to OOM at moderate sequence lengths without activation checkpointing. Historical multi-GPU ZeRO runs did not change that conclusion for long-context BGC windows.

**Why not BioNeMo:** Docker-oriented, NeMo/Megatron stack aimed at multi-node clusters; poor fit for transparent single-GPU iteration on a shared lab host—see `FINETUNE_GUIDE.md` §1 (“Why BioNeMo is still not used”). BioNeMo fine-tuning tutorials remain a useful **reference** for defaults and sequence packing.

**Operational hardware:** Primary host **gputee** — 1× NVIDIA H100 PCIe 80 GB. Memory sweeps, activation checkpointing defaults, queue scripts, and launch templates: `PROJECT_GUIDE.md` §13, `FINETUNE_GUIDE.md` §§4–6 and §12.7. The **40B Evo2** checkpoint remains out of scope on current hardware.

### **Conditioning Token Format**

Training examples prepend **fixed-order** conditioning tokens to the Evo2 taxonomic tag and BGC nucleotide sequence. **`COMPOUND_CLASS` always appears first**; **`COMPOUND` always appears second** (see sentinel below). This yields an identical three-token prefix layout for MIBiG and non-MIBiG corpora before the taxonomic tag and sequence.

**MIBiG 4.0 examples (compound-specific labels):**

|COMPOUND_CLASS:NRPS| \+ |COMPOUND:indigoidine| \+ \[native producer taxonomic tag\] \+ \[BGC nucleotide sequence\]

(Classes are illustrative; the actual string is taken from the harmonised vocabulary in Section 4.2. Compound bodies are normalised names; Section 4.2.)

**antiSMASH DB mass-download examples — Phase 2+ target (breadth / class supervision with explicit `NO_COMPOUND` sentinel):**

|COMPOUND_CLASS:NRPS| \+ |COMPOUND:NO_COMPOUND| \+ \[native producer or source-organism taxonomic tag\] \+ \[BGC nucleotide sequence\]

**Reserved sentinel `NO_COMPOUND`:** For bulk antiSMASH (and any other non-MIBiG source without a reliable specific-product label), the `COMPOUND` slot uses the **single reserved value `NO_COMPOUND`**. It must **not** collide with real metabolite tokens: normalise MIBiG names so `NO_COMPOUND` never appears as a legitimate compound string (e.g. strip to alphanumeric/underscore and reject or remap that literal). The model then sees a **fixed-width conditioning prefix** (`COMPOUND_CLASS` then `COMPOUND`) on every row, with class-level supervision signalling “no specific product identity” explicitly rather than by absence of a token.

**Project decision — sentinel (default) vs omit (ablation):** The **main pipeline uses `|COMPOUND:NO_COMPOUND|`** on non-MIBiG rows as above. **Optional ablation:** omit the entire `COMPOUND` token for those rows (variable-length prefix) to test whether a shorter prompt or different positional bias affects metrics; report which variant was used in methods. Do not use ad hoc strings such as `unknown` or `NULL` in the main experiments—only `NO_COMPOUND` or the omit ablation.

**Rationale:** Without `COMPOUND_CLASS` on MIBiG rows, compound tokens would be sparse and nearly disjoint from antiSMASH supervision; adding the class on MIBiG aligns both datasets to one token set and lets compound tokens specialise as refinements within a class. The sentinel keeps **token positions aligned** between MIBiG and antiSMASH batches during fine-tuning.

**Inference:** Set `COMPOUND_CLASS` and `COMPOUND` to the target (class from harmonised map, compound as in MIBiG—**never** `NO_COMPOUND` for designed generation toward a named product), and switch the taxonomic tag to the desired chassis (e.g. E. coli). Primary evaluation metric 1 (antiSMASH class) is interpreted against the conditioned `COMPOUND_CLASS`.

The training corpus draws from **MIBiG 4.0** (experimentally validated BGCs with harmonised class labels) and **antiSMASH DB v5** bulk GBKs processed into JSONL (**343,923 records** in the migrated pipeline as of the combined-merge snapshot—see `PROJECT_GUIDE.md` §5). **Phase 1** merged training uses **class-only** conditioning aligned across both sources (MIBiG **`COMPOUND` tokens are omitted** in the merged Phase 1 format—see `PROJECT_GUIDE.md` §7 and phased roadmap §9). The hierarchical **`COMPOUND` slot + `NO_COMPOUND` sentinel** design below remains the **Phase 2+ target** once compound-level supervision is re-enabled without breaking class-only antiSMASH rows.

### **Training protocol**

**Implemented (Phase 1, May 2026):** **Single-pass LoRA** fine-tuning on **`data/processed/splits_combined/{train,val,test}.jsonl`** with DeepSpeed + optional block-level activation checkpointing (default-on for long context on one H100). Scripts, hyperparameters, logging, checkpoints, and the **L=32k pilot → production** gate are documented in `PROJECT_GUIDE.md` §13 and `FINETUNE_GUIDE.md`. There is **no separate embedding-only warm-up stage** in the shipped trainer—LoRA already restricts optimisation to adapter weights.

**Optional future curriculum (not required for Phase 1 launch):** If Phase 2 introduces **`COMPOUND`** tokens again alongside class-only antiSMASH rows, consider revisiting **staged corpora** or **embedding-interface warm-ups** to limit catastrophic forgetting:

* Freeze backbone / train only conditioning-related embedding rows first, then unfreeze with conservative backbone LR (historical BioNeMo-centric framing used parameter groups and callbacks—equivalent patterns are expressible in raw PyTorch `requires_grad` masks).  
* Alternatively compare **full LoRA from step zero** vs staged protocols using generic-genomic perplexity holdouts.

Treat this subsection as **research directions** once compound conditioning lands, not as a prerequisite for the current binary.

# **4\. Data Plan**

## **4.1  Training Datasets**

| Database | Version | BGC Count | Access | Role |
| :---- | :---- | :---- | :---- | :---- |
| MIBiG | 4.0 (Dec 2024\) | 3,059 | mibig.secondarymetabolites.org | Primary fine-tuning: **`COMPOUND_CLASS` \+ `COMPOUND`** per entry, experimentally validated BGCs |
| antiSMASH DB | **v5** (Jan 2026\) | **343,923 records** in migrated JSONL; ~497K BGCs at DB scope | antismash-db.secondarymetabolites.org | Class-only **`COMPOUND_CLASS`** rows for breadth (`PROJECT_GUIDE.md` §4.3); merged with MIBiG for Phase 1 training |
| NPAtlas | 3.0 | 36,545 compounds | npatlas.org | Compound label normalisation and **auxiliary mapping** from structures or names to harmonised `COMPOUND_CLASS` where helpful |

## **4.2  Data Pipeline**

* Download MIBiG 4.0 JSON \+ GenBank files. Each entry contains GenBank accession, genomic coordinates, compound name, BGC class, and organism taxonomy. Unpack the GenBank bundle to a directory of per-cluster `.gbk` files (e.g. `mibig_gbk_4.0/`). The repository preprocessing script `scripts/mibig_to_jsonl.py` reads that directory by default (`--mibig-gbk`); `mibig_gbk_4.0.tar.gz` remains supported as an alternative input if the archive is not unpacked.  
* Download antiSMASH DB **v5** bulk GBKs (173 GB source tar optional if regenerated JSONL not needed — migrated pipeline ships processed JSONL; see `PROJECT_GUIDE.md` §4.3).  
* For each entry: extract BGC nucleotide sequence from GenBank record using stored coordinates. Validate integrity (no frameshifts, complete start/stop codons).  
* **Define a single harmonised `COMPOUND_CLASS` vocabulary** used by both corpora. Map MIBiG’s BGC/product class fields and antiSMASH’s predicted product types into this namespace (manual rules table \+ NPAtlas / ontology support where names disagree). Document every mapping for reproducibility.  
* Build the **`COMPOUND`** label vocabulary from MIBiG compound names (normalised, e.g. via NPAtlas). **Reserve `NO_COMPOUND`** and enforce that no real compound normalises to that literal.  
* Look up organism taxonomy from NCBI Taxonomy using the GenBank accession to populate the Evo2 taxonomic tag format.  
* **Format each training example with the same token order as Section 3.2:** MIBiG rows as |COMPOUND\_CLASS:...| \+ |COMPOUND:\<metabolite\>| \+ taxonomic tag \+ sequence; antiSMASH (and other non-MIBiG) rows as |COMPOUND\_CLASS:...| \+ |COMPOUND:NO\_COMPOUND| \+ taxonomic tag \+ sequence. (**Optional ablation:** omit the `COMPOUND` field entirely for non-MIBiG rows; document if used.) Split into train (80%), validation (10%), test (10%) **stratified by `COMPOUND_CLASS`** (and ensure compound-level leakage checks for MIBiG-held-out compounds).  
* **Apply MIBiG–antiSMASH overlap handling** as in Section 4.4 **before** finalising antiSMASH shards used for pretraining or early-phase fine-tuning.

## **4.3  Data Quality Considerations**

* MIBiG 4.0 has only 3,059 entries for a multi-billion parameter model. The antiSMASH augmentation is essential. **`COMPOUND_CLASS` must be harmonised** between MIBiG and antiSMASH; a poor mapping would weaken the shared backbone the dual-token design depends on. Validation loss on held-out MIBiG entries must be monitored. Any generated sequence with \>95% nucleotide identity to a training example is flagged as a memorisation artefact.  
* BGC sequences in MIBiG come from native producer organisms, not E. coli. The taxonomic tag swap at inference is an empirical bet; evaluation metrics 3 and 7 (CAI and dinucleotide statistics) directly test whether it works.  
* BGC sizes range from a few kilobases to over 100 kb for large hybrid clusters. The 7B checkpoint supports a 262k token context window, covering the vast majority of known BGCs.  
* Compound nomenclature is inconsistent across databases. NPAtlas chemical ontology is used to normalise compound names and, where needed, to **support consistent `COMPOUND_CLASS` assignment** alongside MIBiG and antiSMASH native fields.  
* **MIBiG loci often reappear in the antiSMASH database** (same genome region). That is expected; isolation between training **phases** is not required for correctness, but **held-out evaluation must not be contaminated** — follow Section 4.4.

## **4.4  MIBiG Overlap in antiSMASH and Corpus Deduplication**

Many entries in the antiSMASH bulk export are the **same BGCs** as in MIBiG (matching accession, coordinates, or antiSMASH–MIBiG cross-references where available). Implications:

* **Training-phase overlap is acceptable.** Seeing the same nucleotide sequence first under `COMPOUND_CLASS` \+ `COMPOUND:NO_COMPOUND` (antiSMASH row) and later under `COMPOUND_CLASS` \+ a **real** `COMPOUND` (MIBiG row) does not break the hierarchical design; the compound head gains additional supervision on those loci. It is **not** equivalent to leaking the specific metabolite into the broad-corpus rows because those rows carry the **`NO_COMPOUND` sentinel**, not the MIBiG name (Section 3.2).

* **Evaluation leakage must be prevented.** If a BGC in the **MIBiG validation or test split** appears unchanged in the antiSMASH corpus used during an earlier training phase, the model may have seen that exact sequence (with weaker conditioning), which **inflates** validation metrics and underestimates generalisation error.

**Recommended policy (minimum):** After fixing MIBiG train/validation/test splits, **remove from the antiSMASH training corpus** any record that matches a MIBiG **validation or test** entry, using stable identifiers (MIBiG accession \+ locus coordinates, or database cross-refs) and, as a fallback, **≥99% nucleotide identity** over the BGC interval. Document the rule and **fill the deduplication statistics table** below.

**Stricter optional policy (ablations / narrative):** Remove **all** antiSMASH records that overlap **any** MIBiG entry (including train) if the write-up claims the broad corpus phase “never included MIBiG-derived sequences.” This reduces data volume but yields a cleaner isolation story.

**Not required:** Deduplicating MIBiG train rows from antiSMASH train rows solely to “separate phases”; the default recommendation is **protect MIBiG val/test only**, unless a strict ablation dictates otherwise.

**Deduplication statistics (log for reproducibility; report in methods or supplement):** After running the deduplication step, record at least the following (fill with counts from the actual pipeline run):

| Statistic | Value |
| :---- | :---- |
| antiSMASH records before deduplication | *TBD* |
| Removed — match MIBiG **validation** (identifier-based) | *TBD* |
| Removed — match MIBiG **test** (identifier-based) | *TBD* |
| Removed — **≥99% nucleotide identity** fallback (val/test only) | *TBD* |
| antiSMASH records after deduplication (minimum policy) | *TBD* |
| If strict policy used: removed — overlap **any** MIBiG split | *TBD* |

# **5\. Evaluation Framework**

The evaluation framework is designed to be compelling in the absence of wet lab data, attacking the core question — does this sequence plausibly encode a functional BGC? — from seven independent angles. Each metric tests a different dimension of quality. All tools are open-source and pip/conda installable, enabling a single reproducible Python evaluation pipeline.

The framework is organised into three tiers: Tier 1 (primary) tests what matters most for a BGC paper; Tier 2 (secondary) adds depth and addresses reviewer concerns about naturalness and novelty; Tier 3 (descriptive) provides additional context without serving as pass/fail criteria.

## **5.1  Tier 1 — Primary Metrics**

### **1\. AntiSMASH BGC Class Identification**

AntiSMASH is the field-standard tool for BGC detection and classification, used by virtually every paper in the natural products and synthetic biology literature. If the generated sequence is submitted to antiSMASH and it independently identifies a BGC of the correct compound class with high confidence, that is the single most compelling computational validation available — the tool was not involved in generating the sequence, and its classification logic is entirely independent of Evo2. **The predicted class is compared to the conditioned `COMPOUND_CLASS`** (after mapping antiSMASH’s type labels to the same harmonised vocabulary used at training time, if they differ). Correct class identification provides strong evidence that the generated sequence has all the hallmarks of a real BGC.

pip install antismash   \# then: antismash \--genefinding-tool prodigal sequence.fasta

### **2\. Functional Domain Recovery — pyhmmer \+ Pfam 37.0**

Predicted ORFs are translated and scanned against Pfam 37.0 (June 2024, 21,979 families, hosted via InterPro) using pyhmmer — Cython bindings to HMMER3 that run fully in-process. Standard E-value threshold: 1e-10, as used in Kawano et al. The recovered domain architecture is compared against the expected obligate domains for the **conditioned `COMPOUND_CLASS`** (e.g., KS, AT, ACP for PKS; C, A, T domains for NRPS; terpene synthase for terpenoids). The Kawano et al. domain-level model provides an independent second reference for the expected domain set. A generated sequence passes this metric if all obligate domains for the compound class are recovered.

pip install pyhmmer   \# Pfam-A.hmm: ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/

### **3\. Protein Structure Prediction — ESMFold**

Protein structure prediction is increasingly a standard validation step in computational biology. ESMFold is run on each predicted ORF from the generated BGC. Two sub-checks are performed. First, foldability: a predicted local distance difference test (pLDDT) score above 70 for the majority of residues indicates a reliably predicted structure, confirming the sequence encodes a folded protein rather than a disordered or nonsensical polypeptide. Second, structural homology: Foldseek is used to search the predicted structure against the PDB and AlphaFold database. Generated biosynthetic enzymes should structurally resemble their natural counterparts even when primary sequence similarity is modest, providing evidence of genuine functional conservation.

pip install fair-esm   \# ESMFold local inference

pip install foldseek   \# or: conda install \-c conda-forge foldseek

### **4\. Synthesis Feasibility — DNA Chisel**

All generated sequences are checked against commercial synthesis constraints using DNA Chisel's built-in constraint classes, modelled on Twist Bioscience's published guidelines. Sequences failing any constraint cannot be ordered for synthesis and are excluded from experimental validation candidates. Pass rate across all generated sequences is a secondary quality metric. Specific constraints enforced:

* Global GC content: 25–65%  
* Local GC content in any 50 bp window: 35–65%  
* Maximum GC differential between highest and lowest 50 bp stretch: 52%  
* No homopolymer runs \>= 10 bp (hard synthesis failure at \>= 14 bp)  
* No direct or inverted repeats \> 20 bp  
* No CcdB toxin sequence

  pip install 'dnachisel\[reports\]'

## **5.2  Tier 2 — Secondary Metrics**

### **5\. Sequence Naturalness — Evo2 Perplexity**

Perplexity is the model's own measure of how surprising a sequence is. A sequence with low perplexity under the base pretrained Evo2 model (i.e., without the **`COMPOUND_CLASS` / `COMPOUND` fine-tuning**) looks like a real genome sequence in the model's learned representation. This directly measures naturalness and is the same metric used by Evo Designer (which reports per-nucleotide entropy as a quality score for generated sequences). Perplexity is computed by passing the generated sequence through the pretrained Evo2 7B checkpoint and averaging the negative log-likelihood over all nucleotide positions. Generated sequences are compared against the distribution of perplexity values for real BGC sequences from MIBiG as a reference.

### **6\. Structural Novelty and Coherence — BiG-SCAPE 2.0**

BiG-SCAPE 2.0 (published Nature Communications, January 2026\) calculates pairwise distances between BGCs based on protein domain content, order, copy number, and sequence identity. Two complementary checks are performed. Novelty: the generated BGC should have a distance \> 0.3 from all MIBiG training examples, confirming the model is generalising rather than reproducing training data. Coherence: the generated BGC should have a distance \< 0.7 from the MIBiG reference for the target compound, confirming it occupies the correct chemical neighbourhood within the known BGC similarity landscape. The BiG-SCAPE distance metric is well established and directly interpretable by reviewers in the BGC field.

conda install \-c bioconda bigscape

### **7\. Organism Compatibility — CAI, GC Content, and Dinucleotide Statistics**

A composite organism compatibility check assesses whether the generated sequence looks like an E. coli gene rather than a gene from the native producer organism. Three statistics are computed. Codon Adaptation Index (CAI) relative to E. coli K-12 highly expressed gene reference weights is calculated using CodonTransformer's CodonEvaluation module; a score above 0.7 indicates adequate codon compatibility. Overall GC content is compared against the E. coli K-12 genome average (\~51%). Most sensitively, dinucleotide frequencies — particularly CpG suppression, which is a strong signature of real bacterial genomic sequences — are computed for the generated sequence and compared against the E. coli reference distribution. Systematic deviation in dinucleotide statistics is a sensitive indicator that the taxonomic tag swap is not fully working. Together these three statistics form a composite organism compatibility score.

pip install CodonTransformer

## **5.3  Tier 3 — Descriptive Analysis (Non-Pass/Fail)**

### **8\. Protein Sequence Homology — MMseqs2**

MMseqs2 is used to search all predicted ORF sequences against UniRef50. Generated biosynthetic enzymes should be recognisably related to known enzyme families (30–60% identity range is the twilight zone — clearly related in function, genuinely novel in sequence), confirming the model has learned real biosynthetic chemistry. Very high identity (\>95%) to a training sequence is flagged as memorisation. Complete absence of detectable homology to any known protein is flagged as suspicious. This metric is reported descriptively rather than as a pass/fail criterion.

conda install \-c bioconda mmseqs2

## **5.4  Evaluation Summary**

| \# | Metric | Tool | Tier | Pass Criterion | What It Tests |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | BGC class identification | AntiSMASH | Primary | Predicted class matches **conditioned `COMPOUND_CLASS`** (harmonised label set) | Field-standard independent validation that sequence has BGC hallmarks |
| 2 | Domain architecture | pyhmmer \+ Pfam 37.0 | Primary | Obligate domains at E \< 1e-10 | Functional coherence; biosynthetic logic learned by model |
| 3 | Protein foldability \+ structural homology | ESMFold \+ Foldseek | Primary | pLDDT \> 70; fold matches natural enzyme | Proteins are folded and structurally recognisable despite sequence novelty |
| 4 | Synthesis feasibility | DNA Chisel | Primary | Pass all Twist/IDT constraints | Sequence is manufacturable; synthesis-ready |
| 5 | Sequence naturalness | Evo2 perplexity (base model) | Secondary | Perplexity within MIBiG reference range | Generated sequence looks like a real genome sequence |
| 6 | Structural novelty \+ coherence | BiG-SCAPE 2.0 | Secondary | BiG-SCAPE distance 0.3-0.7 from MIBiG ref | Novel but architecturally coherent; not memorised |
| 7 | Organism compatibility | CodonTransformer \+ dinucleotide stats | Secondary | CAI \> 0.7; GC \~51%; dinucleotide freq. matches E. coli | Taxonomic tag conditioning working; E. coli compatible |
| 8 | Protein sequence homology | MMseqs2 vs UniRef50 | Descriptive | 30-60% identity to known enzymes | Novelty vs. memorisation; learned real biosynthetic chemistry |

The eight metrics collectively span every dimension a reviewer can reasonably ask about: biological validity (metrics 1, 2, 3), practical manufacturability (metric 4), language model quality (metric 5), BGC-specific novelty (metric 6), expression host compatibility (metric 7), and sequence novelty (metric 8). No single metric is decisive on its own, but convergence across all eight constitutes a strong computational case.

# **6\. Target Compounds for Experimental Validation**

Three compounds are selected spanning distinct BGC chemical classes (each mapped to a single **`COMPOUND_CLASS`** in the harmonised training vocabulary), all expressible in E. coli BL21(DE3), all easily detectable by colorimetry and HPLC, and all absent from E. coli's native metabolism.

| Compound | BGC Class | Colour | Detection | Genes | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Violacein | Shikimate / oxidative | Blue-purple | HPLC 575 nm | vioABCDE (5) | Primary benchmark. Five well-characterised genes with extensive heterologous expression literature in E. coli. |
| Carotenoid (zeaxanthin) | Terpenoid (MEP) | Yellow-orange | HPLC 450 nm | crtEBIYZ (\~6) | Isoprenoid chemistry. E. coli has native MEP pathway precursor supply. Represented in MIBiG as 17 carotenoid BGC entries (BGC0000633–BGC0000650). Extensive metabolic engineering literature for benchmarking titer. |
| Indigoidine | NRPS | Bright blue | HPLC 612 nm | BpsA \+ sfp (2) | Minimal cluster (two genes). Represents NRPS chemistry. Simplest proof-of-concept; cleanest test of whether the model produces functional sequences at minimum complexity. |

# **7\. Wet Lab Validation**

## **7.1  Experimental Design**

Wet lab validation is planned but contingent on securing an appropriate collaborator. The computational evaluation suite in Section 5 is designed to be independently convincing for publication. If wet lab data is obtained, it strengthens the paper substantially and enables higher-tier journal submission.

For each of the three target compounds, one BGC sequence **generated under the same conditioning template used at inference-time fine-tune evaluation** (`COMPOUND_CLASS` required; **`COMPOUND` only once Phase 2+ compound conditioning is active**) plus E. coli taxonomic tag, passing all Tier 1 computational metrics, is submitted to a commercial gene synthesis provider (Twist Bioscience or IDT), cloned into pETDuet-1 or equivalent, transformed into E. coli BL21(DE3), induced with IPTG, and culture extracts are analysed by HPLC with UV-Vis detection at the compound-specific wavelength.

## **7.2  Controls**

* Negative control: E. coli BL21(DE3) with empty vector — confirms no background production.  
* Reference positive control: the native MIBiG reference BGC sequence expressed under identical conditions — establishes baseline titer for comparison.  
* Authentic chemical standard: commercially sourced, for HPLC calibration and identity confirmation.

## **7.3  Lab Requirements**

Standard academic molecular biology or microbiology labs are sufficient. Required: E. coli transformation and shake-flask culture, BL21-DE3 expression strain, HPLC with UV-Vis detection. All three compounds are non-hazardous BSL-1 metabolites. Gene synthesis is outsourced; the lab performs standard transformation, fermentation, and measurement. Violacein and indigoidine produce blue/purple colony colour on LB plates before any HPLC analysis is needed.

# **8\. Open Questions and Risks**

## **8.1  Adapter fine-tuning vs full-parameter training**

**Resolved for Phase 1:** LoRA via PEFT on Evo2 7B is **implemented and validated** on lab hardware (smoke + production-like preflight through **L=65,536** tokens under activation checkpointing on one H100—see `FINETUNE_GUIDE.md` §12.7.1). Full-parameter fine-tuning remains **disallowed by memory** on single-GPU 80 GB class hardware without architectural changes.

**Residual risk:** Adapter capacity (`--lora-r`, currently 16) may limit expressiveness vs full FT; monitor validation loss and raise rank if training plateaus early (`PROJECT_GUIDE.md` §13 Future enhancements).

## **8.2  Small Fine-Tuning Dataset**

MIBiG 4.0 has ~3k validated clusters—the bulk of supervision comes from **antiSMASH DB v5–derived JSONL** merged into **`splits_combined`** (`PROJECT_GUIDE.md` §5). **Phase 1** uses **shared `COMPOUND_CLASS`** across MIBiG and antiSMASH with **no `COMPOUND` token** on merged rows (MIBiG compound names withheld—see `PROJECT_GUIDE.md` §9 Phase 1). When Phase 2 reintroduces **`COMPOUND`** on MIBiG while keeping antiSMASH class-only, revisit memorisation checks (BiG-SCAPE distance, MMseqs2 identity) and optionally the **optional staged curriculum** sketched in §3.2.

* **Catastrophic forgetting** risk rises when new conditioning slots enter mid-project; LoRA-only Phase 1 already limits backbone drift. Tier 2 perplexity vs generic genome holdouts remains a diagnostic if perplexity collapses after conditioning changes.

## **8.3  Cross-Organism Generalisation**

BGCs in MIBiG come from native producer organisms, not E. coli. Training pairs **`COMPOUND_CLASS`** (and, in Phase 2+, **`COMPOUND`**) with non-E. coli taxonomic tags where applicable. Metric 7 (CAI, GC content, dinucleotide statistics) tests whether the E. coli tag swap at inference yields organism-compatible sequences. If CAI values are systematically low, consider codon-optimising MIBiG sequences to E. coli usage **before** fine-tuning (future experiment).

## **8.4  Compute access**

**Resolved:** Primary GPU is **1× NVIDIA H100 PCIe 80 GB** on **gputee**. Queue helpers (`scripts/queue_h100_smoke.sh`, `scripts/queue_h100_preflight.sh`) coordinate idle-GPU launches on a shared host. HuggingFace cache and run outputs live on **`/data2`** (`HF_HOME=/data2/ds85/hf_cache`, `--output-dir /data2/ds85/bgcmodel_runs/...`) because **`/home` is disk-constrained**. Multi-day wall-clock estimates (~2.7 days at L=32k, ~5.3 days at L=65k for two epochs at measured throughput) are in `FINETUNE_GUIDE.md` §4.

**Outstanding:** First **L=32k pilot** on combined splits must still validate val/checkpoint/resume paths before locking a multi-day production launch (`PROJECT_GUIDE.md` §13 NEXT).

## **8.5  Timeline Risk from Competing Work**

Kawano et al. explicitly named Evo integration as their next direction. A literature alert on 'Evo2 biosynthetic gene cluster', 'genomic language model BGC design', and 'nucleotide-level BGC generation' should be set up immediately. Completing the computational pipeline and at least one wet lab result before submission reduces exposure to being scooped.

# **9\. Publication Strategy**

The intended publication is a full research article. The narrative arc is: (1) establish the gap — no existing tool generates nucleotide-level BGC sequences with **harmonised biosynthetic-class conditioning** tied to large-scale BGC corpora at Evo2 scale; (2) present the **implemented** fine-tuning pipeline (**Phase 1:** shared **`COMPOUND_CLASS`** across MIBiG + antiSMASH DB v5 merged JSONL, LoRA + DeepSpeed + activation checkpointing on **gputee** H100—see `PROJECT_GUIDE.md`; **Phase 2+:** hierarchical **`COMPOUND`** slot as in §3.2 examples); (3) computational validation across eight metrics; (4) optional wet lab proof-of-concept.

Target journals: ACS Synthetic Biology and Metabolic Engineering with positive wet lab results; PLOS Computational Biology or Bioinformatics for a computational-only submission. The eight-metric evaluation suite is designed to be sufficient for a computational-only paper at a strong journal.

# **10\. Computational Environment**

The **`bgcmodel`** environment is defined at the project root (`environment.yml`). On **gputee**, create and activate with **micromamba** (not conda):

```bash
micromamba activate bgcmodel
```

> **`environment.yml` alone does not produce a working env on a fresh create** — the pip phase fails on `flash-attn` before `torch` is installed. Follow **`FINETUNE_GUIDE.md` §2** for the working sequence (torch → prebuilt flash-attn wheel → `micromamba env update` → `deepspeed`/`peft`/`wandb`).

Included (conda-forge/bioconda): antiSMASH 8.x tooling, pyhmmer, DNA Chisel, BiG-SCAPE 2.0, MMseqs2, Foldseek, Prodigal, Biopython, PyYAML. **GPU stack** (PyTorch + flash-attn + transformers + Evo2 + DeepSpeed + PEFT + WandB): installed per §2 of the fine-tune guide. **`PYTHONPATH=src`** when running pipeline scripts.

Reference antiSMASH databases: `download-antismash-databases`

# **11\. Immediate Next Steps**

Operational priorities are tracked in **`PROJECT_GUIDE.md` §13**. As of May 2026:

1. **Run the L=32,768 pilot** on **`data/processed/splits_combined/{train,val}.jsonl`** with production-like settings (`--batch-size 4`, `--grad-accum 32`, activation checkpointing on, **no** `--smoke-pad-to-max-seq-len`), enough steps to cross **`--val-every 250`** and **`--save-every 500`**, output under **`/data2/ds85/bgcmodel_runs/`**.
2. **`scripts/check_data_eval_readiness.py --json`** before launch; archive snapshot beside the run (`PROJECT_GUIDE.md` §13.2).
3. Execute **`FINETUNE_GUIDE.md` §12.8** resume sanity check on a pilot checkpoint before committing wall-clock to the full 2-epoch job.
4. **Choose production `--max-seq-len`:** **32,768** (conservative margin) vs **65,536** (stretch; ~74 GB peak in production-like preflight—see `FINETUNE_GUIDE.md` §4).
5. **Literature alerts** on Evo2 + BGC generation / nucleotide-level design (competitive-awareness).
6. **Obtain Kawano et al. domain-model weights** (`umemura-m/bgc-transformer`) if not already cached—for independent M2-style evaluation.
7. **Wet lab collaborator** search remains parallel to computational milestones (`PROJECT_GUIDE.md` §13).

Research-facing tasks that remain independent of the trainer:

* Harmonised **`COMPOUND_CLASS`** mapping audits and publication-ready counts for Methods (deduplication statistics table §4.4 — fill from pipeline logs when regenerating splits).
* Standalone evaluation harness regression tests on MIBiG positives vs shuffled negatives before trusting metrics on generated sequences.

# **Key References**

Brixi, G. et al. Genome modeling and design across all domains of life with Evo 2\. Nature (2026). doi:10.1038/s41586-026-10176-5

Kawano, T. et al. A novel transformer-based platform for the prediction and design of biosynthetic gene clusters for (un)natural products. PLOS Comput. Biol. (2025). doi:10.1371/journal.pcbi.1013181


Terlouw, B.R. et al. MIBiG 3.0: a community-driven effort to annotate experimentally validated biosynthetic gene clusters. Nucleic Acids Res. 51, D603-D610 (2023).

Blin, K. et al. antiSMASH 7.0: new and improved predictions for detection, regulation, and visualisation. Nucleic Acids Res. 51, W46-W50 (2023).

van der Hooft, J.J.J. et al. BiG-SCAPE 2.0 and BiG-SLiCE 2.0: scalable, accurate and interactive sequence clustering of metabolic gene clusters. Nat. Commun. (2026). doi:10.1038/s41467-026-68733-5

Peng, A. et al. CodonTransformer: a multispecies codon optimizer using context-aware neural networks. Nat. Commun. (2025). doi:10.1038/s41467-025-58588-7

Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023). \[ESMFold\]

van Kempen, M. et al. Fast and accurate protein structure search with Foldseek. Nat. Biotechnol. 42, 243-246 (2024). \[Foldseek\]

Steinegger, M. and Soeding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat. Biotechnol. 35, 1026-1028 (2017).

NVIDIA BioNeMo Evo2 fine-tuning tutorial (reference only — operational stack is PyTorch + DeepSpeed + PEFT): [BioNeMo Evo2 fine-tuning tutorial](https://docs.nvidia.com/bionemo-framework/2.6/user-guide/examples/bionemo-evo2/fine-tuning-tutorial/)