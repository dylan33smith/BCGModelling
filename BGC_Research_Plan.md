**De Novo Generation of Synthetic Biosynthetic Gene Clusters**

**Using Genome Language Models**

*Research Plan  |  Dylan Smith  |  April 2026  |  Version 3*

# **1\. Overview and Motivation**

Biosynthetic gene clusters (BGCs) are genomic regions in bacteria and fungi where all the genes required to synthesise a specific natural product are physically co-localised. Historically, discovering and deploying these clusters has relied on a parts-based approach: isolate individual genetic elements, characterise them separately, and reassemble them by hand. This is slow, fails to account for emergent regulatory complexity, and is limited to variants of what nature has already evolved.

The emergence of large-scale genomic language models, particularly Evo2, creates a new possibility: learning the full nucleotide-level grammar of BGCs across all domains of life and using that knowledge to generate synthetic clusters that are coherent, novel, and optimised for expression in a chosen chassis organism. This project proposes to fine-tune Evo2 to generate complete, synthesis-ready BGC nucleotide sequences conditioned on a target compound and chassis organism, and to validate the pipeline through a multi-layered computational evaluation suite designed to be robust in the absence of experimental data.

# **2\. Novel Contribution and Positioning**

## **2.1  Gap in the Literature**

The most directly relevant prior work is Kawano et al. (2025, PLOS Computational Biology), who trained a RoBERTa-based transformer on Pfam functional domain tokens to predict and design BGC architectures, achieving \>60% top-1 domain prediction accuracy on MIBiG-validated clusters and demonstrating experimental validation with the cyclooctatin pathway. However, Kawano et al. operate exclusively at the domain-token abstraction level and cannot generate a single nucleotide. In their own discussion they explicitly state their intention to integrate 'nucleic acid generation tools such as Evo' as a future direction. A targeted literature search (April 2026\) found no published work that has implemented this integration. No existing tool generates nucleotide-level BGC sequences conditioned on target compound identity.

## **2.2  This Project's Contribution**

This work proposes the first end-to-end computational pipeline for generating synthesis-ready BGC nucleotide sequences conditioned on a specific target compound and chassis organism. The novel elements are:

* Fine-tuning Evo2 with compound-identity conditioning tokens, extending its existing taxonomic tag system to include target compound labels from MIBiG.  
* Using the Kawano et al. domain-level model as an independent evaluation metric to verify that generated sequences encode the expected functional domain architecture.  
* Testing raw model output without post-processing, cleanly isolating what the model itself has learned about BGC structure, codon usage, and regulatory grammar.  
* A seven-layer computational evaluation suite spanning sequence naturalness, protein foldability, biosynthetic architecture, organism compatibility, synthesis manufacturability, structural novelty, and secondary validation — designed to be fully convincing in the absence of wet lab data.  
* Prospective wet lab validation via heterologous expression of three computationally designed BGCs in E. coli, measuring production of three chemically distinct, visually detectable natural products.

# **3\. Technical Pipeline**

## **3.1  Input Specification**

At inference time, the user provides two inputs: (1) a target compound from the MIBiG database, and (2) a chassis organism. Scope is restricted to MIBiG compounds in the initial version, providing experimentally validated ground-truth BGC sequences for evaluation. The chassis organism is encoded using Evo2's existing taxonomic tag system, which forms part of the model's pretraining vocabulary:

|D\_\_BACTERIA;P\_\_PSEUDOMONADOTA;C\_\_GAMMAPROTEOBACTERIA;O\_\_ENTEROBACTERALES;F\_\_ENTEROBACTERIACEAE;G\_\_ESCHERICHIA;S\_\_ESCHERICHIA|

No separate chassis label is needed. This tag conditions the model toward the target organism's codon preferences, GC content, promoter architecture, and RBS context, drawing on knowledge embedded during pretraining on 9.3 trillion nucleotides.

## **3.2  Fine-Tuning Evo2**

### **Architecture**

Evo2 uses the StripedHyena 2 architecture — a hybrid of local convolutional Hyena operators, selective state-space components, and interspersed attention pockets — supporting a 1-million base pair context window at near-linear compute scaling. The 7-billion parameter checkpoint (arcinstitute/evo2\_7b\_262k on HuggingFace) is the primary fine-tuning target.

### **Critical Finding: LoRA Incompatibility**

LoRA (Low-Rank Adaptation) is not currently compatible with Evo2. LoRA assumes linear weight matrices in transformer attention blocks, but StripedHyena 2's convolutional operators and state-space components do not fit this assumption. As of April 2026, LoRA support for Evo2 is an open feature request (NVIDIA BioNeMo issue \#884) but has not been implemented. Full fine-tuning is therefore required, demonstrated to run on a single RTX A6000 (45 GB VRAM) in approximately one hour in bfloat16 precision using the NVIDIA BioNeMo framework. The 40B checkpoint requires multiple H100 GPUs and is deferred to a later project phase. University HPC or cloud GPU access (NSF ACCESS, AWS, Google Cloud) is required.

### **Conditioning Token Format**

Each training example is formatted as a compound label prepended to the Evo2 taxonomic tag and BGC nucleotide sequence:

|COMPOUND:violacein| \+ \[native producer taxonomic tag\] \+ \[BGC nucleotide sequence\]

The training corpus draws from MIBiG 4.0 (3,059 experimentally validated BGCs with known compound labels, released December 2024\) as the primary source and antiSMASH database v4 (231,534 BGCs) labelled at compound class level for breadth and overfitting prevention. At inference, the compound label is set to the target compound and the taxonomic tag is switched to the E. coli tag. The cross-organism generalisation implicit in this tag swap is directly tested by the evaluation suite.

# **4\. Data Plan**

## **4.1  Training Datasets**

| Database | Version | BGC Count | Access | Role |
| :---- | :---- | :---- | :---- | :---- |
| MIBiG | 4.0 (Dec 2024\) | 3,059 | mibig.secondarymetabolites.org | Primary fine-tuning: compound-specific labels, experimentally validated BGCs |
| antiSMASH DB | v4 (2024) | 231,534 | antismash-db.secondarymetabolites.org | Secondary fine-tuning: compound class labels, breadth and generalisation |
| IMG-ABC | v5.0 | 330,884 | img.jgi.doe.gov/abc | Optional augmentation if coverage gaps identified |
| NPAtlas | 3.0 | 36,545 compounds | npatlas.org | Compound label normalisation via chemical ontology |

## **4.2  Data Pipeline**

* Download MIBiG 4.0 JSON \+ GenBank files. Each entry contains GenBank accession, genomic coordinates, compound name, BGC class, and organism taxonomy.  
* Download antiSMASH v4 BGC data as GenBank/FASTA bulk exports.  
* For each entry: extract BGC nucleotide sequence from GenBank record using stored coordinates. Validate integrity (no frameshifts, complete start/stop codons).  
* Build the compound label vocabulary from MIBiG compound names; use compound class labels (e.g., |COMPOUND\_CLASS:NRPS|) for antiSMASH entries lacking specific compound annotation.  
* Look up organism taxonomy from NCBI Taxonomy using the GenBank accession to populate the Evo2 taxonomic tag format.  
* Format each training example: compound token \+ taxonomic tag \+ BGC nucleotide sequence. Split into train (80%), validation (10%), test (10%) stratified by compound class.

## **4.3  Data Quality Considerations**

* MIBiG 4.0 has only 3,059 entries for a multi-billion parameter model. The antiSMASH augmentation is essential. Validation loss on held-out MIBiG entries must be monitored. Any generated sequence with \>95% nucleotide identity to a training example is flagged as a memorisation artefact.  
* BGC sequences in MIBiG come from native producer organisms, not E. coli. The taxonomic tag swap at inference is an empirical bet; evaluation metrics 3 and 7 (CAI and dinucleotide statistics) directly test whether it works.  
* BGC sizes range from a few kilobases to over 100 kb for large hybrid clusters. The 7B checkpoint supports a 262k token context window, covering the vast majority of known BGCs.  
* Compound nomenclature is inconsistent across databases. NPAtlas chemical ontology is used to normalise compound names and map to structural classes.

# **5\. Evaluation Framework**

The evaluation framework is designed to be compelling in the absence of wet lab data, attacking the core question — does this sequence plausibly encode a functional BGC? — from seven independent angles. Each metric tests a different dimension of quality. All tools are open-source and pip/conda installable, enabling a single reproducible Python evaluation pipeline.

The framework is organised into three tiers: Tier 1 (primary) tests what matters most for a BGC paper; Tier 2 (secondary) adds depth and addresses reviewer concerns about naturalness and novelty; Tier 3 (descriptive) provides additional context without serving as pass/fail criteria.

## **5.1  Tier 1 — Primary Metrics**

### **1\. AntiSMASH BGC Class Identification**

AntiSMASH is the field-standard tool for BGC detection and classification, used by virtually every paper in the natural products and synthetic biology literature. If the generated sequence is submitted to antiSMASH and it independently identifies a BGC of the correct compound class with high confidence, that is the single most compelling computational validation available — the tool was not involved in generating the sequence, and its classification logic is entirely independent of Evo2. Correct class identification provides strong evidence that the generated sequence has all the hallmarks of a real BGC.

pip install antismash   \# then: antismash \--genefinding-tool prodigal sequence.fasta

### **2\. Functional Domain Recovery — pyhmmer \+ Pfam 37.0**

Predicted ORFs are translated and scanned against Pfam 37.0 (June 2024, 21,979 families, hosted via InterPro) using pyhmmer — Cython bindings to HMMER3 that run fully in-process. Standard E-value threshold: 1e-10, as used in Kawano et al. The recovered domain architecture is compared against the expected obligate domains for the target BGC class (e.g., KS, AT, ACP for PKS; C, A, T domains for NRPS; terpene synthase for terpenoids). The Kawano et al. domain-level model provides an independent second reference for the expected domain set. A generated sequence passes this metric if all obligate domains for the compound class are recovered.

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

Perplexity is the model's own measure of how surprising a sequence is. A sequence with low perplexity under the base pretrained Evo2 model (i.e., without the compound-label fine-tuning) looks like a real genome sequence in the model's learned representation. This directly measures naturalness and is the same metric used by Evo Designer (which reports per-nucleotide entropy as a quality score for generated sequences). Perplexity is computed by passing the generated sequence through the pretrained Evo2 7B checkpoint and averaging the negative log-likelihood over all nucleotide positions. Generated sequences are compared against the distribution of perplexity values for real BGC sequences from MIBiG as a reference.

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

### **9\. RBS Strength Profile — OSTIR (Descriptive)**

OSTIR (Open Source Translation Initiation Rate predictor) calculates the thermodynamic free energy of ribosome-mRNA binding for each gene in the generated cluster, providing a translation initiation rate profile across all pathway genes. This is reported descriptively: are the predicted expression levels roughly balanced across the pathway? For the violacein case specifically, the RBS profile is compared against Zhang et al.'s (2021) empirically validated finding that VioB and VioE are rate-limiting and require strong RBS sequences. This serves as a useful sanity check and generates interesting discussion material but is not used as a pass/fail criterion, given OSTIR's 53% within-2-fold accuracy and the difficulty of interpreting relative expression predictions without experimental data.

## **5.4  Evaluation Summary**

| \# | Metric | Tool | Tier | Pass Criterion | What It Tests |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | BGC class identification | AntiSMASH | Primary | Correct class detected | Field-standard independent validation that sequence has BGC hallmarks |
| 2 | Domain architecture | pyhmmer \+ Pfam 37.0 | Primary | Obligate domains at E \< 1e-10 | Functional coherence; biosynthetic logic learned by model |
| 3 | Protein foldability \+ structural homology | ESMFold \+ Foldseek | Primary | pLDDT \> 70; fold matches natural enzyme | Proteins are folded and structurally recognisable despite sequence novelty |
| 4 | Synthesis feasibility | DNA Chisel | Primary | Pass all Twist/IDT constraints | Sequence is manufacturable; synthesis-ready |
| 5 | Sequence naturalness | Evo2 perplexity (base model) | Secondary | Perplexity within MIBiG reference range | Generated sequence looks like a real genome sequence |
| 6 | Structural novelty \+ coherence | BiG-SCAPE 2.0 | Secondary | BiG-SCAPE distance 0.3-0.7 from MIBiG ref | Novel but architecturally coherent; not memorised |
| 7 | Organism compatibility | CodonTransformer \+ dinucleotide stats | Secondary | CAI \> 0.7; GC \~51%; dinucleotide freq. matches E. coli | Taxonomic tag conditioning working; E. coli compatible |
| 8 | Protein sequence homology | MMseqs2 vs UniRef50 | Descriptive | 30-60% identity to known enzymes | Novelty vs. memorisation; learned real biosynthetic chemistry |
| 9 | RBS expression profile | OSTIR | Descriptive | Reported for discussion only | TIR balance across pathway; compared to Zhang et al. violacein benchmark |

The nine metrics collectively span every dimension a reviewer can reasonably ask about: biological validity (metrics 1, 2, 3), practical manufacturability (metric 4), language model quality (metric 5), BGC-specific novelty (metric 6), expression host compatibility (metric 7), sequence novelty (metric 8), and expression grammar (metric 9). No single metric is decisive on its own, but convergence across all nine constitutes a strong computational case.

# **6\. Target Compounds for Experimental Validation**

Three compounds are selected spanning distinct BGC chemical classes, all expressible in E. coli BL21(DE3), all easily detectable by colorimetry and HPLC, and all absent from E. coli's native metabolism.

| Compound | BGC Class | Colour | Detection | Genes | Notes |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Violacein | Shikimate / oxidative | Blue-purple | HPLC 575 nm | vioABCDE (5) | Primary benchmark. Zhang et al. (2021) RBS engineering data enables direct comparison of OSTIR predictions vs. experimental results. Five well-characterised genes. |
| Lycopene | Terpenoid (MEP) | Red-pink | HPLC 470 nm | crtEBIY (\~6) | Isoprenoid chemistry. E. coli has native MEP pathway precursor supply. Extensive metabolic engineering literature for benchmarking titer. |
| Indigoidine | NRPS | Bright blue | HPLC 612 nm | BpsA \+ sfp (2) | Minimal cluster (two genes). Represents NRPS chemistry. Simplest proof-of-concept; cleanest test of whether the model produces functional sequences at minimum complexity. |

# **7\. Wet Lab Validation**

## **7.1  Experimental Design**

Wet lab validation is planned but contingent on securing an appropriate collaborator. The computational evaluation suite in Section 5 is designed to be independently convincing for publication. If wet lab data is obtained, it strengthens the paper substantially and enables higher-tier journal submission.

For each of the three target compounds, one BGC sequence passing all Tier 1 computational metrics is submitted to a commercial gene synthesis provider (Twist Bioscience or IDT), cloned into pETDuet-1 or equivalent, transformed into E. coli BL21(DE3), induced with IPTG, and culture extracts are analysed by HPLC with UV-Vis detection at the compound-specific wavelength.

## **7.2  Controls**

* Negative control: E. coli BL21(DE3) with empty vector — confirms no background production.  
* Reference positive control: the native MIBiG reference BGC sequence expressed under identical conditions — establishes baseline titer for comparison.  
* Authentic chemical standard: commercially sourced, for HPLC calibration and identity confirmation.

## **7.3  Lab Requirements**

Standard academic molecular biology or microbiology labs are sufficient. Required: E. coli transformation and shake-flask culture, BL21-DE3 expression strain, HPLC with UV-Vis detection. All three compounds are non-hazardous BSL-1 metabolites. Gene synthesis is outsourced; the lab performs standard transformation, fermentation, and measurement. Violacein and indigoidine produce blue/purple colony colour on LB plates before any HPLC analysis is needed.

# **8\. Open Questions and Risks**

## **8.1  Full Fine-Tuning Without LoRA**

LoRA is not currently compatible with Evo2's StripedHyena 2 architecture. Full fine-tuning requires at minimum a single A6000 (45 GB VRAM), demonstrated via the BioNeMo tutorial. University HPC or cloud GPU access must be secured before beginning. If LoRA support is added to BioNeMo for Evo2 (open issue \#884), this constraint will be lifted.

## **8.2  Small Fine-Tuning Dataset**

MIBiG 4.0 has 3,059 entries. The antiSMASH augmentation at compound class level is important for generalisation. Validation loss on held-out MIBiG entries must be monitored. The BiG-SCAPE distance check and MMseqs2 sequence identity check are the primary defences against memorisation.

## **8.3  Cross-Organism Generalisation**

BGCs in MIBiG come from native producer organisms, not E. coli. The compound labels are paired with non-E. coli taxonomic tags during training. Metric 7 (CAI, GC content, dinucleotide statistics) directly tests whether the E. coli tag swap at inference produces organism-compatible sequences. If CAI values are systematically low, a mitigation is to codon-optimise MIBiG sequences to E. coli usage before fine-tuning so the model learns E. coli-compatible BGC sequences directly.

## **8.4  Compute Access**

Full fine-tuning of Evo2 7B requires approximately 45 GB VRAM. University HPC or cloud GPU allocation should be secured before the fine-tuning phase. BioNeMo on a single A6000 is the minimum viable setup; the 40B model is deferred.

## **8.5  Timeline Risk from Competing Work**

Kawano et al. explicitly named Evo integration as their next direction. A literature alert on 'Evo2 biosynthetic gene cluster', 'genomic language model BGC design', and 'nucleotide-level BGC generation' should be set up immediately. Completing the computational pipeline and at least one wet lab result before submission reduces exposure to being scooped.

# **9\. Publication Strategy**

The intended publication is a full research article. The narrative arc is: (1) establish the gap — no existing tool generates nucleotide-level BGC sequences conditioned on target compound identity; (2) present the fine-tuning pipeline; (3) demonstrate rigorous computational validation across nine independent metrics spanning biological validity, manufacturability, naturalness, novelty, and host compatibility; (4) prospectively demonstrate experimental proof-of-concept with three chemically diverse compounds in E. coli.

Target journals: ACS Synthetic Biology and Metabolic Engineering with positive wet lab results; PLOS Computational Biology or Bioinformatics for a computational-only submission. The nine-metric evaluation suite is designed to be sufficient for a computational-only paper at a strong journal.

# **10\. Immediate Next Steps**

* Set up literature alerts on 'Evo2 biosynthetic gene cluster', 'genomic language model BGC design', and 'nucleotide-level BGC generation' immediately.  
* Download MIBiG 4.0 and antiSMASH v4 data; write and test the data formatting pipeline on a small subset before committing to full preprocessing.  
* Build the complete evaluation pipeline as a standalone Python script and run it against known MIBiG BGC sequences (as positive controls) and randomly shuffled sequences (as negative controls) to validate each metric's discriminative power before generating any model output.  
* Apply for HPC cluster or cloud GPU access (A6000-class minimum) running NVIDIA BioNeMo with the Evo2 7B checkpoint.  
* Obtain the Kawano et al. domain-level model weights (GitHub: umemura-m/bgc-transformer) for use as the independent domain-architecture evaluation metric.  
* Begin fine-tuning on antiSMASH compound class dataset first before the smaller MIBiG compound-specific dataset.  
* Identify wet lab collaborator with E. coli expression capability and HPLC access, treating this as parallel to computational work rather than sequential.

# **Key References**

Brixi, G. et al. Genome modeling and design across all domains of life with Evo 2\. Nature (2026). doi:10.1038/s41586-026-10176-5

Kawano, T. et al. A novel transformer-based platform for the prediction and design of biosynthetic gene clusters for (un)natural products. PLOS Comput. Biol. (2025). doi:10.1371/journal.pcbi.1013181

Zhang, Y. et al. Direct RBS engineering of the biosynthetic gene cluster for efficient productivity of violaceins in E. coli. Microb. Cell Fact. 20, 38 (2021).

Terlouw, B.R. et al. MIBiG 3.0: a community-driven effort to annotate experimentally validated biosynthetic gene clusters. Nucleic Acids Res. 51, D603-D610 (2023).

Blin, K. et al. antiSMASH 7.0: new and improved predictions for detection, regulation, and visualisation. Nucleic Acids Res. 51, W46-W50 (2023).

van der Hooft, J.J.J. et al. BiG-SCAPE 2.0 and BiG-SLiCE 2.0: scalable, accurate and interactive sequence clustering of metabolic gene clusters. Nat. Commun. (2026). doi:10.1038/s41467-026-68733-5

Peng, A. et al. CodonTransformer: a multispecies codon optimizer using context-aware neural networks. Nat. Commun. (2025). doi:10.1038/s41467-025-58588-7

Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023). \[ESMFold\]

van Kempen, M. et al. Fast and accurate protein structure search with Foldseek. Nat. Biotechnol. 42, 243-246 (2024). \[Foldseek\]

Steinegger, M. and Soeding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat. Biotechnol. 35, 1026-1028 (2017).

NVIDIA BioNeMo Evo2 Fine-Tuning Tutorial: docs.nvidia.com/bionemo-framework/2.6/user-guide/examples/bionemo-evo2/fine-tuning-tutorial/