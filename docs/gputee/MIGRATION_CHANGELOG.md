# Migration Changelog — trojai → gputee

A running, chronological log of every change made when porting this project
from the old `trojai` host (4× NVIDIA A40, 48 GB each) to the new `gputee`
host (1× NVIDIA H100 PCIe, 80 GB). Every entry records **what** changed,
**why**, and **what was deliberately kept unchanged**.

Format: newest at the bottom. Each entry is one atomic change (file or
closely related set of files).

---

## Ground truth — hardware context used for every decision below

| | trojai (old) | gputee (new) |
|---|---|---|
| GPUs | 4× NVIDIA A40, 48 GB (46 GB usable) | 1× NVIDIA H100 PCIe, 80 GB |
| Driver / CUDA runtime | not recorded | 575.64.03 / CUDA 12.9 |
| Host CPU | not recorded | 2× AMD EPYC 9124 (16c), 32c/64t |
| Host RAM | not recorded | 376 GiB |
| Home disk free | n/a | 74 GiB free on 1.8 TB (96% used) — **tight** |
| Conda / mamba | conda available | only `micromamba` (at `/usr/local/bin/micromamba`) |
| Shared box? | n/a | yes, but treated as dedicated per user direction |

Key derived constraints for gputee:
1. Full-parameter Evo2 7B fine-tune still does **not** fit in 80 GB
   (14 weights + 14 grads + 56 AdamW ≥ 84 GB, before activations). LoRA remains
   the correct path.
2. LoRA peak on 4× A40 was 23.2 GB/rank at L=1024 (measured). On 1× H100 the
   corresponding forward memory will be **larger**, not smaller, because ZeRO-2
   no longer shards the 84 GB of replicated state across 4 GPUs — the single
   H100 now carries everything that was previously replicated. But 84 GB is
   the full-FT upper bound; under LoRA the replicated state is only the frozen
   base weights (~14 GB bf16) + LoRA params + activations. Still comfortably
   within 80 GB for the L values of interest.
3. DeepSpeed ZeRO-2 at `world_size=1` does no sharding and provides no
   memory benefit. It still works as a DDP-like wrapper. Keeping it for now
   is safer than refactoring the whole script; it can be removed in a later
   pass if desired.
4. `CUDA_VISIBLE_DEVICES=0,1,2,3` and `deepspeed --num_gpus=4` are wrong on
   gputee and must be changed everywhere.

---

## Changes

### 1. Docs folder split (docs/trojai, docs/gputee)

**Files:** created `docs/trojai/`, `docs/gputee/`; moved `README.md`,
`PROJECT_GUIDE.md`, `FINETUNE_GUIDE.md`, `BGC_Research_Plan.md` from repo
root to `docs/trojai/` via `git mv`; copied the same files into
`docs/gputee/`; wrote a new short root `README.md` pointing at both folders.

**Why:** User request. Preserves the trojai docs as a historical snapshot
and gives us a clean place to update gputee-specific guidance without
rewriting history.

**Kept unchanged:** everything inside `docs/trojai/`. Do not edit these.

---

### 2. Full audit — every A40-era item found, with per-item decision

This is the inventory I built by reading every code file and document. Each row
is classified as **KEEP** (still correct, addresses a non-hardware quirk),
**DOC-ONLY** (the code is fine but the doc text is wrong), or **EDIT** (the
artefact itself is A40-specific and must change).

#### Code

| # | Location | A40-era thing | Classification | Rationale |
|---|---|---|---|---|
| C1 | `scripts/finetune_evo2.py` docstring `Launch` block (lines 7–13) | `CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 …` | **EDIT** | Simply wrong on 1 GPU. The rest of the script is world-size-agnostic; only the example invocation needs to change. |
| C2 | `scripts/finetune_evo2_lora.py` docstring `LoRA vs full fine-tune` (lines 6–16) and `Launch` block (lines 29–37) | Justifies LoRA by "full FT OOMs on 4× A40"; `CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 …` | **EDIT** | The *conclusion* (use LoRA) is still right on 1× H100 80 GB because full-FT needs ≥ 84 GB even before activations. But the *reasoning* referenced the wrong hardware. The launch command is wrong. |
| C3 | Per-rank GPU masking in both scripts (setting `CUDA_VISIBLE_DEVICES=<local_rank>`, `LOCAL_RANK=0`, `args.local_rank=0`) | Was added to stop Evo2's vortex loader from auto-sharding across 4 visible GPUs | **KEEP** | Addresses an Evo2 loader behaviour, not an A40 behaviour. At world_size=1 it is a no-op (`local_rank=0` → `CUDA_VISIBLE_DEVICES=0` → same as if unset). Removing it would re-open the multi-GPU hazard on any future multi-GPU box and provides zero gputee benefit. |
| C4 | DeepSpeed ZeRO-2 config in `build_ds_config` in both scripts | `zero_optimization.stage: 2` with full sharding knobs; designed to split optimizer/grad state across 4 ranks | **KEEP (with doc caveat)** | At `world_size=1` ZeRO-2 shards nothing and degrades to a bf16 + grad-accum + scheduler wrapper. It still works; rewriting to raw PyTorch/accelerate would be a refactor and a new smoke-test cycle, which the user explicitly did not ask for ("don't add features… just update so we no longer need A40-era workarounds"). Document the reality in the guide; leave the code alone. |
| C5 | Hyperparameter defaults in both scripts (`batch_size=4`, `grad_accum=8`, `lr=5e-5`/`1e-5`) | Chosen so 4 GPUs × 4 × 8 = 128 effective batch | **KEEP (defaults); add gputee recommended overrides to the guide** | Defaults are CLI-overridable; changing them implicitly would violate the user's "don't make unapproved changes" rule. On gputee with `world_size=1` the default effective batch becomes 4 × 8 = 32, which the user can restore by passing `--grad-accum 32`. The gputee FINETUNE_GUIDE will recommend the override explicitly. |
| C6 | The three Evo2↔DS bug fixes (non-contiguous Wqkv, inference-mode tensors, WarmupCosineLR API) | Fixed during A40 smoke tests | **KEEP** | All three address Evo2/peft/DeepSpeed quirks. None are A40-specific. All still required on gputee. |
| C7 | Step counter uses `model_engine.global_steps` (off-by-one fix) | Same | **KEEP** | DeepSpeed semantics, not A40. |
| C8 | `scripts/antismash_db_to_jsonl.py`, `annotate_contig_edge.py`, `requirements.txt` — header lines that say `conda activate bgcmodel` | Assumes conda | **KEEP** | Still works on any host with conda. gputee uses micromamba; the equivalent command is `micromamba activate bgcmodel`. Documented in the guide; editing every script header would be churn. |
| C9 | `src/bgc_pipeline/evaluation.py` GPU device selection (`"cuda" if torch.cuda.is_available() else "cpu"`) | — | **KEEP** | Already generic; picks whatever single GPU is available. Works on gputee without change. |

#### Documentation

| # | Location | A40-era thing | Classification | Rationale |
|---|---|---|---|---|
| D1 | `docs/gputee/PROJECT_GUIDE.md` §2 repo tree | No `docs/` directory shown | **EDIT** | Reflect the new docs/ split. |
| D2 | `docs/gputee/PROJECT_GUIDE.md` §3.1 install | Uses `conda env create …` only | **EDIT** | Add micromamba equivalent (gputee has no conda). |
| D3 | `docs/gputee/PROJECT_GUIDE.md` §3.3 GPU stack | "Verified GPU setup on this server" lists 4× A40, `CUDA_VISIBLE_DEVICES=0,1,2,3` | **EDIT** | Replace with gputee hardware block. |
| D4 | `docs/gputee/PROJECT_GUIDE.md` §4.1 data table | Claims NPAtlas, UniRef50, asdb5_gbks.tar all present | **EDIT** | All three are **not** on gputee. Mark with precise "not migrated" status + list of downstream steps each blocks. |
| D5 | `docs/gputee/PROJECT_GUIDE.md` §12 known issues — "LoRA fine-tuning (not full fine-tune)" entry | Justification is "Full fine-tune OOMs on 4× A40" | **EDIT** | Reframe: on 1× H100 80 GB full-FT still OOMs (84 GB floor) so LoRA is still the right call, but the reason needs updating. |
| D6 | `docs/gputee/PROJECT_GUIDE.md` §12 — "antiSMASH DB processing: disk filled mid-run" entry | trojai-era anecdote | **KEEP (relabel)** | Still a useful warning (gputee's /home is 96% used — the hazard is worse). Keep the entry but mark it as trojai-era evidence. |
| D7 | `docs/gputee/PROJECT_GUIDE.md` §13 — NEXT task "Per-block activation checkpointing" | Memory numbers are for 4× A40 | **EDIT** | On 1× H100 80 GB + LoRA, the activation-checkpointing question is re-opened: at L=32 k the StripedHyena filter alone is ~14 GB × (L/1024) = big, but H100 has 80 GB and no 4× replication tax. Still a nice-to-have but its *necessity* needs re-evaluation after a single-GPU smoke test. Mark accordingly. |
| D8 | `docs/gputee/FINETUNE_GUIDE.md` §1 "Available GPUs" + "Why all 4 GPUs are required" + "Why BioNeMo is not used" | Entire memory rationale for 4× A40 | **EDIT** | Rewrite for 1× H100 80 GB. Keep the conclusion (LoRA) and explain why full-FT is still not viable on a single 80 GB GPU. |
| D9 | `docs/gputee/FINETUNE_GUIDE.md` §2 "Verified: all 4× A40 visible" | — | **EDIT** | Update to 1× H100. |
| D10 | `docs/gputee/FINETUNE_GUIDE.md` §4 memory-at-runtime table (L=1024 → 23.2 GB etc.) | A40 measurements | **EDIT (preserve as historical)** | Move to a "trojai historical — A40" sub-table, add an empty "gputee pending measurement" sub-table. Do **not** invent numbers; mark as pending. |
| D11 | `docs/gputee/FINETUNE_GUIDE.md` §4 "Effective batch size and throughput" | 128 seq across 4 GPUs | **EDIT** | Recompute for world_size=1; state the recommended `--grad-accum 32` override to recover the original 128 effective batch. |
| D12 | `docs/gputee/FINETUNE_GUIDE.md` §4 "Steps and time estimate" | "18–36 hours" assumed 4 GPUs | **EDIT** | Re-estimate: 1× H100 per-token throughput in bf16 is ~2× an A40. Losing 4× parallelism but gaining ~2× per GPU → ~2× slower overall at the same effective batch. Revised estimate needs a smoke benchmark before any firm number is quoted. Mark the number as pending and give the reasoning. |
| D13 | `docs/gputee/FINETUNE_GUIDE.md` §6 "Pre-flight checks" + all launch and smoke-test commands | 4 GPUs, `--num_gpus=4` | **EDIT** | Rewrite for 1 GPU. |
| D14 | `docs/gputee/FINETUNE_GUIDE.md` §7 warning signs table — "One GPU at 0% utilisation" row | Not applicable with only one GPU | **EDIT** | Drop the row (or replace with a gputee-relevant symptom). |
| D15 | `docs/gputee/FINETUNE_GUIDE.md` §8 training trajectory table | Describes loss/steps assumptions; not strictly hardware-specific | **KEEP** | Trajectory shapes are about the model+data, not the GPU. |
| D16 | `docs/gputee/FINETUNE_GUIDE.md` §12 smoke-test findings | All bugs & memory numbers were observed on 4× A40 | **EDIT (relabel)** | Very valuable history — keep the text verbatim but prepend a clear "These findings are from the trojai smoke tests on 4× A40. A re-run on gputee is pending." note. Do not rewrite the historical record. |
| D17 | `docs/gputee/README.md` | Was simply a pointer to `PROJECT_GUIDE.md` in the old root layout | **EDIT (small)** | Update paths to match the new folder layout. |
| D18 | `docs/gputee/BGC_Research_Plan.md` | Contains one off-hand reference to an RTX A6000 / BioNeMo as a compute option | **KEEP** | This is a research-plan document (compute options landscape), not a hardware reference for this server. Out of scope for a hardware migration. Leave unchanged. |

#### Deliberately out of scope

- **No hyperparameter changes applied to the scripts.** User said "don't add new features". Overrides are documented in the guide instead.
- **No DeepSpeed removal.** Would be a materially new training path that needs its own smoke-test; explicitly out of scope.
- **No FP8 / Transformer-Engine / flash-attn bump.** These would be H100 performance features, not migration fixes.
- **No data-movement changes.** Per user instruction, data is untouched. The two empty `data/npatlas/` and `data/uniref50/` directories and the missing `asdb5_gbks.tar` are recorded in the guide as "not migrated" with downstream-impact notes only.

---

*(individual change entries appended below as each edit is applied)*

### 3. `scripts/finetune_evo2.py` — module docstring

**Change:** rewrote the top-of-file Launch block. Added a "Status" paragraph
stating that this script is a reference implementation: it OOMed on trojai
and will not fit on a single 80 GB H100 either (84 GB base > 80 GB). The
launch example now shows `deepspeed --num_gpus=1` for gputee and keeps the
old trojai `--num_gpus=4` line as a commented historical record. Redirected
the docs pointer from `FINETUNE_GUIDE.md` (root) to
`docs/gputee/FINETUNE_GUIDE.md`.

**Why:** The old example could not be copy-pasted on gputee (no device
cuda:1–3 exists). The status paragraph makes it explicit that this script
is not the right path on gputee either — use the LoRA script.

**Code body untouched.** All the Evo2 ↔ DeepSpeed fixes (per-rank GPU
masking, non-contiguous Wqkv, inference-mode tensor cloning,
WarmupCosineLR params, `global_steps` fix) remain identical; they address
Evo2/DS quirks that persist across hardware.

### 4. `scripts/finetune_evo2_lora.py` — module docstring

**Change:** rewrote the "LoRA vs full fine-tune" explanation under a new
heading "Why LoRA (not full fine-tune)". New text explains the 84 GB base
memory floor and why that rules out full-parameter fine-tuning on both
trojai (4× A40) and gputee (1× H100 80 GB). Points at
`docs/trojai/FINETUNE_GUIDE.md` §12 for the historical A40 smoke-test
evidence and `docs/gputee/FINETUNE_GUIDE.md` §1 for the gputee analysis.

Updated the Launch block: `deepspeed --num_gpus=1` on gputee, with an
explicit note that `--grad-accum 32` is the recommended override to
preserve the original 128-sequence effective batch (default `grad_accum=8`
× `world_size=4` = 128 on trojai; × `world_size=1` = 32 on gputee).

**Why:** same as C1/D3 in the audit. Script body untouched, including LoRA
config, bug fixes, ZeRO-2 config, and default hyperparameters.

**Deliberately NOT changed:** the script's `DEFAULTS` dict
(batch_size=4, grad_accum=8, lr=5e-5, max_seq_len=32768). These stay at
their trojai values; the gputee override is a CLI flag so nothing is done
implicitly.

### 5. `docs/gputee/PROJECT_GUIDE.md` §1 preamble

**Change:** added a "Hardware context" paragraph to the opening that states
the gputee hardware, points at the trojai copy for history, and points at
this changelog. Changed the top-of-file "last updated" date to today.

**Why:** makes clear at the top which host the guide describes; prevents
future confusion.

### 6. `docs/gputee/PROJECT_GUIDE.md` §2 repository layout

**Change:** rewrote the ASCII tree. Added the new top-level `docs/`
directory with both `gputee/` and `trojai/` subfolders enumerated.
Added the `splits_combined/` entry (was missing from the trojai doc).
Added a per-line "✅ on gputee" / "⚠️ NOT migrated" status on the data
subtree so the layout doubles as a migration status map.

**Why:** the old tree claimed the guide file was at repo root
(`PROJECT_GUIDE.md`). After the docs/ split that's wrong, and the tree
would mislead anyone clicking through the repo.

### 7. `docs/gputee/PROJECT_GUIDE.md` §3.1 environment setup

**Change:** replaced the `conda env create` block with a dual
micromamba/conda block and added a one-line note that the `bgcmodel`
env has **not** been created on gputee yet.

**Why:** `conda` is not installed on gputee; only `micromamba` is. The
project will fail at the first `conda activate bgcmodel` without this.

### 8. `docs/gputee/PROJECT_GUIDE.md` §3.3 GPU stack

**Change:** replaced the "Verified GPU setup on this server" bullet list
(which described 4× A40 and `CUDA_VISIBLE_DEVICES=0,1,2,3`) with the
gputee bullets: 1× H100 PCIe 80 GB, driver 575.64.03 / CUDA 12.9, no
device-masking needed, `deepspeed --num_gpus=1`. Added an explicit
paragraph explaining that the pinned `torch==2.5.1+cu124` wheel is
forward-compatible with the CUDA 12.9 driver, so no torch bump is
required. Added a disk-pressure callout (`/home` at 96% used). Pointed
at `docs/trojai/PROJECT_GUIDE.md` §3.3 for the archived A40 setup.

**Why:** the old bullets mis-describe the hardware and would generate a
broken launch command if followed literally.

### 9. `docs/gputee/PROJECT_GUIDE.md` §3.4 UniRef50 section

**Change:** replaced "Already downloaded — 29 GB at data/uniref50/" with
a block documenting that the directory is empty on gputee, the disk check
to run first, and the `mmseqs databases UniRef50 …` command. Updated the
"Important" callout to prefer `micromamba activate bgcmodel`.

**Why:** reflect the actual gputee filesystem state.

### 10. `docs/gputee/PROJECT_GUIDE.md` §4.1 data table

**Change:** replaced the single-status-column "Downloaded/Not" table
with a two-status-column table (gputee status + downstream blocks).
Marked NPAtlas, UniRef50, and the 173 GB `asdb5_gbks.tar` as **not
migrated**, noting which downstream steps each blocks. Added a new
bulleted "To un-block each missing item" section with the exact `wget`
/ `mmseqs` command for each, and flagged that the 173 GB tar will not
fit on the current `/home` mount (74 GiB free).

**Why:** the old table claimed all three artefacts were present. The
two empty directories (`data/npatlas/`, `data/uniref50/`) and the
48 MB beta tar in `data/antismash_db/` (instead of the full 173 GB)
would have caused silent failures the moment someone tried to re-run
the SMILES audit, Metric 8, or the antiSMASH DB pipeline.

### 11. `docs/gputee/PROJECT_GUIDE.md` §4.3 antiSMASH DB v5 block

**Change:** retitled the section header from "… downloaded and processed ✅"
to "… processed output migrated; source tar not migrated". Replaced the
`# Already downloaded to data/antismash_db/` comments with the explicit
present/missing status on gputee. Added the note that re-processing is
a future task, disk-bound by /home. Updated `conda activate` → `micromamba
activate` in the sample command.

**Why:** same as §4.1 but specific to the antiSMASH DB section, which has
its own standalone set of example commands that were factually wrong.

### 12. `docs/gputee/PROJECT_GUIDE.md` §12 known issues — LoRA design-decision row

**Change:** rewrote the rationale for "LoRA fine-tuning (not full fine-tune)".
Old text said "Full fine-tune OOMs on 4× A40"; new text gives the 84 GB base
memory floor (14 weights + 14 grads + 56 AdamW) as the reason, then explains
why this rules out full-FT on **both** trojai (can only reach 84 GB via
ZeRO-2 sharding across 4 ranks, activations push over the A40 46 GB budget)
**and** gputee (no second GPU to shard to, 84 GB > 80 GB even before
activations). Conclusion (LoRA is correct) is unchanged.

**Why:** the old wording implied LoRA was only needed because of a 4× A40
constraint, which would suggest LoRA could be dropped on a bigger GPU.
On a single 80 GB H100 this is still wrong; LoRA is mandatory.

**Deliberately KEPT:** the "antiSMASH DB processing: disk filled mid-run"
row. It is still valuable as a warning — gputee's /home is 96% used,
making the hazard worse, not better.

### 13. `docs/gputee/PROJECT_GUIDE.md` §13 "Ready to start" task list

**Change:** reordered the task list for the gputee context. The first
task is now **"single-GPU LoRA smoke benchmark"** (re-measure peak GPU
memory at a series of L values on the H100 before committing to a
production config). The old "Per-block activation checkpointing" task
is demoted from "required before production launch" to "conditional —
implement only if the H100 benchmark shows OOM at the target L", because
the old memory math (which motivated the 112 → 18–22 GB estimate) was
for 4× A40 and no longer applies. Added the `--grad-accum 32` hint to
the fine-tune row so the 128-sequence effective batch is preserved
at world_size=1. Added NPAtlas + UniRef50 as prerequisites to the
"Full 8-metric evaluation" row.

**Why:** the old "NEXT" task was defined by memory math that no longer
holds. We don't have evidence that activation checkpointing is required
on gputee until a smoke benchmark is run. Until then, the right NEXT is
the benchmark itself.

### 14. `docs/gputee/FINETUNE_GUIDE.md` §1 hardware & memory constraints

**Change:** replaced the 4× A40 "Available GPUs" block with a 1× H100
PCIe 80 GB block. Rewrote "Why all 4 GPUs are required" as
"Why full-parameter fine-tuning still doesn't fit", showing the 84 GB
baseline and a small table of alternatives (LoRA, ZeRO-3 offload,
8-bit AdamW, FP8) with explicit risk/effort classification. The
conclusion — LoRA is the right choice — is unchanged. Rewrote the
StripedHyena filter activations discussion, preserving the trojai
L=1024 → 23.2 GB / L=4096 → OOM history as **trojai measurement** and
explicitly flagging that those numbers do not transfer to gputee's
single 80 GB GPU. Updated the "Why BioNeMo is not used" paragraph to
match the single-GPU context.

**Why:** this is the guide's central memory-analysis section; it
drove most downstream decisions on trojai and was the most load-bearing
piece of stale hardware reasoning.

### 15. `docs/gputee/FINETUNE_GUIDE.md` §2 required installations

**Change:** noted that the `bgcmodel` env is not yet created on gputee
and must be (re-)built via micromamba; replaced the "Verified: all 4× A40
visible" line with a runnable verification snippet that checks torch,
transformers, peft, deepspeed, evo2, flash_attn and asserts
`torch.cuda.device_count() == 1` with device name `NVIDIA H100 PCIe`.

**Why:** trojai's "verified all 4× A40" line is factually wrong on gputee
and users would otherwise skip the verification step.

### 16. `docs/gputee/FINETUNE_GUIDE.md` §4 effective batch / memory / timing

**Change:** rewrote "Effective batch size and throughput" to compute
both the default-world_size=1 result (32) and the recommended override
(`--grad-accum 32` → 128) explicitly; stated that the script `DEFAULTS`
dict is NOT being changed. Replaced the "Memory at runtime" single-table
with two sub-tables — **trojai historical** (preserves the 23.2 GB / OOM
numbers verbatim) and **gputee pending** (TBD rows for L=1024, 4096,
8192, 32768). Rewrote the "Steps and time estimate" block with the new
steps-per-epoch math under `--grad-accum 32` and a reasoned-but-flagged
"~1.5–3 days, pending benchmark" wall-clock estimate.

**Why:** the old memory table (all A40 numbers) is the single most
dangerous page if read as authoritative for gputee. Splitting it into
historical vs pending and refusing to invent new numbers is the honest
call.

### 17. `docs/gputee/FINETUNE_GUIDE.md` §6 launch section + pre-flight

**Change:** rewrote pre-flight checks (`nvidia-smi` now expects one GPU;
added HF-cache check and disk-pressure callout specific to gputee's
96%-full `/home`). Rewrote the smoke-test, production-launch, and
resume blocks to use `deepspeed --num_gpus=1` (no `CUDA_VISIBLE_DEVICES`).
Bumped `--grad-accum` to 32 in the production launch. Added an inline
note explaining the grad-accum change and what the "Fixed N tensors"
line should look like on gputee (1/4 of the trojai count). Revised the
"⚠️ DO NOT launch production yet" callout above §6.1 to reflect that
the gating item is now the §12.7 smoke benchmark rather than activation
checkpointing per se.

**Why:** the old commands (`CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed
--num_gpus=4 …`) will fail immediately on gputee. Pre-flight checks
should reflect the actual hardware and the actual disk pressure.

### 18. `docs/gputee/FINETUNE_GUIDE.md` §7 warning signs table

**Change:** removed the "One GPU at 0% utilisation → DDP init failure →
verify all 4 GPUs visible" row (not applicable at world_size=1). Added
a new row "Peak GPU memory approaching 80 GB" pointing at the
`--max-seq-len` / activation-checkpointing levers.

**Why:** the removed row could never fire on gputee; the new row is
the most likely actual memory-related failure mode.

### 19. `docs/gputee/FINETUNE_GUIDE.md` §12 smoke-test findings

**Change:** added a "Scope note" callout at the top of §12 explaining
that §§12.1–12.6 are **trojai** findings preserved verbatim. The bug
fixes documented there remain in effect; the memory numbers do not
transfer. Added a new **§12.7 Gputee smoke-benchmark plan (pending)**
section that defines the concrete procedure (one-line bash loop over
L ∈ {1 k, 4 k, 8 k, 16 k, 32 k}), an empty result table to fill in, and
a decision rule for whether per-block activation checkpointing is
required.

**Why:** §12 was the most historically-valuable part of the guide
(seven distinct bug reports + fixes). Rewriting it would destroy the
record; fronting it with a clear scope note and adding a new §12.7 for
the gputee work preserves both layers.

### 20. `docs/gputee/README.md`

**Change:** rewrote the 17-line stub to describe the gputee docs set,
point at `MIGRATION_CHANGELOG.md`, and show the micromamba create
command alongside the conda equivalent.

**Why:** the old stub pointed at `PROJECT_GUIDE.md` with no hardware
context and referenced `conda env create` as the default — both stale
under the new folder layout.

---

## Post-migration findings (surfaced during first gputee run, 2026-04-22)

These entries document changes made **after** the initial migration pass
above, during the first attempt to build the env and run the smoke
benchmark on gputee. All are "fixes to things we didn't know were
broken until we ran it", not new features.

### 21. Environment-install procedure documented in FINETUNE_GUIDE §2

**Change:** Rewrote §2 ("Required installations") to include the
working fresh-install sequence: (a) run `micromamba env create -f
environment.yml` (conda side succeeds, pip side crashes — expected),
(b) pip install torch alone with the cu124 index-url, (c) pip install
the prebuilt flash-attn wheel from Dao-AILab's GitHub releases,
(d) rerun `micromamba env update -n bgcmodel -f environment.yml` to
install the rest of the pip list, (e) pip install the three
training-only deps (deepspeed, peft, wandb) that aren't in the env
file.

**Why:** `environment.yml` lists both `torch==2.5.1+cu124` and
`flash-attn==2.7.4.post1` in the same pip block. pip resolves the
whole block before installing anything, and flash-attn's `setup.py`
does `import torch` at build time, so it crashes with
`ModuleNotFoundError: No module named 'torch'`. Falling back to
pip's "build from source" path then hits a second bug (EXDEV on
cross-filesystem `os.rename` between `/tmp` and the pip cache on
`/home`, because flash-attn's setup.py uses `os.rename` instead of
`shutil.move`). Installing the prebuilt wheel sidesteps both.

This is a genuine env-file bug that would also have bitten a fresh
`conda env create` on trojai if it had ever been recreated. Left
`environment.yml` itself unchanged (still a valid lock-style export);
only documented the manual sequence. Updating the yaml would involve
removing `flash-attn` from the pip list and adding an install-order
hook, which is more invasive than the doc fix warrants.

**Files:** `docs/gputee/FINETUNE_GUIDE.md` §2 (rewritten).

### 22. `HF_HOME=/data2/ds85/hf_cache` — cache off /home

**Change:** Documented the `HF_HOME` environment variable pointing at
`/data2/ds85/hf_cache` as a required setup step in `FINETUNE_GUIDE.md`
§2 ("Storage layout on gputee"). Added to every launch command example
in §6 (`export HF_HOME=...` prefixed before `deepspeed`). Used this
path for the first Evo2 7B model download on gputee.

**Why:** `/home` on gputee is near-full (~30 GB free). The Evo2 7B
checkpoint is ~14 GB and HuggingFace typically needs ~2× during
download for temp files. Downloading into the default
`~/.cache/huggingface/` would have pushed `/home` to 100% and broken
the shared server for everyone. `/data2` is a 7 TB XFS volume with
~1.5 TB free, shared but not disk-constrained.

**Kept unchanged:** the `evo2` library's HF download mechanism itself.
It already honours `HF_HOME` without modification.

**Files:** `docs/gputee/FINETUNE_GUIDE.md` §2 (new "Storage layout"
subsection), §6 (three launch command examples updated), §12.7 (smoke
benchmark loop updated).

### 23. `--output-dir /data2/ds85/bgcmodel_runs/<run_name>` — runs off /home

**Change:** Changed the recommended `--output-dir` path in every
documented launch command from `checkpoints/phase1_lora` (under the
repo, on `/home`) to `/data2/ds85/bgcmodel_runs/phase1_lora` (on
`/data2`). Also updated the top-of-file docstring launch example in
`scripts/finetune_evo2_lora.py` to use the new path, and explained
the change in the "Notes" block under that example. No code logic
changed — `--output-dir` has always been CLI-configurable; this is
purely the documented default pattern.

**Why:** same `/home` disk pressure as #22, plus training produces
checkpoints, plots, offline wandb logs, and sample FASTAs under the
output dir. Even with the checkpoint-size fix (entry #24 below), a
long run produces hundreds of MB of logs + plots + samples that
should not compete for the ~30 GB free on `/home`.

**Files:** `docs/gputee/FINETUNE_GUIDE.md` §6 (all three launch
examples: smoke, production, resume), §11 (path in the file-layout
diagram), §12.7 (smoke benchmark loop); `scripts/finetune_evo2_lora.py`
docstring only (no logic change).

### 24. `exclude_frozen_parameters=True` on DeepSpeed checkpoint save

**Change:** Added `exclude_frozen_parameters=True` to the
`model_engine.save_checkpoint(...)` call inside
`save_lora_checkpoint` in `scripts/finetune_evo2_lora.py`.

**Why:** The pre-fix save wrote a 25 GB `mp_rank_00_model_states.pt`
per checkpoint — the full 6.5B Evo2 base weights, serialised at every
`--save-every` step. With `keep_last_ckpts=5` that was ~127 GB in
flight for a LoRA run, plus another ~25 GB for the `best/` copy. None
of those bytes are useful: the frozen base is loaded fresh from the
HF cache via `Evo2("evo2_7b")` on every run, and the LoRA adapter is
restored from `checkpoints/step_N/adapter/` via
`PeftModel.from_pretrained`. The only genuinely useful content in
`mp_rank_00_model_states.pt` is the scheduler state + client_state
(step counter, best_val_loss), which is a few KB.

Post-fix footprint per checkpoint: ~55 MB adapter + ~330 MB
ZeRO-partitioned optimizer state + ~few MB model state = ~390 MB.
A full production run at default retention drops from ~150 GB to
~3 GB in flight.

**Resume correctness:** unchanged. DeepSpeed handles
`exclude_frozen_parameters=True` symmetrically on load; the frozen
base comes from the init path (`Evo2("evo2_7b")`) rather than the
checkpoint, and the scheduler + trainable-param optimizer state still
round-trip correctly. Manual resume-correctness verification is
pending (documented in §12.8) and is the last prerequisite before a
long run.

**Discovery context:** surfaced on the first gputee smoke run when a
3-step L=1024 benchmark consumed 20 GB of `/home` for
`checkpoints/smoke_L1024/checkpoints/step_3_final/mp_rank_00_model_states.pt`.
The fix was reviewed against DeepSpeed's documented semantics for
LoRA-compatible checkpointing; the flag was added to DeepSpeed for
exactly this case.

**Files:** `scripts/finetune_evo2_lora.py` (`save_lora_checkpoint`
function, 1 call-site change + comment);
`docs/gputee/FINETUNE_GUIDE.md` §11 (file-layout diagram + new size
table), §12.8 (new section documenting the fix).

### 25. `final_adapter/` is now a copy of `step_N_final/adapter/`

**Change:** In `scripts/finetune_evo2_lora.py`, replaced the
end-of-training `model_engine.module.save_pretrained(final_adapter)`
call with `shutil.copytree(checkpoints/step_N_final/adapter,
final_adapter)`. Added `import shutil`.

**Why:** `save_lora_checkpoint` has already written the adapter to
`checkpoints/step_N_final/adapter/` (via peft's `save_pretrained`)
just before this code runs. The pre-fix script immediately re-wrote
the same bytes to `final_adapter/` via a second, independent peft
call. Two code paths serialising the same LoRA adapter is a drift
hazard (version skew in peft's format would produce two inconsistent
copies). `shutil.copytree` of the already-written bytes guarantees
identity, and skips the ~1-second overhead of re-serialising through
peft's internals.

**Trade-off considered:** could have made `final_adapter/` a symlink
to `step_N_final/adapter/` to save 55 MB of disk. Decided against —
a user who later trims the `checkpoints/` subtree (common cleanup
action) would silently break the documented inference-time load path
(`PeftModel.from_pretrained("…/final_adapter")`). The 55 MB copy is
negligible on `/data2`.

**Files:** `scripts/finetune_evo2_lora.py` (`import shutil` added,
final-adapter export block rewritten); `docs/gputee/FINETUNE_GUIDE.md`
§11 (noted the copytree mechanism).

### 26. LoRA script docstring rewritten (post-fix launch block)

**Change:** Rewrote the top-of-file docstring `Launch` block in
`scripts/finetune_evo2_lora.py` to reflect the new canonical gputee
setup: `export HF_HOME=/data2/ds85/hf_cache` before `deepspeed`,
`--output-dir /data2/ds85/bgcmodel_runs/phase1_lora` instead of the
old in-repo path, and added a "Notes" block calling out grad-accum,
output-dir, and the checkpoint-size fix (with cross-reference to
`docs/gputee/FINETUNE_GUIDE.md` §11). No code logic changed in this
entry — it is purely a docstring update.

**Why:** keeps `python scripts/finetune_evo2_lora.py --help` and the
raw module docstring consistent with the guide. The trojai-era
docstring implicitly assumed 4-GPU via `CUDA_VISIBLE_DEVICES=0,1,2,3`
and `--num_gpus=4`, which entry #2/C2 in this changelog had already
flagged as an EDIT.

**Files:** `scripts/finetune_evo2_lora.py` top docstring only.

---

## Summary of what was intentionally left unchanged

- **All code logic in `scripts/finetune_evo2*.py`** apart from the top
  docstring AND the two post-migration fixes in entries #24–#25 below
  (the `exclude_frozen_parameters=True` flag on DeepSpeed save, and
  `final_adapter/` via `shutil.copytree` of the already-saved adapter).
  Every other bug fix, the per-rank GPU masking, the DS config, the
  LoRA config, the default hyperparameters, the training loop, and the
  plotting code are byte-identical to the trojai version.
- **The DeepSpeed ZeRO-2 config** stays at stage 2 with full sharding knobs.
  At world_size=1 this is a no-op sharding-wise but still provides bf16,
  grad-accum, grad-clip, LR schedule, and the peft-compatible checkpoint
  path. Replacing it with raw PyTorch / `accelerate` would be a new
  training path needing its own smoke test; out of scope.
- **`environment.yml`, `environment.min.yml`, `requirements.txt`** are
  unchanged. The CUDA 12.4 torch wheel is forward-compatible with the
  CUDA 12.9 driver on gputee. A torch / flash-attn bump is an H100
  performance optimisation, not a migration fix.
- **`BGC_Research_Plan.md`** is unchanged on both sides. It is a research
  plan, not a hardware reference.
- **All data files.** Per the user's explicit instruction, no data was
  copied, moved, or deleted. The three "not migrated" items in
  `PROJECT_GUIDE.md` §4.1 are a status-only record.
- **The `docs/trojai/` tree.** Pristine snapshot of what was in the repo
  root before the migration.


