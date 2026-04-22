# Evo2 7B Fine-Tuning Guide

*Last updated: 2026-04-15 (LoRA smoke test passed; pipeline ready to launch)*

Everything needed to fine-tune Evo2 7B on the combined BGC training dataset.
Covers hardware constraints, hyperparameter rationale, what to log, what to watch for,
and how to resume from a checkpoint. Read this before starting a run.

---

## 1  Hardware and memory constraints

### Available GPUs

```
GPU 0: NVIDIA A40  48 GB VRAM  (46,068 MiB per nvidia-smi)
GPU 1: NVIDIA A40  48 GB VRAM
GPU 2: NVIDIA A40  48 GB VRAM
GPU 3: NVIDIA A40  48 GB VRAM
```

All 4 GPUs must be used. This is not optional — the memory math requires it.

### Why all 4 GPUs are required

Evo2 7B has 7 billion parameters. In bf16 mixed precision:

| Component                        | Size    |
| -------------------------------- | ------: |
| Model weights (bf16)             |  14 GB  |
| Gradients (bf16)                 |  14 GB  |
| AdamW optimizer states (fp32 m+v)|  56 GB  |
| **Total (no activations)**       | **84 GB** |

84 GB far exceeds any single 46 GB A40. The solution is **DeepSpeed ZeRO-2**, which
shards optimizer states and gradients across GPUs while keeping a full weight copy on
each (compatible with Evo2's non-standard StripedHyena loader):

| Config              | Per-GPU footprint | Verdict         |
| ------------------- | ----------------: | --------------- |
| 1 GPU, no sharding  |           84 GB   | ❌ OOM          |
| 2 GPU, ZeRO-2       |           49 GB   | ❌ OOM          |
| 2 GPU, ZeRO-3       |           42 GB   | ⚠️ tight + risky |
| **4 GPU, ZeRO-2**   |       **31.5 GB** | **✅ original estimate** |
| 4 GPU, ZeRO-3       |           21 GB   | ✅ but complex  |

> **Important update from smoke testing (2026-04-15):** The theoretical ZeRO-2 estimate
> of 31.5 GB/rank does not account for StripedHyena's activation memory.
> In practice, the forward pass alone uses **~37–41 GB per rank** at L=1024–4096 (batch=1),
> leaving no room for the AdamW state tensors that are lazily allocated on first
> `optimizer.step()`. See §12 (Smoke-Test Findings) for the full breakdown and recommended
> path forward (LoRA).

ZeRO-3 shards the weights themselves, which conflicts with how `evo2.Evo2` loads the
StripedHyena 2 model. ZeRO-2 is the right choice: it keeps full weights on each GPU
(matching the existing load path) and shards only optimizer state and gradients.

With gradient checkpointing enabled, activations for a 32,768-token sequence add
approximately 3–4 GB per GPU, bringing the total to ~35 GB — well within the 46 GB limit.

> **Correction (2026-04-15):** The "3–4 GB activation" estimate was too optimistic.
> StripedHyena's long-convolution filter computation (`compute_filter`) builds an
> `[poles × channels × L]` tensor during every forward pass. At L=1024 this already
> reaches ~14 GB. Activation checkpointing is required and must be applied at the block
> level; without it, a full-parameter fine-tune is not feasible on A40s.

### Why BioNeMo is not used

NVIDIA BioNeMo Framework 2 (`bionemo-evo2`) wraps Evo2 for fine-tuning using
NeMo/Megatron-LM. It is the most production-ready option but requires their Docker
container and is designed for multi-node H100 clusters. It is not installed and is not
practical on this 4× A40 setup. The training script in this project uses PyTorch +
DeepSpeed directly, which gives equivalent capability with more transparency.

BioNeMo's fine-tuning examples are a useful reference for hyperparameter defaults and
sequence packing implementation:
https://github.com/NVIDIA/bionemo-framework/tree/main/sub-packages/bionemo-evo2

---

## 2  Required installations (one-time)

All required packages are installed in the `bgcmodel` env (installed 2026-04-15):

```
deepspeed  0.18.9
wandb      0.26.0
peft       0.19.0     ← LoRA adapters
```

Verified: all 4× A40 (48 GB each) visible to PyTorch and DeepSpeed.

Before your first run, log in to WandB once:
```bash
conda activate bgcmodel
wandb login  # paste API key from wandb.ai/authorize
```

---

## 3  Training data

| Split | Records  | File                                            |
| ----- | -------: | ----------------------------------------------- |
| Train | 277,238  | `data/processed/splits_combined/train.jsonl`    |
| Val   |  34,655  | `data/processed/splits_combined/val.jsonl`      |
| Test  |  34,655  | `data/processed/splits_combined/test.jsonl`     |

All records use **Phase 1 conditioning format** — `COMPOUND` token stripped:
```
|COMPOUND_CLASS:{cls}|{tax_tag}{sequence}
```

MIBiG and antiSMASH DB records are identical in format. The `training_text` field
in each JSONL record is the exact string to feed to the model.

### Sequence length profile (train set)

| Length range | Records  |    % |
| ------------ | -------: | ---: |
| < 5 kb       |    1,180 |  0.4% |
| 5 – 20 kb    |   48,301 | 17.4% |
| 20 – 50 kb   |  183,000 | 66.0% |
| 50 – 100 kb  |   38,774 | 14.0% |
| 100 – 262 kb |    5,983 |  2.2% |

Median: **22,951 bp** · p90: 58,859 bp · p99: 123,724 bp

Training is capped at **32,768 bp** per sequence (see §4). This covers 83% of sequences
at full length; the remaining 17% are centre-cropped. The crop preserves the core
biosynthetic genes, which are in the middle of the region antiSMASH calls.

---

## 4  Hyperparameters

**Active script: `scripts/finetune_evo2_lora.py`** (LoRA, smoke-test passed ✅)
Reference script: `scripts/finetune_evo2.py` (full fine-tune, memory wall — see §12)

All defaults are baked into the script and saved to `config.json` at run start.
Override via CLI flags.

### LoRA-specific hyperparameters (`finetune_evo2_lora.py`)

| Parameter          | Value  | Rationale                                                          |
| ------------------ | ------ | ------------------------------------------------------------------ |
| `--lora-r`         | `16`   | Adapter rank. Gives 28.7M trainable params (0.44% of 6.5B). Increase to 32/64 if val loss plateaus early. |
| `--lora-alpha`     | `32`   | Scaling = alpha/r = 2.0. Standard starting value.                 |
| `--lora-dropout`   | `0.05` | Light regularisation on adapter paths.                             |
| `--lora-targets`   | (see below) | All Linear layers: Wqkv, out_proj, out_filter_dense, l1, l2, l3 |

**LoRA target modules** cover all 133 `nn.Linear` layers in Evo2 7B:
- Attention (5 blocks × `Wqkv` + `out_proj`): ~1.6M params
- Hyena filter output (27 blocks × `out_filter_dense`): ~3.7M params
- MLP (32 blocks × `l1`, `l2`, `l3`): ~23.2M params
- **Total trainable: 28.7M / 6.51B = 0.44%**

### Core training hyperparameters

| Parameter              | LoRA value | Full FT value | Rationale                                          |
| ---------------------- | ---------- | ------------- | -------------------------------------------------- |
| `--lr`                 | `5e-5`     | `1e-5`        | Higher LR for adapters is standard (adapters are randomly initialised) |
| `--lr-min-ratio`       | `0.1`      | `0.1`         | Cosine decay floor = 10% of peak                   |
| `--warmup-steps`       | `200`      | `200`         | ~1% of first epoch                                 |
| `--batch-size`         | `4`        | `4`           | Sequences per GPU per step                         |
| `--grad-accum`         | `8`        | `8`           | Effective batch = 128 sequences                    |
| `--max-seq-len`        | `32768`    | `32768`       | Covers 83% of training sequences at full length    |
| `--max-epochs`         | `2`        | `2`           | ~4,332 steps; early stop on val loss plateau       |
| `--weight-decay`       | `0.01`     | `0.1`         | Lower for LoRA — heavy WD hurts tiny adapter params |
| `--grad-clip`          | `1.0`      | `1.0`         | Max gradient norm                                  |
| `--beta1`              | `0.9`      | `0.9`         | Standard AdamW                                     |
| `--beta2`              | `0.95`     | `0.95`        | From Evo2 paper                                    |
| `--seed`               | `42`       | `42`          | Reproducibility                                    |

### Effective batch size and throughput

```
4 sequences/GPU × 4 GPUs × 8 grad-accum steps = 128 sequences/effective step
128 × ~32,000 tokens = ~4M tokens/step
```

### Memory at runtime (verified, LoRA)

| Sequence length | Peak GPU memory (rank 0) | Status |
|---|---:|---|
| L = 1,024  | **23.2 GB** | ✅ Confirmed in smoke test |
| L = 4,096  | ~43 GB      | ⚠️ OOM — StripedHyena filter |
| L = 32,768 | ~?? GB      | ❓ Needs activation checkpointing |

At L=1,024 with LoRA, each A40 uses 23.2 GB — 21 GB headroom. The StripedHyena
`compute_filter` tensor scales as O(L), so longer sequences hit OOM in the forward
pass regardless of LoRA. Start with `--max-seq-len 8192` and increase if memory allows.

### Steps and time estimate

```
277,238 records / (4 seq/GPU × 4 GPUs × 8 accum) = ~2,166 steps/epoch
2 epochs = ~4,332 steps total

LoRA estimated step time: ~15–30 s (vs 30–60 s full FT — less compute per step)
LoRA estimated total: 18–36 hours (< 2 days)
```

---

## 5  What to record — required logging

If a run crashes and you haven't checkpointed, you lose days of compute.
The training script enforces all of the following.

### At run start (once)

- `{output_dir}/config.json` — all hyperparameters, data file paths, git commit hash,
  hostname, timestamp, CUDA/driver version
- `{output_dir}/data_fingerprint.txt` — line count of each split + SHA256 of first
  100 lines (verifies data hasn't changed between runs)
- `{output_dir}/env.txt` — full `pip freeze` output (exact package versions)

### Every 10 steps (cheap, always on)

```json
{"step": 1240, "epoch": 0.57, "train_loss": 1.823, "lr": 9.8e-6,
 "grad_norm": 0.42, "gpu_mem_gb": [34.1, 33.9, 34.2, 34.0],
 "tokens_per_sec": 18400, "elapsed_sec": 37200}
```

Written to `{output_dir}/train_log.jsonl` and streamed to WandB.

### Every 250 steps (~2–3 hrs)

- Validation loss on 2,000 randomly sampled val sequences
- Validation perplexity (= exp(val_loss))
- Written to `{output_dir}/val_log.jsonl` and WandB

### Every 500 steps (~5–6 hrs)

**Checkpoint** saved to `{output_dir}/checkpoints/step_{N}/`:
```
step_{N}/
  model_weights.pt          # model state dict
  optimizer_state.pt        # DeepSpeed ZeRO-2 optimizer shards
  scheduler_state.pt        # LR scheduler state
  step.txt                  # step number for easy resume
```

**Generation sample** — 4 sequences generated (one per class: PKS/NRPS/TERPENE/RIPP)
with *E. coli* taxonomy tag, saved to `{output_dir}/samples/step_{N}.fasta`.
M1 (antiSMASH class prediction) and M2 (Pfam domain check) run automatically;
results written to `{output_dir}/samples/step_{N}_eval.json`.

### Checkpoint retention policy

Keep: **last 5 checkpoints** + **best validation loss checkpoint** (tagged `best/`).
Delete older checkpoints automatically to stay within disk budget (~28 GB/checkpoint).

---

## 6  Launch command

**Use `finetune_evo2_lora.py`.** The full-parameter script (`finetune_evo2.py`) OOMs
at all tested sequence lengths — see §12. The LoRA script has passed a full smoke test
(4 steps, L=1024, 4× A40, forward+backward+optimizer+checkpoint+WandB all confirmed).

> **⚠️ DO NOT launch production yet.** Per-block activation checkpointing must be
> implemented first. Without it, the maximum feasible sequence length is ~L=2048 (2 kb),
> which covers only ~9% of the median BGC (23 kb). The core biosynthetic domains
> (PKS modules, NRPS adenylation/condensation domains, etc.) are past the truncation
> point for most training sequences — fine-tuning at L=2048 would not teach the model
> the architecture it needs to reproduce.
>
> With per-block `torch.utils.checkpoint`, estimated peak GPU memory at L=32,768 drops
> from ~112 GB → ~18–22 GB (filter tensor only live for one block at a time).
> Implement, smoke-test at L=32768, then launch.

### Pre-flight checks

```bash
conda activate bgcmodel

# 1. All 4 GPUs must be idle
nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_uuid --format=csv
# Should return header only. Any resident process > 500 MB will cause OOM.

# 2. Disk space
df -h .
# Need at least 50 GB free (LoRA checkpoints are ~2 GB each, keep_last=5)

# 3. WandB auth
python -c "import netrc, os; n=netrc.netrc(os.path.expanduser('~/.netrc')); print('wandb auth OK:', bool(n.hosts.get('api.wandb.ai')))"
```

### Smoke test (run before every real training run)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
  scripts/finetune_evo2_lora.py \
  --train data/processed/splits_combined/val.jsonl \
  --val   data/processed/splits_combined/val.jsonl \
  --output-dir checkpoints/smoketest_lora \
  --max-seq-len 1024 --batch-size 1 --grad-accum 1 \
  --warmup-steps 2 --max-epochs 1 --max-steps 10 \
  --log-every 1 --val-every 5 --save-every 10 --val-max-batches 4 \
  --wandb-project bcg-evo2-phase1
```

Expected output (key lines to verify):
```
[rank0] World size: 4
[rank0] Fixed 40 tensors (40 cloned out of inference mode)
[rank0] LoRA applied: r=16  alpha=32  dropout=0.05
[rank0] Trainable params: 28,704,768 / 6,509,764,352  (0.441%)
[rank0] step     1 | ep 0.00 | loss X.XXXX | lr X.XXe-XX | ...
[rank0] VAL @ step 5: loss X.XXXX  ppl X.XXX
[rank0] Adapter saved → checkpoints/smoketest_lora/checkpoints/step_10/adapter
[rank0] Done. step=10  best_val_loss=X.XXXX
```

### Production launch

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
  scripts/finetune_evo2_lora.py \
  --train         data/processed/splits_combined/train.jsonl \
  --val           data/processed/splits_combined/val.jsonl \
  --output-dir    checkpoints/phase1_lora \
  --max-seq-len   8192 \
  --batch-size    4 \
  --grad-accum    8 \
  --lr            5e-5 \
  --lora-r        16 \
  --lora-alpha    32 \
  --warmup-steps  200 \
  --max-epochs    2 \
  --save-every    500 \
  --val-every     250 \
  --wandb-project bcg-evo2-phase1 \
  --seed          42
```

> **Note on `--max-seq-len`:** Start at 8192 rather than 32768. L=4096 already
> hits OOM in the StripedHyena filter (see §12.3). Activation checkpointing (not
> yet implemented) is needed before using the full 32K context. At L=8192 the
> filter tensor is 8× larger than at L=1024 — test with a smoke run first.

### Resuming from a checkpoint

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \
  scripts/finetune_evo2_lora.py \
  --train         data/processed/splits_combined/train.jsonl \
  --val           data/processed/splits_combined/val.jsonl \
  --output-dir    checkpoints/phase1_lora \
  --resume-from   checkpoints/phase1_lora/checkpoints/step_2000 \
  [... same flags as original run ...]
```

Resume reloads:
1. LoRA adapter weights from `checkpoints/step_N/adapter/` via `PeftModel.from_pretrained`
2. DeepSpeed optimizer + scheduler state from the ZeRO partition files
3. Step counter and best val loss from `client_state`

### Monitoring (from any machine)

```bash
# Live tail of training log
tail -f checkpoints/phase1_classonly/train_log.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"step {r['step']:5d} | loss {r['train_loss']:.4f} | lr {r['lr']:.2e} | {r['tokens_per_sec']:,} tok/s\")
"

# GPU utilisation
watch -n 5 nvidia-smi

# WandB dashboard: https://wandb.ai/{your-username}/bcg-evo2-phase1
```

---

## 7  Warning signs during training

| Symptom | Likely cause | Action |
| ------- | ------------ | ------ |
| Loss spikes then recovers in first 200 steps | Normal LR warmup | Ignore |
| Loss spike after warmup (step > 200) | LR too high | Stop, halve `--lr`, resume from last checkpoint |
| Loss plateaus at > 3.0 after 500 steps | Data pipeline error or LR too low | Check that `training_text` format is correct; verify tokeniser |
| Val loss rises while train loss falls | Overfitting | Stop early; use best-val checkpoint for generation |
| `nan` in `grad_norm` | Exploding gradients or corrupt batch | Gradient clipping should catch this; if persistent, check data for non-ACGTN characters |
| GPU memory increasing step-over-step | Memory leak in data loader | Restart from last checkpoint; check `del` and `torch.cuda.empty_cache()` in loader |
| One GPU at 0% utilisation | DDP init failure | Check NCCL; verify all 4 GPUs visible with `nvidia-smi` |
| `tokens_per_sec` drops 50%+ mid-run | Long sequence batch hit | Expected variance; batches containing 100K+ sequences are slower |

---

## 8  Expected training trajectory

Based on similar genomic LM fine-tuning runs (GenSLMs, HyenaDNA, NucleotideTransformer):

| Phase | Steps | Expected train loss | Expected val loss |
| ----- | ----: | ------------------: | ----------------: |
| Warmup | 0–200 | 3.5 → 2.5 (dropping fast) | — |
| Early | 200–1000 | 2.5 → 2.0 | ~2.2 |
| Mid | 1000–2500 | 2.0 → 1.7 | ~1.9 |
| Late (epoch 2) | 2500–4332 | 1.7 → 1.5 | ~1.6–1.7 |

Evo2 base model perplexity on random genomic sequence is ~2.0 (loss ~0.7 nats).
Fine-tuned loss will be higher because BGC sequences + conditioning prefixes are
a narrower and more structured distribution than the pretraining data.
A final val loss of ~1.5–1.8 is a reasonable target.

The M1 class match rate from generation samples is the more meaningful quality signal:
- Random baseline: ~25% (TERPENE+RIPP dominate at ~49%; guessing always-TERPENE gives 25%)
- Target for "model is learning": > 50% by step 1000
- Target for "model is working well": > 75% by end of training

---

## 9  Phased conditioning plan

This guide covers Phase 1. Future phases require small changes to the data pipeline
and training launch, not to the training script itself.

### Phase 1 — Class-only (current)

Format: `|COMPOUND_CLASS:{cls}|{tax_tag}{sequence}`
Data: all 346,559 records, uniform format
Goal: class-correct, chassis-appropriate BGC generation
Validation: M1 class match rate + M2 domain recovery

### Phase 2 — Compound conditioning for well-represented compounds

Format for MIBiG records with ≥ 3 examples of the same compound:
`|COMPOUND_CLASS:{cls}||COMPOUND:{tok}|{tax_tag}{sequence}`
All other records: class-only (same as Phase 1)

Data prep change: filter MIBiG JSONL by `compound_token` count threshold before merge.
~45 compounds qualify (≥ 3 examples: `carotenoid` ×17, `o-antigen` ×11, `ectoine` ×10, etc.)

### Phase 3 — SMILES conditioning (future)

Format: `|COMPOUND_CLASS:{cls}||SMILES:{canonical_smiles}|{tax_tag}{sequence}`
Requires: RDKit canonicalisation of 2,118 MIBiG records with SMILES (80.3% coverage)
SACCHARIDE (52.9%) and RIPP (42.7%) fall back to class-only.

---

## 10  Transposition vs invention — why this matters for interpreting results

This section is reproduced from PROJECT_GUIDE.md §7 for standalone reference.

The model is trained to do **transposition**: given a known compound class (and optionally
a known compound), generate a BGC with the correct biosynthetic architecture expressed
with chassis-appropriate sequence statistics. It is **not** trained to invent new
biosynthetic pathways.

**Transposition is tractable because:**
- What biosynthetic architecture is associated with a class/compound → learned from MIBiG
- What sequences look like in *E. coli* → learned from 6,239 *E. coli* antiSMASH records

**Invention is not tractable because:**
- 91% of MIBiG compound tokens have exactly one training example
- A model cannot learn the chemical meaning of a compound name from a single sequence
- It can at best memorise the one example and adapt it to a new chassis

**Consequence for generated sequences:** A generated sequence conditioned on
`|COMPOUND_CLASS:NRPS||D__BACTERIA;...;S__ESCHERICHIA|` should be annotated as NRPS
by antiSMASH (M1), contain C/A/T domains (M2), and have *E. coli*-appropriate codon
usage (M7). It will not reliably produce any specific known natural product — it will
produce a novel NRPS cluster architecture that has never existed in nature.
That is the intended output.

---

## 12  Smoke-test findings (2026-04-15)

A full integration smoke test was run (4 steps, L=1024–4096, batch=1) to validate the
training pipeline before committing to a multi-day run. Every bug discovered and its fix
is recorded here.

### 12.1  Evo2 ↔ DeepSpeed integration bugs (all fixed in `finetune_evo2.py`)

Evo2's `StripedHyena` loader is non-standard and required three patches to cooperate
with DeepSpeed ZeRO-2. These are permanent fixes in the script.

**Bug 1 — Evo2 auto-shards layers across all visible GPUs**

Evo2's vortex loader spreads model layers across every visible CUDA device
(e.g. `Assigned layer_idx=30 to device='cuda:3'`). This is a pipeline-parallel
layout and directly conflicts with ZeRO-2, which expects a full replica on each rank.

*Fix:* At the very start of `main()`, before any CUDA call:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank_from_env)
os.environ["LOCAL_RANK"] = "0"
args.local_rank = 0
```
Each worker process is restricted to its own single GPU; Evo2 then sees exactly one
device and places all layers there.

**Bug 2 — Non-contiguous Wqkv tensors after model load**

Evo2 applies `.permute()` post-load to Wqkv projection weights
(`Adjusting Wqkv for column split (permuting rows)`). The permuted tensors are
non-contiguous. DeepSpeed's `_broadcast_model` fails with
`ValueError: Tensors must be contiguous`.

*Fix:* After `evo2.Evo2(...)`, walk all parameters and buffers:
```python
for p in model.parameters():
    if not p.is_contiguous():
        p.data = p.data.contiguous()
```
5 tensors are fixed on each rank.

**Bug 3 — DeepSpeed `WarmupCosineLR` API mismatch**

The original DS config used `warmup_min_lr` and `warmup_max_lr`. In DS 0.18.x,
`WarmupCosineLR.__init__` has a different signature:
```
total_num_steps, warmup_num_steps, warmup_min_ratio, cos_min_ratio, warmup_type
```
Peak LR is taken from the optimizer, not from the scheduler config.

*Fix:* Updated `build_ds_config()` to use the correct parameter names.

### 12.2  Smoke-test progression

| Attempt | Furthest point reached | Blocker |
|---|---|---|
| v1 | `deepspeed.initialize → _broadcast_model` | Non-contiguous Wqkv + Evo2 multi-GPU auto-shard |
| v2 | `init_distributed` | NCCL `invalid device ordinal` (LOCAL_RANK=3, no cuda:3 after masking) |
| v3 | `init_distributed` | Same; needed both env and `args.local_rank` rewritten |
| v4 | `DeepSpeedEngine._do_args_sanity_check` | `args.local_rank` vs `env['LOCAL_RANK']` mismatch |
| v5 | `_configure_lr_scheduler` | `WarmupCosineLR` API changed in DS 0.18 |
| **v6** | **Forward pass runs (L=4096)** | OOM: ~41.5 GiB used before filter allocation |
| **v7** | **Forward + backward complete (L=1024)** | OOM at `optimizer.step()` allocating `exp_avg_sq` |

v7 confirmed: forward pass, loss computation, and backward pass all work correctly.
The pipeline is sound. The only remaining issue is memory budget for `optimizer.step()`.

### 12.3  Memory analysis — why OOM occurs

At L=1024, batch=1, ZeRO-2 across 4 ranks (3 free GPUs in test):

| Component | Memory (actual, per rank) | Notes |
|---|---:|---|
| bf16 weights (not sharded by ZeRO-2) | ~13 GB | Replicated on every rank |
| fp32 weight master copy shard | ~8.6 GB | ZeRO-2 divides by world_size |
| Gradient shard | ~4.3 GB | ZeRO-2 divides by world_size |
| StripedHyena activations at L=1024 | ~10–14 GB | Long-conv filter per block is large |
| **Total before optimizer.step()** | **~36–40 GB** | Leaves < 8 GB free |
| AdamW `exp_avg_sq` shard (lazy init) | ~8.6 GB | Fails on first step |

The root cause is StripedHyena's `compute_filter()` which materialises a full
`[poles × channels × L]` tensor during every forward pass. This is not a ZeRO bug —
it's inherent to the architecture. Activation checkpointing at the block level is needed
to reclaim this memory between blocks.

### 12.4  Recommended path forward: LoRA

Full-parameter fine-tuning of Evo2 7B on 4× A40 (48 GB) requires one or more of:

| Option | Memory saving | Implementation risk |
|---|---|---|
| **LoRA on attention + MLP projections** | **~100× optimizer state** | **Low — well-tested with peft** |
| Activation checkpointing (block-level) | ~4× activation memory | Medium — needs vortex block wrapping |
| ZeRO-3 | Shards weights (~3.2 GB/rank) | High — Evo2 loader resists weight sharding |
| 8-bit AdamW (bitsandbytes) | 4× optimizer state | Medium — requires bitsandbytes install |

**Recommendation: LoRA.** Freeze base weights, add low-rank adapters (rank 16–64) to
the attention `Wq`, `Wk`, `Wv`, `Wo` and optionally MLP `fc1`/`fc2` projections.
Optimizer state drops ~100×, fitting easily on a single rank. The rest of the training
infrastructure (ZeRO-2, WandB, checkpointing, dataset indexing) is unchanged.

For the transposition goal (Phase 1), LoRA is a legitimate scientific choice:
the base model already contains rich sequence representations; adapters steer
generation without altering the underlying genomic language model.

### 12.5  Pre-run checklist

Before starting any real training run:

- [ ] **GPU 0 must be free** — check `nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_uuid --format=csv`. A separate training job (or any resident process > 1 GB) on any of the 4 GPUs will cause OOM.
- [ ] Confirm `wandb login` is active (`wandb status` should show a valid API key in netrc)
- [ ] Run a 10-step smoke test first: `--max-steps 10 --max-seq-len 1024 --batch-size 1 --grad-accum 1`
- [ ] Ensure at least 200 GB disk free for checkpoints (`df -h .`)

### 12.6  LoRA smoke test — bugs fixed and confirmed results

After the full-parameter smoke test confirmed the DeepSpeed integration bugs were fixed
(§12.1), a separate LoRA script (`finetune_evo2_lora.py`) was written and smoke-tested.
Five additional bugs were encountered and fixed before the test passed end-to-end.

**LoRA Bug 1 — `TypeError: 'NoneType' object is not callable` in `get_peft_model`**

peft calls `model.config.to_dict()` when building the adapter. Evo2's config is a
`vortex.model.utils.dotdict`, which returns `None` for missing keys rather than raising
`AttributeError`. The key `to_dict` does not exist, so `model.config.to_dict` is `None`,
and `None()` raises `TypeError`. A standard `hasattr()` check returns `True` (the attribute
"exists" with value `None`), so the peft guard does not catch it.

*Fix:* Inject a working `to_dict` via dict-key assignment (bypasses dotdict attribute
routing):
```python
model.config["to_dict"] = lambda: {
    k: v for k, v in model.config.items() if not callable(v)
}
```

**LoRA Bug 2 — `AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'`**

peft 0.19's `_cast_adapter_dtype` tries to detect float8 support by accessing
`torch.float8_e8m0fnu`. This dtype does not exist in torch 2.5.1.

*Fix:* Pass `autocast_adapter_dtype=False` to `get_peft_model()`:
```python
peft_model = get_peft_model(model, config, autocast_adapter_dtype=False)
```

**LoRA Bug 3 — `RuntimeError: Inference tensors cannot be saved for backward`**

Evo2 sets `inference_mode: True` in its vortex config. Several tensors (notably the
`RMSNorm.scale` buffers in every block) are allocated as **inference-mode tensors** during
the model load. PyTorch's autograd cannot save inference-mode tensors in the backward
graph — they are read-only and outside the autograd engine.

This only appeared with LoRA (not the full-FT script) because full-FT OOMs before
reaching the backward pass, whereas LoRA has enough memory.

*Fix:* After model load, clone every inference-mode parameter and buffer, then call
`model.train()`:
```python
with torch.no_grad():
    for p in model.parameters():
        if hasattr(p.data, "is_inference") and p.data.is_inference():
            p.data = p.data.clone().contiguous()
    for mod_name, mod in model.named_modules():
        for buf_name, buf in list(mod.named_buffers(recurse=False)):
            if buf is not None and hasattr(buf, "is_inference") and buf.is_inference():
                setattr(mod, buf_name, buf.clone().contiguous())
model.train()
```
40 tensors are fixed on each rank (all 40 are inference-mode; they also need
`.contiguous()` for the DeepSpeed broadcast).

**LoRA smoke test — confirmed results (2026-04-15)**

Run: 10 steps, L=1024, batch=1, grad_accum=1, 4× A40, world_size=4.

```
[rank0] Fixed 40 tensors (40 cloned out of inference mode)
[rank0] LoRA applied: r=16  alpha=32  dropout=0.05
[rank0] Trainable params: 28,704,768 / 6,509,764,352  (0.441%)
[rank0] step  1 | ep 0.00 | loss 1.9344 | lr 2.50e-05 | gnorm 0.045 | mem 23.2 GB
[rank0] step  2 | ep 0.00 | loss 1.9303 | ...
...
[rank0] VAL @ step 5:  loss 1.9279  ppl 6.871
[rank0] step  6 | ep 0.00 | loss 1.9250 | ...
...
[rank0] VAL @ step 10: loss 1.8954  ppl 6.655
[rank0] Adapter saved → checkpoints/smoketest_lora/checkpoints/step_10/adapter
[rank0] Done. step=10  best_val_loss=1.8954
```

All systems confirmed operational:
- ✅ Model loads + LoRA applied (0.44% trainable)
- ✅ Forward pass completes at L=1024 (23.2 GB peak)
- ✅ Backward pass + `optimizer.step()` complete (AdamW state ~336 MB total)
- ✅ Val loss falls over 10 steps (1.9344 → 1.8954 = −2.0%)
- ✅ Checkpoint saved (adapter_config.json + adapter_model.safetensors)
- ✅ WandB metrics logged
- ✅ Resume path tested

**LoRA smoke test progression:**

| Attempt | Furthest point | Blocker |
|---|---|---|
| v1 | `get_peft_model()` | `dotdict.to_dict` returns None (Bug 1) |
| v2 | `get_peft_model()` | `torch.float8_e8m0fnu` missing (Bug 2) |
| v3 | Backward pass | Inference-mode tensors (Bug 3) |
| v4 | Forward pass (L=4096) | OOM: StripedHyena filter |
| **v5** | **Full 10 steps (L=1024)** | ✅ All confirmed |

**Post-smoke-test review (2026-04-15) — step-counter off-by-one fix**

During a code audit after the smoke test passed, a latent bug was found in both
`finetune_evo2.py` and `finetune_evo2_lora.py`:

```python
# BEFORE (buggy at grad_accum > 1):
model_engine.step()
if model_engine.is_gradient_accumulation_boundary():
    step += 1
    # log, val, checkpoint...
```

DeepSpeed's `is_gradient_accumulation_boundary()` uses the formula
`(micro_steps + 1) % ga == 0` and is designed to be called **before** `engine.step()`
(to predict whether the upcoming step will apply the optimizer). Called **after**
step(), it fires one micro-batch early — at micro-step 7, not micro-step 8 (for ga=8).

At `grad_accum=1` (smoke test config), `(micro_steps+1) % 1 == 0` is always True, so
the bug is invisible — the check coincidentally fires every micro-step, matching the
opt step rate.

At `grad_accum=8` (production config), logged `grad_norm` and `lr` would reflect the
*previous* optimizer step's state, and checkpoints labeled "step N" would actually
capture weights after N−1 optimizer steps.

**Fix (applied to both scripts):** use `model_engine.global_steps`, which DeepSpeed
increments exactly once per real optimizer step inside `_take_model_step`:

```python
# AFTER (correct at any grad_accum):
model_engine.step()
if model_engine.global_steps != step:
    step = model_engine.global_steps
    # log, val, checkpoint...
```

**Verified (2026-04-15):** Re-run with `--grad-accum 2` confirmed the fix. Key output:
```
[rank0] Steps per epoch:    4,332  (grad_accum=2)   ← 8,664 batches / 2 = correct
[rank0] VAL @ step 5:  loss 1.8569  ppl 6.404
[rank0] step   10 | loss 1.8125 | lr 5.00e-05 | gn 2.834   ← real grad_norm (not stale)
[rank0] VAL @ step 10: loss 1.8042  ppl 6.075
[rank0] Reached --max-steps=10; exiting
[rank0] Done. step=10  best_val_loss=1.8042
```
Step 10 was reached after exactly 20 micro-batches (2 per optimizer step). ✅
The NCCL "process group not destroyed" warning on exit is harmless — PyTorch 2.4+
raises it whenever cleanup happens at interpreter shutdown rather than via
`destroy_process_group`. No action needed.

---

## 11  Files written by the training script

LoRA checkpoints use a different structure from full fine-tune:
the adapter weights are saved via `peft`'s `save_pretrained` (two small files),
while the DeepSpeed optimizer/scheduler state is saved alongside.

```
checkpoints/phase1_lora/
├── config.json                    # all hyperparameters + git hash + timestamp
├── data_fingerprint.txt           # line counts + SHA256 of split files
├── env.txt                        # pip freeze output
├── train_log.jsonl                # per-step metrics (step, loss, lr, grad_norm, ...)
├── val_log.jsonl                  # per-validation-run metrics
├── checkpoints/
│   ├── step_500/
│   │   ├── adapter/
│   │   │   ├── adapter_config.json      # LoRA config (r, alpha, target_modules, ...)
│   │   │   └── adapter_model.safetensors # adapter weights only (~110 MB for r=16)
│   │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt   # ZeRO-2 optimizer shards
│   │   ├── zero_pp_rank_1_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_2_mp_rank_00_optim_states.pt
│   │   ├── zero_pp_rank_3_mp_rank_00_optim_states.pt
│   │   └── mp_rank_00_model_states.pt                  # scheduler + client_state (step, best_val_loss)
│   ├── step_1000/  ...
│   ├── step_1500/  ...  (older ones deleted per retention policy)
│   └── best/                      # copy of best-val-loss checkpoint dir
└── samples/
    ├── step_500.fasta             # 4 generated sequences
    ├── step_500_eval.json         # M1 + M2 results on those sequences
    ├── step_1000.fasta
    └── step_1000_eval.json
```

At the end of training, the final merged adapter is exported to:
```
checkpoints/phase1_lora/final_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

To use the adapter for inference:
```python
from peft import PeftModel
import evo2

base_model = evo2.Evo2("evo2_7b_262k")
model = PeftModel.from_pretrained(base_model, "checkpoints/phase1_lora/final_adapter")
model.eval()
```
