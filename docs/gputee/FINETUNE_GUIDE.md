# Evo2 7B Fine-Tuning Guide (gputee)

*Last updated: 2026-04-22 (trojai → gputee migration pass; numbers marked
"A40" are historical; numbers marked "H100" are pending re-measurement).*

Everything needed to fine-tune Evo2 7B on the combined BGC training dataset.
Covers hardware constraints, hyperparameter rationale, what to log, what to
watch for, and how to resume from a checkpoint. Read this before starting a run.

**Hardware context:** this copy of the guide is for the `gputee` host
(1× NVIDIA H100 PCIe, 80 GB). The archived `docs/trojai/FINETUNE_GUIDE.md`
documents the original 4× NVIDIA A40 analysis. Every number below that
is marked **(trojai measurement)** comes from the A40 smoke tests; every
number marked **(gputee, pending)** still needs to be re-measured on the
H100. Do not take the trojai numbers as load-bearing for any production
decision.

---

## 1  Hardware and memory constraints

### Available GPU

```
GPU 0: NVIDIA H100 PCIe  80 GB VRAM  (81,559 MiB per nvidia-smi)
Driver 575.64.03 / CUDA 12.9 runtime
```

There is only one GPU. All training runs use `deepspeed --num_gpus=1`.

### Why full-parameter fine-tuning still doesn't fit

Evo2 7B has 6.51 billion trainable params. In bf16 mixed precision the
baseline memory need for a full fine-tune is:

| Component                         |    Size |
| --------------------------------- | ------: |
| Model weights (bf16)              |   14 GB |
| Gradients (bf16)                  |   14 GB |
| AdamW optimizer states (fp32 m+v) |   56 GB |
| **Total (no activations)**        | **84 GB** |

84 GB > 80 GB, so full fine-tuning does not fit on the gputee H100
even before we count StripedHyena's activation tensors. On trojai the
same 84 GB was reached by sharding optimizer + grads across 4 ranks via
DeepSpeed ZeRO-2 (≈ 31.5 GB/rank budget, under the 46 GB A40 limit) —
but smoke testing showed activations ate the remaining 14 GB and OOM'd
at `optimizer.step()`. That analysis is preserved verbatim in
`docs/trojai/FINETUNE_GUIDE.md` §1; see §12 below for the recorded
smoke-test evidence.

On gputee the options for making the full 84 GB fit on a single 80 GB
GPU would be:

| Option                             | Memory saving                         | Implementation risk |
| ---------------------------------- | ------------------------------------- | ------------------- |
| **LoRA (current choice)**          | ~167× drop in optimizer state         | Low — already working; only needs re-smoke on H100 |
| ZeRO-3 with CPU/NVMe offload       | Weights offloaded to host RAM / disk  | High — Evo2's vortex loader resists ZeRO-3 sharding; also slow |
| 8-bit AdamW (bitsandbytes)         | 4× optimizer state                    | Medium — new dependency; unknown interaction with StripedHyena |
| FP8 full fine-tune (Transformer Engine) | ~2× weights + optimizer overall  | High — new dependency; Evo2 not FP8-native |

**LoRA is still the correct choice on gputee.** Freezing the base weights
drops the optimizer-state term from ~56 GB to ~336 MB (≈ 28.7 M trainable
params × 12 B per AdamW slot), leaving the ~14 GB of frozen bf16 weights
and forward/backward activations as the only significant memory consumers.

### Activations: what the StripedHyena filter does to memory

On trojai the dominant activation cost was the Hyena long-convolution
filter (`compute_filter`), which materialises a
`[poles × channels × L]` tensor in every forward pass. The trojai
measurements (with ZeRO-2 across 4 ranks) were:

| Sequence length | Peak per rank (A40 ZeRO-2, trojai) |
| --------------- | ---------------------------------: |
| L = 1,024       | 23.2 GB                            |
| L = 4,096       | ~43 GB (tight)                     |
| L = 8,192+      | OOM                                |

On gputee (world_size = 1, no ZeRO sharding) the same forward-pass
memory lands on a single GPU, but the 14 GB of bf16 weights that were
replicated on every trojai rank are still just 14 GB here (not 4× 14 GB).
The net effect on peak memory is not a simple scale — it needs to be
re-measured. Specifically: **do not** assume the A40 L=4096 OOM transfers
to H100; an 80 GB H100 with LoRA (no replicated fp32 master, no ZeRO
overhead) should have materially more headroom at the same L than a
ZeRO-2 A40 rank did. This is the first thing the H100 smoke benchmark
(§12.7 below) needs to pin down.

### Why BioNeMo is still not used

NVIDIA BioNeMo Framework 2 (`bionemo-evo2`) wraps Evo2 for fine-tuning
via NeMo/Megatron-LM. It is the most production-ready option but requires
a Docker container and is designed for multi-node H100 clusters. It is
not installed on gputee, and the single-GPU use case is a bad fit.
PyTorch + DeepSpeed + peft, with the per-script fixes already in place,
gives equivalent capability with more transparency.

BioNeMo's fine-tuning examples remain useful as a reference for hyperparameter
defaults and sequence packing implementation:
https://github.com/NVIDIA/bionemo-framework/tree/main/sub-packages/bionemo-evo2

---

## 2  Required installations (one-time)

The `bgcmodel` env on gputee was (re-)created from scratch via micromamba
(see `PROJECT_GUIDE.md` §3.1); it was not carried over from trojai. The
pinned training-stack additions that are **not in `environment.yml`** and
must be installed explicitly after the env create:

```
deepspeed  0.18.9
wandb      0.26.0
peft       0.19.0     ← LoRA adapters
```

### Fresh-install procedure (the sequence that actually works)

`environment.yml` lists both `torch==2.5.1+cu124` and
`flash-attn==2.7.4.post1` in its pip section. A naive
`micromamba env create -f environment.yml` fails because pip runs its
dependency resolution once across the whole pip list, and flash-attn's
`setup.py` does `import torch` at build time — but torch isn't installed
in the target env yet when that happens. The conda-side of the env does
finish cleanly; only the pip phase crashes.

The working sequence (tested on gputee 2026-04-22):

```bash
# 1. conda-side env (this succeeds; flash-attn failure is non-fatal here)
micromamba create -n bgcmodel -f environment.yml   # pip step will fail on flash-attn — ignore
micromamba activate bgcmodel

# 2. torch first. The index-url is required; it points at the cu124 wheel.
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. flash-attn. The prebuilt wheel from GitHub matches this stack exactly
#    (cu12 + torch2.5 + cp312 + cxx11abiFALSE). Avoids a 10-min source build
#    AND avoids the known setup.py EXDEV bug that triggers when /tmp and
#    the pip cache live on different filesystems (which is the case on
#    gputee: /tmp is on / and the pip cache is on /home).
cd /tmp
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install ./flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# 4. resume environment.yml's pip list (torch + flash-attn are now satisfied
#    and will be skipped; the remaining packages install cleanly)
cd ~/projects/BCGModelling
micromamba env update -n bgcmodel -f environment.yml

# 5. the three training-only deps that are NOT in environment.yml
pip install deepspeed==0.18.9 peft==0.19.0 wandb==0.26.0
```

Version verification after the env is built:

```bash
micromamba activate bgcmodel
python - <<'PY'
import importlib.metadata as md
import torch, transformers, peft, deepspeed, evo2, flash_attn

def ver(pkg):
    try: return md.version(pkg)
    except md.PackageNotFoundError: return "?"

print("torch         ", torch.__version__, "cuda", torch.version.cuda)
print("transformers  ", transformers.__version__)
print("peft          ", peft.__version__)
print("deepspeed     ", deepspeed.__version__)
print("evo2          ", ver("evo2"))
print("flash_attn    ", flash_attn.__version__)
print("cuda_available", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count(),
      "device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

from evo2 import Evo2
print("Evo2 class   ", Evo2)
PY
```

Expected on gputee: `cuda_available=True`, `device_count=1`,
`device_name=NVIDIA H100 PCIe`. Note that the `evo2` package does not
export a `__version__` attribute, so the metadata lookup above is the
correct way to read its version.

### Storage layout on gputee

`/home` on gputee is tight (1.8 TB, ~30–40 GB free as of 2026-04-22).
Two large shared filesystems provide the actual working space:

| Path | Size | Free (2026-04-22) | Use |
|---|---:|---:|---|
| `/home` | 1.8 TB | ~30 GB | code (`~/projects/BCGModelling`), env (`~/.local/share/mamba/envs/bgcmodel`), shell state |
| `/data2` | 7 TB | ~1.5 TB | HuggingFace cache, per-run output dirs, training logs |
| `/data` | 7 TB | ~420 GB | alternative if `/data2` fills |

Required environment variable (add to `~/.bashrc`):

```bash
# HuggingFace cache on /data2 — Evo2 7B is ~14 GB, /home cannot hold it
export HF_HOME=/data2/ds85/hf_cache
```

Create the cache directory once:
```bash
mkdir -p /data2/ds85/hf_cache
```

Per-run output directories should live on `/data2` too. The documented
pattern is:
```bash
--output-dir /data2/ds85/bgcmodel_runs/<run_name>
```

Even after the LoRA checkpoint fix (§11), long runs produce adapter
checkpoints, plots, offline wandb logs, and sample FASTA files that add
up to hundreds of MB per run — comfortable on `/data2`, unpleasant on
`/home`.

### WandB login

Before your first online-mode run:
```bash
micromamba activate bgcmodel
wandb login  # paste API key from wandb.ai/authorize
```
For smoke benchmarks, use `--wandb-mode offline` to skip this entirely.

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
at full length; the remaining ~17% (≈44,757 records — 14.0% in the 50–100 kb band and
2.2% in the 100–262 kb band) are centre-cropped. The crop preserves the core
biosynthetic genes, which are in the middle of the region antiSMASH calls.

**Known limitations of the centre-crop-only policy.** The 17% tail is substantial and
the current plan addresses it with a single strategy (centre-crop). A few caveats are
worth stating explicitly so the decision is recorded rather than implicit:

1. **The "core genes are in the middle" assumption is stated but not tested.**
   antiSMASH defines the region *around* the core biosynthetic genes, so centre-cropping
   preserves those genes by construction. Flanking tailoring, regulatory, transport, and
   resistance genes — which can matter for a synthesis-ready cluster — are what gets
   dropped. For records in the 100–262 kb band (2.2%, ~5,983 records), a 32 k centre
   crop discards ~70–90% of the record.
2. **Val loss is computed on the same 32 k crop.** Train/val/test are split on
   full-length records, but both training and validation see the crop. Val loss will
   look fine even if the model never observes a flank longer than 16 k either side of
   centre. No long-context held-out evaluation is currently defined.
3. **Train / inference length mismatch.** Evo2 7B supports 262 k at inference, so the
   fine-tuned model *can* be asked to generate full-length 50–150 kb BGCs. It would do
   so having only ever been fine-tuned on ≤32 k windows. The docs don't currently
   discuss this mismatch or how it interacts with the evaluation suite.
4. **No plan currently pushes L past 32 k at train time.** The §12.7 smoke benchmark
   decision rule terminates at L = 32 768; it doesn't have a branch for "if the H100
   shows headroom, try L = 65 536 to shrink the long tail." Block-level activation
   checkpointing (§12.7 and §13 of `PROJECT_GUIDE.md`) is the memory lever that would
   enable this, but the cost/benefit has not been worked through.

Candidate strategies to address (1)–(4) — random-window-per-epoch, multi-chunk per
record, curriculum L, and/or pushing L > 32 768 under block-level activation
checkpointing — are listed as open options in `PROJECT_GUIDE.md` §13 "Future
enhancements" and are **not** adopted decisions.

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

The script defaults (`batch_size=4`, `grad_accum=8`) were chosen for
`world_size=4` on trojai:

```
trojai (4× A40):   4 seq/GPU × 4 GPUs × 8 grad-accum = 128 sequences/effective step
```

On gputee `world_size=1`, so the defaults give a much smaller effective
batch:

```
gputee (1× H100), defaults as-is:    4 × 1 × 8  = 32 sequences/effective step
gputee, with --grad-accum 32:        4 × 1 × 32 = 128 sequences/effective step  ← recommended
```

**Recommendation:** pass `--grad-accum 32` on gputee to preserve the original
128-sequence effective batch. The LoRA defaults (lr, weight decay, betas,
warmup) were tuned against 128, and shrinking the effective batch to 32
without retuning would change the optimisation dynamics. The `DEFAULTS`
dict in the scripts themselves is **not** being changed; use the CLI flag.

### Memory at runtime

Two sub-tables: the first is historical (4× A40, trojai); the second is
empty until the gputee H100 smoke benchmark runs.

**Historical — trojai (4× A40 ZeRO-2, LoRA, batch=1):**

| Sequence length | Peak GPU memory (rank 0) | Status |
|---|---:|---|
| L = 1,024  | **23.2 GB** | ✅ Confirmed in trojai smoke test |
| L = 4,096  | ~43 GB      | ⚠️ OOM — StripedHyena filter |
| L = 32,768 | unknown     | ❓ Would require activation checkpointing |

**Pending — gputee (1× H100 80 GB, LoRA, batch=1):**

| Sequence length | Peak GPU memory | Status |
|---|---:|---|
| L = 1,024  | TBD | ❓ Pending smoke benchmark |
| L = 4,096  | TBD | ❓ Pending smoke benchmark |
| L = 8,192  | TBD | ❓ Pending smoke benchmark |
| L = 32,768 | TBD | ❓ Pending smoke benchmark |

Two notes on what to expect from the gputee benchmark:
1. The 14 GB of replicated bf16 weights that sat on each A40 rank now live
   on a single GPU once instead of four times. That is *lower* total
   memory, not higher. Peak at L=1024 on gputee should be materially
   below 23.2 GB.
2. The StripedHyena `compute_filter` activation tensor still scales O(L)
   in the forward pass and is per-block. Whether L=32 768 fits on one
   80 GB GPU without block-level checkpointing is the open question — the
   benchmark should answer it.

### Steps and time estimate

With `--grad-accum 32` (recommended gputee override):

```
277,238 records / (4 seq × 1 GPU × 32 accum) = ~2,166 steps/epoch   ← same as trojai
2 epochs = ~4,332 steps total
```

The step count is unchanged because the effective batch is preserved.
Wall-clock time, on the other hand, depends on the H100's tokens/sec at
the chosen L and is **pending re-measurement**. A reasonable a priori
estimate:

- An H100 is ~2× an A40 in bf16 matmul throughput per GPU.
- trojai had 4× A40 working in parallel ⇒ ~4× aggregate throughput.
- gputee has 1× H100 ⇒ ~2× aggregate throughput vs one A40, but ~2×
  *slower* than four A40s at the same per-step work.
- Expect the trojai "18–36 hours" estimate to roughly double on gputee,
  i.e. ~1.5–3 days at target L. Treat this as speculative until the
  smoke benchmark reports `tokens_per_sec`.

> **If block-level activation checkpointing is enabled** (see §12.7): every
> checkpointed block costs one extra forward pass during backward. Wrapping all
> 32 StripedHyena blocks turns a `1F + 1B` step into roughly `2F + 1B`, so
> expect step wall-clock time to be ~1.33× longer than the unchecked baseline.
> The step count is unchanged. Fold this factor into the time estimate above if
> the §12.7 decision rule selects the checkpointed path.

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

**Use `finetune_evo2_lora.py`.** The full-parameter script (`finetune_evo2.py`)
does not fit on gputee's 80 GB H100 (84 GB base budget) and OOM'd on trojai
(see §12). The LoRA script has passed a full smoke test on trojai
(4 steps, L=1024, 4× A40, forward+backward+optimizer+checkpoint+WandB all
confirmed). **No gputee smoke test has been run yet** — §12.7 below is the
first production task after the environment is rebuilt.

> **⚠️ Do not launch a long training run on gputee until the §12.7 smoke
> benchmark has recorded peak GPU memory at L = 1024, 4096, 8192, 32768.**
> The old "L=4096 OOMs" finding was measured with ZeRO-2 across 4× A40
> and does not transfer. On 1× H100 80 GB with LoRA we expect materially
> more headroom, but the question "does L=32 768 fit without block-level
> activation checkpointing?" is open until measured. If it does not fit,
> implement per-block `torch.utils.checkpoint` around the 32 StripedHyena
> blocks (see §13 NEXT in `PROJECT_GUIDE.md`) before launching.

### Pre-flight checks

```bash
micromamba activate bgcmodel    # or: conda activate bgcmodel

# 1. GPU must be idle
nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_uuid --format=csv
# Should return header only. Any resident process > 500 MB on the H100 will
# eat into the training budget. The gputee host is shared; coordinate with
# other users before launching.

# 2. Disk space
df -h /home
# Need at least 50 GB free (LoRA checkpoints are ~2 GB each, keep_last=5,
# plus ~17 GB for HF cache if Evo2 and ESMFold weights are not already cached).
# gputee's /home is tight (96% used as of 2026-04-22) — verify, don't assume.

# 3. HF cache
ls -lh ~/.cache/huggingface/hub 2>/dev/null | head
# If the Evo2 7B weights (~14 GB) are not there, the first model load
# will download them. Budget accordingly.

# 4. WandB auth
python -c "import netrc, os; n=netrc.netrc(os.path.expanduser('~/.netrc')); print('wandb auth OK:', bool(n.hosts.get('api.wandb.ai')))"
```

### Smoke test (run before every real training run)

```bash
export HF_HOME=/data2/ds85/hf_cache
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=1 \
  scripts/finetune_evo2_lora.py \
  --train data/processed/splits_combined/val.jsonl \
  --val   data/processed/splits_combined/val.jsonl \
  --output-dir /data2/ds85/bgcmodel_runs/smoketest_lora \
  --max-seq-len 1024 --batch-size 1 --grad-accum 1 \
  --warmup-steps 2 --max-epochs 1 --max-steps 10 \
  --log-every 1 --val-every 5 --save-every 10 --val-max-batches 4 \
  --wandb-project bcg-evo2-phase1
```

Expected output on gputee (key lines to verify):
```
[rank0] World size: 1
[rank0] Fixed 10 tensors  (X cloned out of inference mode)         ← count differs from trojai (see §12.6)
[rank0] LoRA applied: r=16  alpha=32  dropout=0.05
[rank0] Trainable params: 28,704,768 / 6,509,764,352  (0.441%)
[rank0] step     1 | ep 0.00 | loss X.XXXX | lr X.XXe-XX | ...
[rank0] VAL @ step 5: loss X.XXXX  ppl X.XXX
[rank0] Adapter saved → /data2/ds85/bgcmodel_runs/smoketest_lora/checkpoints/step_10/adapter
[rank0] Done. step=10  best_val_loss=X.XXXX
```

The trojai smoke test printed "Fixed 40 tensors" because it fixes tensors
on every rank; at `world_size=1` only rank-0's tensors are fixed, so the
count will be 1/4 of the trojai number. The `LoRA applied` and
`Trainable params` lines should match exactly — they depend only on the
model architecture and LoRA config, not on GPU count.

### Production launch

```bash
export HF_HOME=/data2/ds85/hf_cache
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=1 \
  scripts/finetune_evo2_lora.py \
  --train         data/processed/splits_combined/train.jsonl \
  --val           data/processed/splits_combined/val.jsonl \
  --output-dir    /data2/ds85/bgcmodel_runs/phase1_lora \
  --max-seq-len   8192 \
  --batch-size    4 \
  --grad-accum    32 \
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

> **Note on `--grad-accum`:** the script default is `8`, which was chosen
> for trojai's 4-GPU setup and gives an effective batch of 32 at
> `world_size=1`. Setting `--grad-accum 32` here restores the original
> 128-sequence effective batch. See §4 "Effective batch size and throughput".
>
> **Note on `--max-seq-len`:** Start at 8192 rather than 32768. On trojai
> L=4096 already OOM'd in the StripedHyena filter (see §12.3), but that
> was with ZeRO-2 sharding adding 14 GB of filter tensor on top of a
> rank that already held ~14 GB of bf16 weights. On an 80 GB gputee H100
> with LoRA (no optimizer-state term, no 4× weight replication across
> ranks) the same L might fit without activation checkpointing — but do
> not assume it. Run the §12.7 smoke benchmark first.
>
> **Update after 2026-04-25 §12.7 results:** no-checkpoint LoRA reaches
> `L=8192` only at near-ceiling memory (~80.1 GB) and OOMs at `L>=16384`.
> For no-checkpoint training, use `--max-seq-len 4096` as the current safe
> operating point. Reaching `L=32768` requires block-level activation
> checkpointing, which is now implemented as the
> `--activation-checkpointing` flag (see §12.7 "Retest with
> `--activation-checkpointing`"). Run that retest sweep before launching a
> long production run at `L>4096`.

### Resuming from a checkpoint

```bash
export HF_HOME=/data2/ds85/hf_cache
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --num_gpus=1 \
  scripts/finetune_evo2_lora.py \
  --train         data/processed/splits_combined/train.jsonl \
  --val           data/processed/splits_combined/val.jsonl \
  --output-dir    /data2/ds85/bgcmodel_runs/phase1_lora \
  --resume-from   /data2/ds85/bgcmodel_runs/phase1_lora/checkpoints/step_2000 \
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
| `tokens_per_sec` drops 50%+ mid-run | Long sequence batch hit | Expected variance; batches containing 100K+ sequences are slower |
| Peak GPU memory approaching 80 GB   | Sequence length too large; no activation checkpointing | Reduce `--max-seq-len`, or enable block-level checkpointing (see §13 NEXT) |

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

## 12  Smoke-test findings

> **Scope note — 2026-04-22 (gputee).** §§12.1–12.6 below are the
> **trojai (4× A40)** smoke-test findings from 2026-04-15. They are
> preserved verbatim because the bug fixes they document live in the
> training scripts and remain in effect on gputee — all three Evo2↔DS
> integration bugs, the three peft bugs, and the step-counter fix are
> architecture / library quirks, not hardware quirks.
>
> The **memory numbers** in §12.2 and §12.3 (OOMs at L≥1024 for full FT;
> 23.2 GB peak at L=1024 for LoRA) were measured with 4 ranks of ZeRO-2
> on 46 GB A40s. They **do not** directly apply to gputee's single 80 GB
> H100. The gputee re-measurement plan lives in §12.7.

A full integration smoke test was run on trojai (4 steps, L=1024–4096,
batch=1) to validate the training pipeline before committing to a
multi-day run. Every bug discovered and its fix is recorded here.

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
| Activation checkpointing (block-level) | ~4× activation memory | Implemented — `--activation-checkpointing` flag (§12.7) |
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

### 12.7  Gputee smoke-benchmark results (2026-04-25)

> **Status: first full sweep complete on gputee (1x H100 80 GB).**
> Runs were executed via the queued wrapper (`scripts/queue_h100_smoke.sh`)
> so each length started only after the GPU was idle. Artefacts:
> `/data2/ds85/bgcmodel_runs/queued_smoke_20260423_152219/`.

The goal is to answer two concrete questions:

1. What is the peak GPU memory, per sequence length, for LoRA on 1× H100 80 GB?
2. Is block-level activation checkpointing required to reach L=32 768?

Procedure:

```bash
micromamba activate bgcmodel
export HF_HOME=/data2/ds85/hf_cache
mkdir -p /data2/ds85/bgcmodel_runs

for L in 1024 4096 8192 16384 32768; do
  echo "=== L=$L ==="
  OUT=/data2/ds85/bgcmodel_runs/smoke_L${L}
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  deepspeed --num_gpus=1 \
    scripts/finetune_evo2_lora.py \
    --train data/processed/splits_combined/val.jsonl \
    --val   data/processed/splits_combined/val.jsonl \
    --output-dir "$OUT" \
    --max-seq-len $L --batch-size 1 --grad-accum 1 \
    --warmup-steps 2 --max-epochs 1 --max-steps 3 \
    --log-every 1 --val-every 99 --save-every 99 \
    --wandb-mode offline  2>&1 | tee "$OUT.log"
done
```

### Queued smoke benchmark (shared-GPU safe)

If the H100 is in use by someone else, use the queued wrapper script instead of
manually polling `nvidia-smi`:

```bash
cd ~/projects/BCGModelling
scripts/queue_h100_smoke.sh
```

What it does:
- waits for GPU idle (`proc_count==0`) and a free-memory threshold before launch
- requires a continuous idle hold window to reduce false starts
- re-checks idleness before **each** length run in the matrix
- runs the full default set: `1024 4096 8192 16384 32768`
- passes `--smoke-pad-to-max-seq-len` to the trainer by default so each
  train micro-batch is padded to the sweep's `--max-seq-len` (memory reflects
  the requested `L` even when individual JSONL samples are shorter). Use
  `scripts/queue_h100_smoke.sh --no-smoke-pad-to-max-seq-len` to restore the
  old natural-length collation behaviour.
- writes per-length stdout logs and a machine-readable summary table

Default result locations:
- Run root: `/data2/ds85/bgcmodel_runs/queued_smoke_<YYYYmmdd_HHMMSS>/`
- Queue/meta log: `.../queue.log`
- Summary table (first file to check): `.../summary.tsv`
- Per-length trainer stdout: `.../smoke_L<LEN>.log`
- Per-length training artefacts: `.../smoke_L<LEN>/` (includes `train_log.jsonl`)

Useful variants:
```bash
# Skip L=1024 if already measured
scripts/queue_h100_smoke.sh --skip-1024

# Tighten "idle" requirements on a busy shared host
scripts/queue_h100_smoke.sh --min-free-mib 79000 --idle-hold-sec 90
```

After completion, copy peak memory from `summary.tsv` into the table below.

With the checkpoint-size fix (§12.8, `exclude_frozen_parameters=True`)
each smoke run writes ~400 MB of output — small enough to leave
everything on `/data2` without per-run cleanup. On the pre-fix script
each run wrote ~25 GB.

Extract peak GPU memory from each `train_log.jsonl` (`gpu_mem_gb[0]`
field per step) and record here:

| L       | Peak GPU mem (H100, LoRA, batch=1) | Status | Notes |
|---------|-----------------------------------:|--------|-------|
| 1 024   | **23.52 GB** | ✅ Pass | no-AC baseline (`queued_smoke_20260423_152219`) |
| 4 096   | **47.77 GB** | ✅ Pass | no-AC baseline (`queued_smoke_20260423_152219`) |
| 8 192   | **80.10 GB** | ⚠️ Borderline pass | no-AC baseline (`queued_smoke_20260423_152219`) |
| 16 384  | N/A (OOM before step 1 log) | ❌ OOM | no-AC baseline (`queued_smoke_20260423_152219`) |
| 32 768  | N/A (OOM before step 1 log) | ❌ OOM | no-AC baseline (`queued_smoke_20260423_152219`) |

Interpretation from the 2026-04-25 sweep:

1. **No-checkpoint path does not reach L=32,768.** Both 16,384 and 32,768
   fail in the first forward pass with CUDA OOM.
2. **L=8,192 is technically runnable but operationally unsafe.** The run
   completes 3 steps, but peak memory lands at ~80.1 GB on an 80 GB card,
   leaving negligible headroom for runtime variance or shared-host noise.
3. **L=4,096 is the highest clearly stable no-checkpoint point measured so
   far** at this smoke config (`batch_size=1`, `grad_accum=1`).
4. The OOM message appears together with
   `expandable_segments not supported on this platform`, so allocator
   tuning via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not a
   viable fix on this host.

Project decision from these no-checkpoint results:
- **If training without block-level activation checkpointing:** treat
  `L=4096` as the current safe ceiling.
- **For `L=32768` target training:** no-checkpoint path is invalid; AC is
  required.
- **Do not use the 8,192 no-AC smoke pass as production evidence.** It
  indicates near-saturation, not a comfortable operating point.

The result of this benchmark will also replace the "pending" row in the
§4 "Memory at runtime" table.

**What "block-level activation checkpointing" actually means.** This is
PyTorch's `torch.utils.checkpoint` API, applied at the granularity of one
StripedHyena block (Evo2 7B has 32 of them). Details worth recording before
anyone implements it:

1. **Mechanics.** During a standard forward pass every intermediate tensor
   needed by backward is kept in memory until gradients are computed.
   Checkpointing instead saves only the *inputs* to a region; when backward
   needs the intermediates, PyTorch re-runs the forward for that region from
   the saved inputs. You pay ~1 extra forward pass per checkpointed region in
   exchange for collapsing its activation footprint to roughly just its
   inputs.
2. **Why "block-level" specifically.** The dominant activation in Evo2 is the
   Hyena `compute_filter()` output — a `[poles × channels × L]` tensor
   materialised *inside every block* on every forward pass and scaling O(L).
   Under a standard forward, all 32 per-block filter tensors coexist in
   memory until backward starts. Wrapping each block as a checkpoint boundary
   means at any moment only one block's filter tensor has to be live — the
   other 31 are discarded and recomputed one at a time during backward. This
   is the "~4× activation memory" saving quoted in §12.4.
3. **Compute cost.** Expect step wall-clock time to increase by ~33% (a
   `1F + 1B` step becomes roughly `2F + 1B` when every block is wrapped).
   The step count is unchanged. The §4 "Steps and time estimate" a-priori
   number does **not** include this factor and must be multiplied by ~1.33
   if the decision rule above selects the checkpointed path.
4. **Dropout / determinism caveat.** The recomputed forward must produce the
   same intermediates as the original. `--lora-dropout` defaults to `0.05`,
   so the checkpoint call must either use `use_reentrant=False` (the
   recommended modern API, which handles RNG state correctly) or explicitly
   preserve and restore RNG state between the two forwards. With a reentrant
   checkpoint and non-zero dropout the recomputed mask will not match the
   original and gradients will be silently wrong. This is a standard
   `torch.utils.checkpoint` footgun and needs to be verified in
   implementation, not assumed.
5. **Implementation surface.** Evo2 does not load through HuggingFace's
   standard `AutoModel` stack (it uses the StripedHyena `vortex` loader), so
   `model.gradient_checkpointing_enable()` will not "just work". The
   implementation reaches into the block list and wraps each block's
   `forward` in `torch.utils.checkpoint.checkpoint(fn, ..., use_reentrant=False)`.
   As of 2026-04-26 this lives in
   `scripts/finetune_evo2_lora.py::enable_block_activation_checkpointing()`
   and is toggled via `--activation-checkpointing` /
   `--no-activation-checkpointing` (default: enabled).
   It is applied after `apply_lora()` and before `deepspeed.initialize`, so
   peft's Linear adapters and DeepSpeed's engine wrap both interpose around
   the checkpointed blocks rather than inside them.
6. **Secondary use as an L-lever.** Even if the §12.7 benchmark shows L =
   32 768 fits on the H100 *without* checkpointing, block-level checkpointing
   is still the memory mechanism that would let L be pushed past 32 768 —
   which is the only lever in sight for reducing the ~17% long-sequence tail
   documented in §3. Any decision to explore L > 32 768 should route through
   this option and its compute cost first.

#### Retest with activation checkpointing (completed 2026-04-26)

The AC-enabled rerun completed successfully at all five lengths using:

```bash
cd ~/projects/BCGModelling
# checkpointing is now default-on in finetune_evo2_lora.py
scripts/queue_h100_smoke.sh
```

Run root:
- `/data2/ds85/bgcmodel_runs/queued_smoke_20260426_142830/`

Measured peaks (AC on):

| L       | Peak GPU mem (H100, LoRA, batch=1, AC on) | Status | Notes |
|---------|-------------------------------------------:|--------|-------|
| 1 024   | **16.35 GB** | ✅ Pass | down from 23.52 GB (no-AC) |
| 4 096   | **19.10 GB** | ✅ Pass | down from 47.77 GB (no-AC) |
| 8 192   | **22.77 GB** | ✅ Pass | down from 80.10 GB (no-AC) |
| 16 384  | **30.10 GB** | ✅ Pass | no-AC path OOMed |
| 32 768  | **43.92 GB** | ✅ Pass | no-AC path OOMed |

Interpretation from the AC-enabled sweep:

1. **Project target (`L=32768`) is now feasible on 1x H100** for LoRA smoke
   settings, with substantial margin (~44 GB peak on an 80 GB device).
2. **Checkpointing is functioning as intended.** Trainer logs include
   `Activation checkpointing ENABLED: wrapped 32 blocks (use_reentrant=False)`,
   and low-/mid-L memory falls sharply relative to the no-AC baseline.
3. **Next memory frontier is above 32k.** Current data supports probing larger
   windows (`L > 32768`) rather than spending more time re-validating <=32k.

**Caveat on the 2026-04-26 `49152 / 65536 / 98304` probe
(`queued_smoke_20260426_153622`).** That run used natural-length collation
(`batch_size=1` pads only to each sample's token length). The first few
training steps can therefore hit *shorter* sequences than `--max-seq-len`,
which produced identical peak memory and loss traces across those three
lengths — not evidence that 65k/98k tensors actually fit. Re-run that probe
after upgrading `queue_h100_smoke.sh` (default `--smoke-pad-to-max-seq-len`)
and confirm in each `train_log.jsonl` line that `collated_seq_len` equals the
sweep `L` while `content_max_len` reports how many non-pad tokens came from the
JSONL sample.

Follow-up probe (recommended):

```bash
cd ~/projects/BCGModelling
scripts/queue_h100_smoke.sh --lengths "49152 65536 98304"
```

Decision rule for the follow-up:
- If `L=65536` completes with comfortable margin (<~75 GB peak), then
  `L=65536` becomes a practical candidate for long-run planning.
- If `L=98304` OOMs but `L=65536` passes, use the midpoint sweep
  (`73728`, `81920`) to bracket the true ceiling.
- Do not assume `L=131072` is feasible without evidence; linear extrapolation
  from the current AC sweep suggests it likely exceeds 80 GB.

#### Extended-context probe results (completed 2026-04-26 evening)

The follow-up long-context probe was rerun with the upgraded queue script
that defaults to `--smoke-pad-to-max-seq-len`:

- Run root: `/data2/ds85/bgcmodel_runs/queued_smoke_20260426_185444/`
- Summary: `.../summary.tsv`

| L       | Status     | Peak GPU mem (H100, LoRA, batch=1, AC on) | Notes |
|---------|------------|-------------------------------------------:|-------|
| 49 152  | ✅ Pass    | **59.44 GB** | `collated_seq_len=49152` confirmed in `train_log.jsonl` |
| 65 536  | ✅ Pass    | **74.11 GB** | `collated_seq_len=65536` confirmed in `train_log.jsonl` |
| 98 304  | ❌ OOM     | N/A | OOM before step-1 log; `compute_filter()` tried to allocate 24.00 GiB |

Interpretation:

1. **The practical ceiling is now bracketed between 65 536 and 98 304.**
2. **`L=65 536` is feasible but near the H100 limit** (74.11/80 GB under smoke
   settings), so production headroom is limited.
3. **`L=98 304` is not currently feasible** with the present config
   (`batch_size=1`, `grad_accum=1`, AC on).
4. The earlier long-L run (`queued_smoke_20260426_153622`) is now definitively
   superseded by this padded-collation run and should be treated as diagnostic
   history only.

### 12.7.1  Consolidated smoke-test synthesis (current project decision)

Across all completed gputee smoke sweeps:

- **No activation checkpointing (2026-04-25):**
  - stable only to `L=4096`
  - near-saturation at `L=8192` (~80.10 GB)
  - OOM at `L>=16384`
- **With block-level activation checkpointing (2026-04-26):**
  - stable through `L=32768` with large margin (~43.92 GB)
  - stable at `L=49152` (~59.44 GB)
  - stable at `L=65536` (~74.11 GB)
  - OOM at `L=98304`

Current recommendation for production planning:

1. **`L=32768` remains the conservative default** (large margin and strong
   evidence across repeated smoke runs).
2. **`L=65536` is a plausible stretch target** if the project explicitly wants
   more long-sequence coverage and accepts reduced memory headroom + slower
   steps from checkpointing.
3. **Do not plan around `L=98304` or higher** without a model/config change.
4. Any final `L` choice above 32k should be validated again under
   production-like settings (same `grad_accum`, logging/checkpoint cadence,
   and background host conditions) before multi-hour launch.

#### Coverage impact of `L=65536` on current combined train split

Using `data/processed/splits_combined/train.jsonl` (277,238 records), counting
full-length inclusion by `training_text` length:

- `L=32768`: 179,685 / 277,238 = **64.8%** included without truncation
- `L=65536`: 256,875 / 277,238 = **92.7%** included without truncation
- Delta: **+77,190 records** (**+27.8 percentage points**) from 32k -> 65,536

This quantifies the trade-off: 65,536 materially improves full-length coverage,
but requires passing production-like stability checks (not only short smoke
fits) before adoption.

### 12.8  LoRA checkpoint size fix (2026-04-22)

**Discovered during the L=1024 smoke run.** `save_lora_checkpoint`
wrote a 25 GB `mp_rank_00_model_states.pt` file per save — the full
6.5B Evo2 base-model weights — because DeepSpeed's `save_checkpoint`
serialises the whole engine by default. None of those bytes are useful
for LoRA: the frozen base is loaded from the HF cache on every run,
and `PeftModel.from_pretrained` restores the adapter from the peft
checkpoint. The scheduler state and LoRA-trainable optimizer moments
are the only parts needed for resume, and they fit in ~330 MB.

**Fix:** add `exclude_frozen_parameters=True` to the
`model_engine.save_checkpoint(...)` call in
`scripts/finetune_evo2_lora.py::save_lora_checkpoint`. This flag was
added to DeepSpeed specifically for the LoRA/frozen-base case and is
handled symmetrically on load — the frozen params stay at whatever the
base-model init produced, and DeepSpeed only restores the non-excluded
ones.

**Also bundled with this fix:** `final_adapter/` is now a
`shutil.copytree` of `checkpoints/step_<N>_final/adapter/` rather than
a second `model_engine.module.save_pretrained(...)` call. The bytes
are identical; removing the redundant peft call eliminates one
opportunity for the two paths to drift.

**Impact on disk footprint** (see §11 table): per-checkpoint drops
from ~25.4 GB → ~390 MB. A full production run at default retention
(`keep_last_ckpts=5`) goes from ~150 GB in flight to ~3 GB.

**Impact on resume correctness:** none. Verify with a short
smoke-resume sequence:
1. run N steps with `--save-every 1`
2. kill after step 2
3. rerun with `--resume-from .../checkpoints/step_2`
4. confirm step counter + lr + loss continue consistently

The existing `load_lora_checkpoint` already pulls adapter weights via
`PeftModel.from_pretrained` and optimizer/scheduler state via
`model_engine.load_checkpoint`, neither of which depends on the frozen
base being serialised.

**Status:** fix applied on gputee (not backported to trojai docs, per
migration policy). Resume verification is still pending and is the
remaining prerequisite before a long production run.

---

## 11  Files written by the training script

LoRA checkpoints use a different structure from full fine-tune:
the adapter weights are saved via `peft`'s `save_pretrained` (two small
files), while the DeepSpeed scheduler/optimizer state is saved alongside.
**The frozen base-model weights are intentionally excluded from every
checkpoint** — they already live in the HF cache as `evo2_7b.pt`
(~14 GB) and are reloaded via `Evo2("evo2_7b")` at the start of every
run. Re-serialising them into every checkpoint would add ~25 GB per
save for no resume benefit. The excl-frozen behaviour is controlled by
`exclude_frozen_parameters=True` in `save_checkpoint()` (see
`save_lora_checkpoint` in `scripts/finetune_evo2_lora.py`).

```
/data2/ds85/bgcmodel_runs/phase1_lora/
├── config.json                    # all hyperparameters + git hash + timestamp
├── data_fingerprint.json          # line counts + SHA256 of split files
├── deepspeed_config.json          # effective DS config at run time
├── env.txt                        # pip freeze output
├── train_log.jsonl                # per-step metrics (step, loss, lr, grad_norm, ...)
├── val_log.jsonl                  # per-validation-run metrics
├── plots/                         # loss.png, lr.png, grad_norm.png, throughput.png, summary.png
├── checkpoints/
│   ├── step_500/
│   │   ├── adapter/
│   │   │   ├── adapter_config.json      # LoRA config (r, alpha, target_modules, ...)
│   │   │   └── adapter_model.safetensors # ~55 MB for r=16 (28.7M params × 2 bytes)
│   │   ├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
│   │   │                                 # ZeRO-2 optimizer shard for LoRA-trainable params only (~330 MB)
│   │   └── mp_rank_00_model_states.pt   # ~few MB: scheduler + client_state + LoRA-param model state
│   │                                     # (NOT the 6.5B frozen base; see exclude_frozen_parameters above)
│   ├── step_1000/  ...            # same layout
│   ├── step_1500/  ...            # older ones deleted per keep_last_ckpts retention
│   ├── best/                      # full copy of best-val-loss step_N/ at the time it was saved
│   └── step_<N>_final/            # written once at end-of-training
├── final_adapter/                 # copy of checkpoints/step_<N>_final/adapter/
│   ├── adapter_config.json        #   (same bytes; exported for a stable load path)
│   └── adapter_model.safetensors
└── samples/                       # only written if sample-generation is enabled
    ├── step_500.fasta             # 4 generated sequences
    ├── step_500_eval.json         # M1 + M2 results on those sequences
    └── ...
```

Single-GPU on gputee means only one ZeRO rank (`rank_0`); on trojai
the `step_N/` directories also held `zero_pp_rank_{1,2,3}_…` shards.
Those are not produced at `world_size=1`.

**Per-checkpoint disk footprint (gputee, with `--lora-r 16`):**

| Artefact | Size | Notes |
|---|---:|---|
| `adapter/adapter_model.safetensors` | ~55 MB | 28.7M params × 2 bytes (bf16) |
| `bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt` | ~330 MB | fp32 master + m + v for 28.7M params |
| `mp_rank_00_model_states.pt` | ~few MB | scheduler + client_state + trainable params only |
| **Total per `step_N/`** | **~390 MB** | |
| `keep_last_ckpts=5` steady state | ~2 GB | 5 × 390 MB |
| `best/` + `step_<N>_final/` + `final_adapter/` | +~830 MB | |
| **Typical full-run on-disk max** | **~3 GB** | plus `plots/`, logs, optional `samples/` |

For comparison, the pre-fix layout wrote the full 6.5B base model into
`mp_rank_00_model_states.pt` at every save (~25 GB per checkpoint, so
`keep_last_ckpts=5` meant 127 GB in flight plus another 25 GB for the
`best/` copy). The `exclude_frozen_parameters=True` change makes that
go away without affecting resume semantics — see §12.8.

### Final adapter export

At end-of-training the script additionally writes:
```
/data2/ds85/bgcmodel_runs/phase1_lora/final_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

This is a **copy** of the adapter from the most recent `step_<N>_final/`
checkpoint (via `shutil.copytree`, not a second `peft.save_pretrained`
call — guarantees byte-identical contents). The `final_adapter/` path
is the canonical inference-time load target and will not be affected
by later retention-based cleanup of the `checkpoints/` subtree.

### Using the adapter for inference

```python
from peft import PeftModel
import evo2

base_model = evo2.Evo2("evo2_7b")
model = PeftModel.from_pretrained(
    base_model,
    "/data2/ds85/bgcmodel_runs/phase1_lora/final_adapter",
)
model.eval()
```
