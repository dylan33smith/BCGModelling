# BCGModelling — CLAUDE.md

This file gives AI coding agents project-specific context so they can work
correctly without repeated setup explanations.

## Purpose

- Keep edits aligned with this repo's actual workflow and constraints.
- Prioritize reproducible, documented decisions over speculative refactors.
- Treat `docs/gputee/FINETUNE_GUIDE.md` and `docs/gputee/PROJECT_GUIDE.md`
  as the primary project-memory sources for ongoing model-training work.

## Project Snapshot

- Goal: fine-tune Evo2 7B for BGC sequence generation/evaluation.
- Current production host focus: `gputee` (single H100 80 GB).
- Training strategy: LoRA adapters on Evo2 (not full-parameter FT).
- Orchestration stack: DeepSpeed + PEFT + PyTorch (bf16).
- Data split path currently used for smoke runs:
  `data/processed/splits_combined/{train,val}.jsonl`.

## Source of Truth

- Training implementation:
  `scripts/finetune_evo2_lora.py`
- Smoke queue wrapper:
  `scripts/queue_h100_smoke.sh`
- Main runbook and findings:
  `docs/gputee/FINETUNE_GUIDE.md`
- Project status / priorities:
  `docs/gputee/PROJECT_GUIDE.md`

## Current Decisions (as of latest smoke sweeps)

- Block-level activation checkpointing is implemented and default-on in
  `scripts/finetune_evo2_lora.py` (opt out with
  `--no-activation-checkpointing`).
- Queue smoke runs default to padded train collation via
  `--smoke-pad-to-max-seq-len` so measured memory reflects requested `L`.
- No-checkpoint path is not viable above short contexts on H100.
- With checkpointing + padded sweeps:
  - `L=32768` passes with large margin.
  - `L=65536` passes but is near limit.
  - `L=98304` OOMs.
- Conservative default remains `L=32768`; `L=65536` is stretch/conditional.

## Common Commands

Environment:

```bash
cd ~/projects/BCGModelling
micromamba activate bgcmodel
export HF_HOME=/data2/ds85/hf_cache
```

Shared-GPU-safe smoke matrix (default lengths):

```bash
scripts/queue_h100_smoke.sh
```

Long-context probe:

```bash
scripts/queue_h100_smoke.sh --lengths "49152 65536 98304"
```

Disable padded smoke collation (diagnostic only):

```bash
scripts/queue_h100_smoke.sh --no-smoke-pad-to-max-seq-len
```

Single run sanity:

```bash
deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
  --train data/processed/splits_combined/val.jsonl \
  --val   data/processed/splits_combined/val.jsonl \
  --output-dir /data2/ds85/bgcmodel_runs/ac_sanity \
  --max-seq-len 1024 --batch-size 1 --grad-accum 1 \
  --warmup-steps 2 --max-epochs 1 --max-steps 3 \
  --log-every 1 --val-every 99 --save-every 99 \
  --wandb-mode offline
```

## Working Conventions for Agents

- Make targeted edits; avoid broad cleanup unless requested.
- Update docs in the same change when behavior/flags/decisions change.
- Do not claim memory conclusions from long-L sweeps unless
  `train_log.jsonl` confirms `collated_seq_len == L`.
- For benchmark interpretation, check both:
  - `summary.tsv` (status/peaks)
  - per-length `smoke_L*.log` (actual failure signatures).
- Prefer explicit CLI flags over hidden behavior changes.

## Validation Expectations

- After code edits:
  - run syntax checks for touched scripts;
  - run lint checks where available;
  - verify new CLI flags appear in `--help`.
- After smoke runs:
  - verify `summary.tsv`;
  - inspect `train_log.jsonl` for `gpu_mem_gb`, `collated_seq_len`,
    and `content_max_len`;
  - inspect failing `smoke_L*.log` for exact OOM site.

## Known Gotchas

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reports unsupported on
  this platform; do not rely on it as an OOM fix.
- NCCL "process group not destroyed" warning on shutdown is expected in these
  short smoke runs.
- Shared host contention can invalidate measurements; use queued runs with
  idle-gpu gating.
