#!/usr/bin/env python3
"""Fine-tune Evo2 7B on BGC data using LoRA adapters (PEFT).

Phase 1 conditioning: COMPOUND_CLASS + taxonomic tag, no COMPOUND token.

LoRA vs full fine-tune
----------------------
Full fine-tune OOMs on 4× A40 (48 GB) because StripedHyena's long-conv
filter activations are large (~14 GB at L=1024) and the AdamW optimizer
states for 6.48B params are ~56 GB total. With LoRA:
  - Base weights are frozen → no optimizer states for frozen params
  - Only adapter params are trained (~28 M at rank 16 vs 6.48 B)
  - AdamW states drop from ~56 GB → ~336 MB
  - Forward/backward still run through frozen base, so activations are
    unchanged — but the critical OOM (optimizer.step) is eliminated
  - Activation checkpointing still recommended for L > 4096

LoRA targets (all nn.Linear layers in Evo2):
  - Attention blocks (every ~8th of 32): Wqkv, out_proj
  - All blocks: out_filter_dense (Hyena dense output projection)
  - All blocks MLP: l1, l2, l3

Trainable parameter budget (r=16):
  - Attention (4 blocks × Wqkv + out_proj):   ~1.6 M
  - Hyena filter output (28 blocks):           ~3.7 M
  - MLP (32 blocks × l1,l2,l3):              ~23.2 M
  - Total:                                   ~28.5 M  (0.44% of 6.48 B)

Launch
------
    conda activate bgcmodel
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 \\
        scripts/finetune_evo2_lora.py \\
        --train         data/processed/splits_combined/train.jsonl \\
        --val           data/processed/splits_combined/val.jsonl \\
        --output-dir    checkpoints/phase1_lora \\
        --wandb-project bcg-evo2-phase1

Full documentation: FINETUNE_GUIDE.md §12.

Project design notes:
  - Phase 1 is class-only. COMPOUND token stripped during merge step.
  - Goal is transposition, not invention (FINETUNE_GUIDE.md §10).
  - Reproducibility: config + data fingerprint + pip freeze saved at start.
  - Three parallel viz paths: WandB, JSONL logs, PNG plots.
  - Adapter weights saved in peft format (adapter_config.json + adapter_model.safetensors)
    alongside DeepSpeed optimizer/scheduler state, so checkpoints are
    loadable both via peft and DeepSpeed resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────

EVO2_MODEL_NAME = "evo2_7b_262k"
PAD_TOKEN_ID    = 0
IGNORE_INDEX    = -100

# Base hyperparameters — same schedule as full fine-tune
DEFAULTS = dict(
    max_seq_len       = 32768,
    batch_size        = 4,
    grad_accum        = 8,
    lr                = 5e-5,   # higher than full FT (1e-5) — LoRA adapters can use more LR
    lr_min_ratio      = 0.1,
    warmup_steps      = 200,
    max_epochs        = 2,
    weight_decay      = 0.01,   # lower WD for LoRA (adapters are tiny; heavy WD hurts)
    grad_clip         = 1.0,
    beta1             = 0.9,
    beta2             = 0.95,
    seed              = 42,
    log_every         = 10,
    val_every         = 250,
    save_every        = 500,
    val_max_batches   = 500,
    keep_last_ckpts   = 5,
)

# LoRA-specific defaults
LORA_DEFAULTS = dict(
    lora_r           = 16,    # adapter rank
    lora_alpha       = 32,    # scaling = alpha / r = 2.0
    lora_dropout     = 0.05,  # light regularisation
)

# LoRA target modules — suffix-matched against all named Linear layers
# Covers attention (Wqkv, out_proj), Hyena dense (out_filter_dense), MLP (l1,l2,l3)
LORA_TARGET_MODULES = [
    "Wqkv",
    "out_proj",
    "out_filter_dense",
    "l1",
    "l2",
    "l3",
]


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--train",      type=Path, required=True)
    p.add_argument("--val",        type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-name",   type=str,  default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    for k, v in LORA_DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    p.add_argument("--lora-targets", type=str, nargs="+",
                   default=LORA_TARGET_MODULES,
                   help="Linear module name suffixes to add LoRA adapters to")
    p.add_argument("--max-steps",  type=int, default=0,
                   help="If >0, stop after this many optimizer steps (smoke tests)")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Adapter checkpoint directory to resume from")
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
# Distributed helpers  (identical to finetune_evo2.py)
# ────────────────────────────────────────────────────────────────────────

def ddp_rank() -> int:        return int(os.environ.get("RANK", 0))
def ddp_local_rank() -> int:  return int(os.environ.get("LOCAL_RANK", 0))
def ddp_world_size() -> int:  return int(os.environ.get("WORLD_SIZE", 1))
def is_main() -> bool:        return ddp_rank() == 0

def rank0_print(*msg, **kw) -> None:
    if is_main():
        print(f"[rank0 {datetime.now().strftime('%H:%M:%S')}]", *msg, **kw, flush=True)


# ────────────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def save_config(args: argparse.Namespace, output_dir: Path) -> None:
    cfg = {
        "hyperparameters": {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in vars(args).items()},
        "git_commit":      git_commit_hash(),
        "hostname":        os.uname().nodename,
        "start_time_utc":  datetime.now(timezone.utc).isoformat(),
        "torch_version":   torch.__version__,
        "cuda_version":    torch.version.cuda,
        "world_size":      ddp_world_size(),
        "gpus":            [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())],
        "mode":            "lora",
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def save_data_fingerprint(args: argparse.Namespace, output_dir: Path) -> None:
    def fp(path: Path) -> dict:
        n, hasher = 0, hashlib.sha256()
        with path.open("rb") as f:
            for i, line in enumerate(f):
                n += 1
                if i < 100:
                    hasher.update(line)
        return {"path": str(path), "lines": n,
                "sha256_first_100": hasher.hexdigest()}
    (output_dir / "data_fingerprint.json").write_text(
        json.dumps({"train": fp(args.train), "val": fp(args.val)}, indent=2))


def save_env(output_dir: Path) -> None:
    try:
        pf = subprocess.check_output(["pip", "freeze"],
                                     stderr=subprocess.DEVNULL).decode()
        (output_dir / "env.txt").write_text(pf)
    except Exception as e:
        (output_dir / "env.txt").write_text(f"pip freeze failed: {e}\n")


# ────────────────────────────────────────────────────────────────────────
# Dataset  (identical to finetune_evo2.py)
# ────────────────────────────────────────────────────────────────────────

class BGCTextDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int) -> None:
        self.jsonl_path  = jsonl_path
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.offsets: list[int] = []
        with jsonl_path.open("rb") as f:
            offset = 0
            while True:
                line = f.readline()
                if not line:
                    break
                self.offsets.append(offset)
                offset += len(line)
        rank0_print(f"  {jsonl_path.name}: indexed {len(self.offsets):,} records")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with self.jsonl_path.open("rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        rec  = json.loads(line)
        text = rec["training_text"]
        tokens = self.tokenizer.tokenize(text)
        ids = list(tokens) if isinstance(tokens, (list, tuple)) else [int(t) for t in tokens]
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
        return {
            "input_ids": ids,
            "length":    len(ids),
            "accession": rec.get("accession", ""),
            "class":     rec.get("compound_class", ""),
        }


def collate_pad(batch: list[dict[str, Any]],
                pad_id: int = PAD_TOKEN_ID) -> dict[str, torch.Tensor]:
    max_len = max(b["length"] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        L   = b["length"]
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        input_ids[i, :L] = ids
        labels[i, :L]    = ids
    return {"input_ids": input_ids, "labels": labels}


# ────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────

def causal_lm_loss(logits: torch.Tensor,
                   labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


# ────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(engine, val_loader,
                   local_rank: int, max_batches: int) -> float:
    engine.eval()
    total_loss = 0.0
    n_tokens   = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
        labels    = batch["labels"].to(local_rank, non_blocking=True)
        logits, _ = engine(input_ids)
        loss = causal_lm_loss(logits, labels)
        n = (labels[..., 1:] != IGNORE_INDEX).sum().item()
        total_loss += loss.item() * n
        n_tokens   += n
    if torch.distributed.is_initialized():
        t = torch.tensor([total_loss, n_tokens], dtype=torch.float64,
                         device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_loss, n_tokens = t.tolist()
    engine.train()
    return total_loss / max(n_tokens, 1)


# ────────────────────────────────────────────────────────────────────────
# DeepSpeed config  (ZeRO-2, same as full FT; optimizer state is tiny now)
# ────────────────────────────────────────────────────────────────────────

def build_ds_config(args: argparse.Namespace, world_size: int,
                    total_steps: int) -> dict[str, Any]:
    effective_batch = args.batch_size * world_size * args.grad_accum
    return {
        "train_batch_size":               effective_batch,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps":    args.grad_accum,
        "gradient_clipping":              args.grad_clip,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage":                2,
            "allgather_partitions": True,
            "reduce_scatter":       True,
            "overlap_comm":         True,
            "contiguous_gradients": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size":    5e8,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr":           args.lr,
                "betas":        [args.beta1, args.beta2],
                "eps":          1e-8,
                "weight_decay": args.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps":  total_steps,
                "warmup_num_steps": args.warmup_steps,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio":    args.lr_min_ratio,
                "warmup_type":      "linear",
            },
        },
        "steps_per_print":       max(args.log_every, 1),
        "wall_clock_breakdown":  False,
    }


# ────────────────────────────────────────────────────────────────────────
# LoRA application
# ────────────────────────────────────────────────────────────────────────

def apply_lora(model: torch.nn.Module, args: argparse.Namespace,
               ) -> torch.nn.Module:
    """Wrap model with peft LoRA adapters and freeze base weights."""
    from peft import LoraConfig, get_peft_model

    # Verify requested targets exist in the model
    found = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            for t in args.lora_targets:
                if name.endswith(t):
                    found.add(t)
    missing = set(args.lora_targets) - found
    if missing:
        rank0_print(f"  WARNING: LoRA targets not found in model: {missing}")

    # Fix 1: peft's BaseTuner.get_model_config() calls model.config.to_dict().
    # Evo2's vortex dotdict returns None for missing attribute lookups
    # (rather than raising AttributeError), so hasattr() lies and
    # model.config.to_dict is None → 'NoneType' object is not callable.
    # Always inject the shim unconditionally via dict-key assignment.
    try:
        model.config["to_dict"] = lambda: {
            k: v for k, v in model.config.items()
            if not callable(v)  # skip any function/class values
        }
    except Exception:
        pass  # last-resort: if config isn't mutable, peft may still work

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        bias="none",
        inference_mode=False,
        # task_type omitted: CAUSAL_LM adds PeftModelForCausalLM wrapper
        # which expects HF-style attributes Evo2 doesn't have.
    )
    # Fix 2: peft 0.19 tries to auto-cast adapters to float8_e8m0fnu,
    # which doesn't exist in torch 2.5. Disable with autocast_adapter_dtype=False.
    peft_model = get_peft_model(model, config, autocast_adapter_dtype=False)

    # Summary
    trainable, total = 0, 0
    for p in peft_model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    rank0_print(f"  LoRA applied: r={args.lora_r}  alpha={args.lora_alpha}  "
                f"dropout={args.lora_dropout}")
    rank0_print(f"  Trainable params: {trainable:,} / {total:,}  "
                f"({100*trainable/total:.3f}%)")
    rank0_print(f"  Target modules:   {args.lora_targets}")

    return peft_model


# ────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (LoRA-specific: save adapter + DS state separately)
# ────────────────────────────────────────────────────────────────────────

def save_lora_checkpoint(model_engine, args: argparse.Namespace,
                         ckpt_root: Path, tag: str,
                         client_state: dict) -> None:
    """Save both the peft adapter (human-readable) and the DeepSpeed
    optimizer/scheduler state (needed for exact resume)."""
    ckpt_dir = ckpt_root / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. DeepSpeed checkpoint — saves ZeRO-2 optimizer shards across all ranks
    model_engine.save_checkpoint(str(ckpt_root), tag=tag,
                                 client_state=client_state)

    # 2. peft adapter weights — rank 0 only (ZeRO-2 keeps full weights on each rank)
    if is_main():
        adapter_dir = ckpt_dir / "adapter"
        # model_engine.module is the peft model
        model_engine.module.save_pretrained(str(adapter_dir))
        rank0_print(f"  Adapter saved → {adapter_dir}")


def load_lora_checkpoint(model_engine, resume_from: Path) -> tuple[int, float]:
    """Load DeepSpeed optimizer state for resume. Returns (step, best_val_loss)."""
    _, client_state = model_engine.load_checkpoint(
        str(resume_from.parent), tag=resume_from.name
    )
    step          = int(client_state.get("step", 0)) if client_state else 0
    best_val_loss = float(client_state.get("best_val_loss", float("inf"))) if client_state else float("inf")
    return step, best_val_loss


# ────────────────────────────────────────────────────────────────────────
# Plot generation  (identical to finetune_evo2.py)
# ────────────────────────────────────────────────────────────────────────

def generate_plots(output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    train_log = output_dir / "train_log.jsonl"
    val_log   = output_dir / "val_log.jsonl"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    if not train_log.exists():
        return

    train = [json.loads(l) for l in train_log.open()]
    val   = [json.loads(l) for l in val_log.open()] if val_log.exists() else []
    if not train:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([e["step"] for e in train], [e["train_loss"] for e in train],
            label="train", alpha=0.8, linewidth=1)
    if val:
        ax.plot([e["step"] for e in val], [e["val_loss"] for e in val],
                label="val", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training / Validation Loss (LoRA)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "loss.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], [e["lr"] for e in train], color="C2")
    ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.set_title("LR Schedule")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "lr.png", dpi=100); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], [e["grad_norm"] for e in train],
            color="C3", alpha=0.7, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("grad norm"); ax.set_title("Gradient Norm")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "grad_norm.png", dpi=100); plt.close(fig)

    tps = [e.get("tokens_per_sec", 0) for e in train]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], tps, color="C4", alpha=0.7, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("tokens/sec"); ax.set_title("Throughput")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plots_dir / "throughput.png", dpi=100); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax = axes[0, 0]
    ax.plot([e["step"] for e in train], [e["train_loss"] for e in train],
            label="train", alpha=0.6, linewidth=1)
    if val:
        ax.plot([e["step"] for e in val], [e["val_loss"] for e in val],
                label="val", linewidth=2, marker="o", markersize=4)
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("step")
    axes[0,1].plot([e["step"] for e in train], [e["lr"] for e in train], color="C2")
    axes[0,1].set_title("LR (log)"); axes[0,1].set_yscale("log"); axes[0,1].grid(alpha=0.3)
    axes[0,1].set_xlabel("step")
    axes[1,0].plot([e["step"] for e in train], [e["grad_norm"] for e in train],
                   color="C3", alpha=0.6, linewidth=1)
    axes[1,0].set_title("Gradient norm"); axes[1,0].grid(alpha=0.3); axes[1,0].set_xlabel("step")
    axes[1,1].plot([e["step"] for e in train], tps, color="C4", alpha=0.6, linewidth=1)
    axes[1,1].set_title("Throughput (tok/s)"); axes[1,1].grid(alpha=0.3); axes[1,1].set_xlabel("step")
    fig.suptitle(f"LoRA training — {output_dir.name}", y=0.995)
    fig.tight_layout(); fig.savefig(plots_dir / "summary.png", dpi=110); plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def gpu_memory_gb() -> list[float]:
    return [torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())]

def cleanup_old_checkpoints(ckpt_root: Path, keep_last: int) -> None:
    if not ckpt_root.exists():
        return
    step_dirs = sorted(
        [p for p in ckpt_root.iterdir()
         if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    for stale in step_dirs[:-keep_last]:
        try:
            subprocess.run(["rm", "-rf", str(stale)], check=False)
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ────────────────────────────────────────────────────────────────────────

_SHOULD_STOP = False

def _set_stop_flag(signum, frame):
    global _SHOULD_STOP
    _SHOULD_STOP = True
    rank0_print(f"Signal {signum} received; will checkpoint after current step")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Per-rank GPU masking (see FINETUNE_GUIDE.md §12.1 for explanation) ──
    # Evo2's StripedHyena loader auto-shards layers across all visible CUDA
    # devices. Restrict each worker to its own GPU before any CUDA init.
    local_rank_env = ddp_local_rank()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank_env)
    os.environ["LOCAL_RANK"] = "0"   # DS reads this for device_id in init_distributed
    local_rank = 0
    args.local_rank = 0              # DS sanity-checks args.local_rank == env LOCAL_RANK

    seed_everything(args.seed + ddp_rank())

    import deepspeed
    import evo2

    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    # Run setup
    if args.run_name is None:
        args.run_name = f"{args.output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)

    if is_main():
        save_config(args, args.output_dir)
        save_data_fingerprint(args, args.output_dir)
        save_env(args.output_dir)
        rank0_print(f"Output dir: {args.output_dir}")
        rank0_print(f"Run name:   {args.run_name}")
        rank0_print(f"World size: {ddp_world_size()}  (RANK={ddp_rank()}  LOCAL_RANK={local_rank})")
        rank0_print(f"Mode:       LoRA  r={args.lora_r}  alpha={args.lora_alpha}")

    # WandB
    wandb_run = None
    if is_main() and args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
                mode=args.wandb_mode,
                resume="allow",
                dir=str(args.output_dir),
                tags=["lora", f"r{args.lora_r}"],
            )
            rank0_print(f"WandB: {wandb_run.url}")
        except Exception as e:
            rank0_print(f"WandB init failed ({e}); continuing without it")

    signal.signal(signal.SIGTERM, _set_stop_flag)
    signal.signal(signal.SIGINT,  _set_stop_flag)

    # ── Load Evo2 base model ──────────────────────────────────────────
    rank0_print("Loading Evo2 7B base model + tokenizer...")
    t0 = time.time()
    evo_wrapper = evo2.Evo2(EVO2_MODEL_NAME)
    model     = evo_wrapper.model
    tokenizer = evo_wrapper.tokenizer
    rank0_print(f"  Loaded in {time.time()-t0:.0f} s")
    rank0_print(f"  Base params: {sum(p.numel() for p in model.parameters()):,}")

    # Fix non-contiguous Wqkv tensors (FINETUNE_GUIDE.md §12.1 Bug 2)
    # AND: clone any inference-mode tensors so backward can save them.
    # Evo2 config sets inference_mode=True, which tags some tensors (e.g.
    # RMSNorm scale) as inference tensors. Autograd cannot save these for
    # backward, raising "Inference tensors cannot be saved for backward."
    # clone() produces a normal (non-inference-mode) tensor.
    n_fixed = 0
    n_cloned = 0
    with torch.no_grad():
        for p in model.parameters():
            needs_clone = hasattr(p.data, "is_inference") and p.data.is_inference()
            if not p.is_contiguous() or needs_clone:
                p.data = p.data.clone().contiguous()
                n_fixed += 1
                if needs_clone:
                    n_cloned += 1
        # Buffers: need to replace via the owning module
        for mod_name, mod in model.named_modules():
            for buf_name, buf in list(mod.named_buffers(recurse=False)):
                if buf is None:
                    continue
                needs_clone = hasattr(buf, "is_inference") and buf.is_inference()
                if (not buf.is_contiguous()) or needs_clone:
                    setattr(mod, buf_name, buf.clone().contiguous())
                    n_fixed += 1
                    if needs_clone:
                        n_cloned += 1
    rank0_print(f"  Fixed {n_fixed} tensors ({n_cloned} cloned out of inference mode)")
    # Ensure the model is in training mode (Evo2 loads in eval/inference mode)
    model.train()

    # ── Apply LoRA ────────────────────────────────────────────────────
    # Load adapter from checkpoint if resuming, otherwise apply fresh config.
    if args.resume_from is not None and (args.resume_from / "adapter").exists():
        from peft import PeftModel
        rank0_print(f"  Loading LoRA adapter from {args.resume_from / 'adapter'}")
        model = PeftModel.from_pretrained(
            model, str(args.resume_from / "adapter"), is_trainable=True
        )
        rank0_print("  Adapter loaded (weights restored)")
    else:
        model = apply_lora(model, args)

    # ── Datasets ──────────────────────────────────────────────────────
    rank0_print("Indexing datasets...")
    train_ds = BGCTextDataset(args.train, tokenizer, args.max_seq_len)
    val_ds   = BGCTextDataset(args.val,   tokenizer, args.max_seq_len)

    world_size    = ddp_world_size()
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=ddp_rank(), shuffle=True, seed=args.seed)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=ddp_rank(), shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        collate_fn=collate_pad, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=collate_pad, num_workers=2, pin_memory=True, drop_last=False,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps     = steps_per_epoch * args.max_epochs
    rank0_print(f"Train batches/rank: {len(train_loader):,}")
    rank0_print(f"Steps per epoch:    {steps_per_epoch:,}  (grad_accum={args.grad_accum})")
    rank0_print(f"Total steps:        {total_steps:,}  (epochs={args.max_epochs})")

    # ── DeepSpeed init ────────────────────────────────────────────────
    # Only trainable (adapter) parameters are passed — frozen base params
    # are excluded, so ZeRO-2 only shards the ~28 M adapter gradients/states.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    rank0_print(f"Trainable tensors passed to DeepSpeed: {len(trainable_params)}")

    ds_config = build_ds_config(args, world_size, total_steps)
    if is_main():
        (args.output_dir / "deepspeed_config.json").write_text(
            json.dumps(ds_config, indent=2))

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable_params,
        config=ds_config,
    )

    # ── Resume optimizer state ────────────────────────────────────────
    start_step    = 0
    best_val_loss = float("inf")
    if args.resume_from is not None and args.resume_from.exists():
        rank0_print(f"Loading DeepSpeed checkpoint from {args.resume_from}")
        start_step, best_val_loss = load_lora_checkpoint(model_engine, args.resume_from)
        rank0_print(f"  Resumed at step {start_step}  best_val_loss={best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────
    rank0_print(f"Starting LoRA training from step {start_step}")
    model_engine.train()

    step             = start_step
    log_buffer_loss  = 0.0
    log_buffer_count = 0
    last_log_time    = time.time()
    last_log_tokens  = 0

    train_log_path = args.output_dir / "train_log.jsonl"
    val_log_path   = args.output_dir / "val_log.jsonl"
    ckpt_root      = args.output_dir / "checkpoints"

    try:
        for epoch in range(args.max_epochs):
            train_sampler.set_epoch(epoch)
            rank0_print(f"── Epoch {epoch+1}/{args.max_epochs} ──")

            for micro_step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
                labels    = batch["labels"].to(local_rank, non_blocking=True)

                logits, _ = model_engine(input_ids)
                loss = causal_lm_loss(logits, labels)

                model_engine.backward(loss)
                model_engine.step()

                # Average loss across ranks so logged train_loss reflects
                # the true mean over all GPUs' data shards (not rank-0 only).
                loss_scalar = loss.detach()
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(loss_scalar,
                                                 op=torch.distributed.ReduceOp.SUM)
                    loss_scalar = loss_scalar / world_size
                log_buffer_loss  += loss_scalar.item()
                log_buffer_count += 1
                last_log_tokens  += input_ids.numel() * world_size

                # DS's `global_steps` is the authoritative optimizer-step counter:
                # it is incremented exactly once per accumulation boundary, inside
                # `_take_model_step`, AFTER optimizer.step() and scheduler.step().
                # DO NOT use `is_gradient_accumulation_boundary()` here — that check
                # returns True one micro-batch EARLY when called after engine.step()
                # (formula: (micro_steps + 1) % ga == 0, designed for pre-step() use).
                # At ga=1 the two coincide (always True), but at ga>1 it would cause
                # logged grad_norm/lr/checkpoints to reflect the previous opt step.
                if model_engine.global_steps != step:
                    step = model_engine.global_steps

                    # ── Logging
                    if step % args.log_every == 0:
                        avg_loss  = log_buffer_loss / max(log_buffer_count, 1)
                        lr        = scheduler.get_last_lr()[0] if scheduler else args.lr
                        grad_norm = float(model_engine.get_global_grad_norm() or 0.0)
                        elapsed   = time.time() - last_log_time
                        tps       = last_log_tokens / max(elapsed, 1e-3)

                        if is_main():
                            entry = {
                                "step":           step,
                                "epoch":          round(step / max(steps_per_epoch,1), 4),
                                "train_loss":     round(avg_loss, 6),
                                "lr":             lr,
                                "grad_norm":      round(grad_norm, 4),
                                "gpu_mem_gb":     [round(x, 2) for x in gpu_memory_gb()],
                                "tokens_per_sec": int(tps),
                                "elapsed_sec":    int(time.time() - t0),
                            }
                            append_jsonl(train_log_path, entry)
                            if wandb_run:
                                wandb_run.log(
                                    {k: v for k, v in entry.items()
                                     if not isinstance(v, list)}, step=step)
                            if step % (args.log_every * 10) == 0:
                                rank0_print(
                                    f"step {step:5d} | ep {entry['epoch']:.2f} | "
                                    f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                                    f"gn {grad_norm:.3f} | {int(tps):,} tok/s"
                                )

                        log_buffer_loss  = 0.0
                        log_buffer_count = 0
                        last_log_time    = time.time()
                        last_log_tokens  = 0

                    # ── Validation
                    if step % args.val_every == 0:
                        val_loss = run_validation(
                            model_engine, val_loader, local_rank, args.val_max_batches)
                        if is_main():
                            ventry = {
                                "step":     step,
                                "val_loss": round(val_loss, 6),
                                "val_ppl":  round(math.exp(min(val_loss, 20)), 4),
                            }
                            append_jsonl(val_log_path, ventry)
                            if wandb_run:
                                wandb_run.log(ventry, step=step)
                            rank0_print(
                                f"  VAL @ step {step}: loss {val_loss:.4f}  "
                                f"ppl {math.exp(min(val_loss,20)):.3f}  "
                                f"(best: {best_val_loss:.4f})"
                            )
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                rank0_print("  NEW best val_loss → saving best/ checkpoint")

                        # Broadcast best_val_loss to all ranks
                        if torch.distributed.is_initialized():
                            t_tensor = torch.tensor(
                                [best_val_loss, val_loss],
                                dtype=torch.float64, device=f"cuda:{local_rank}")
                            torch.distributed.broadcast(t_tensor, 0)
                            best_val_loss = t_tensor[0].item()
                            current_val   = t_tensor[1].item()
                        else:
                            current_val = val_loss

                        if current_val <= best_val_loss + 1e-9:
                            save_lora_checkpoint(
                                model_engine, args, ckpt_root, "best",
                                {"step": step, "best_val_loss": best_val_loss})

                    # ── Periodic checkpoint
                    if step % args.save_every == 0:
                        if is_main():
                            rank0_print(f"  Saving checkpoint step_{step}...")
                        save_lora_checkpoint(
                            model_engine, args, ckpt_root, f"step_{step}",
                            {"step": step, "best_val_loss": best_val_loss})
                        if is_main():
                            cleanup_old_checkpoints(ckpt_root, keep_last=args.keep_last_ckpts)
                            generate_plots(args.output_dir)
                            rank0_print("  Checkpoint saved; plots refreshed")

                    # ── Smoke-test exit
                    if args.max_steps and step >= args.max_steps:
                        rank0_print(f"Reached --max-steps={args.max_steps}; exiting")
                        if is_main():
                            generate_plots(args.output_dir)
                        return

                    # ── Graceful shutdown
                    if _SHOULD_STOP:
                        rank0_print("Graceful shutdown → saving interrupted checkpoint")
                        save_lora_checkpoint(
                            model_engine, args, ckpt_root, f"step_{step}_interrupted",
                            {"step": step, "best_val_loss": best_val_loss})
                        if is_main():
                            generate_plots(args.output_dir)
                        return

                    if step >= total_steps:
                        break
            if step >= total_steps:
                break

    except torch.cuda.OutOfMemoryError as e:
        rank0_print(f"CUDA OOM: {e}")
        rank0_print("Attempting emergency checkpoint...")
        try:
            save_lora_checkpoint(
                model_engine, args, ckpt_root, f"step_{step}_oom",
                {"step": step, "best_val_loss": best_val_loss})
        except Exception as ee:
            rank0_print(f"Emergency checkpoint failed: {ee}")
        raise

    finally:
        # Final adapter export (clean peft format, no DS wrapper)
        if step > start_step:
            rank0_print(f"Saving final checkpoint at step {step}")
            try:
                save_lora_checkpoint(
                    model_engine, args, ckpt_root, f"step_{step}_final",
                    {"step": step, "best_val_loss": best_val_loss})
                if is_main():
                    # Export a clean merged-free adapter at the top level for easy loading
                    final_adapter = args.output_dir / "final_adapter"
                    model_engine.module.save_pretrained(str(final_adapter))
                    rank0_print(f"Final adapter exported → {final_adapter}")
            except Exception as e:
                rank0_print(f"Final checkpoint/export failed: {e}")
            if is_main():
                generate_plots(args.output_dir)

        if wandb_run:
            wandb_run.finish()
        rank0_print(f"Done. step={step}  best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
