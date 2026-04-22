#!/usr/bin/env python3
"""Fine-tune Evo2 7B on the combined MIBiG + antiSMASH BGC training dataset.

Phase 1 conditioning: COMPOUND_CLASS + taxonomic tag, no COMPOUND token.

Launch
------
    conda activate bgcmodel
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 scripts/finetune_evo2.py \\
        --train         data/processed/splits_combined/train.jsonl \\
        --val           data/processed/splits_combined/val.jsonl \\
        --output-dir    checkpoints/phase1_classonly \\
        --wandb-project bcg-evo2-phase1

Full documentation: FINETUNE_GUIDE.md in the repo root.

Project design notes this script honours:
  - Phase 1 is class-only. The COMPOUND token has already been stripped from
    the combined JSONL during the merge step; training_text is used verbatim.
  - Goal is transposition, not invention (see FINETUNE_GUIDE.md §10).
  - Every run is reproducible: config + data fingerprint + pip freeze are
    saved before training starts. Resume from any checkpoint without losing
    a single step.
  - Three parallel visualisation paths (WandB, JSONL logs, PNG plots) so
    training progress is visible even if WandB is unavailable.
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
CHAR_VOCAB_SIZE = 512   # CharLevelTokenizer vocab
PAD_TOKEN_ID    = 0     # 0 is a safe padding id for CharLevelTokenizer
IGNORE_INDEX    = -100  # cross-entropy ignore

# Default hyperparameters — see FINETUNE_GUIDE.md §4 for rationale
DEFAULTS = dict(
    max_seq_len       = 32768,
    batch_size        = 4,
    grad_accum        = 8,
    lr                = 1e-5,
    lr_min_ratio      = 0.1,     # final LR = 0.1 * peak
    warmup_steps      = 200,
    max_epochs        = 2,
    weight_decay      = 0.1,
    grad_clip         = 1.0,
    beta1             = 0.9,
    beta2             = 0.95,
    seed              = 42,
    log_every         = 10,
    val_every         = 250,
    save_every        = 500,
    val_max_batches   = 500,  # cap val-loss estimation cost (~2000 seqs at bs=4)
    keep_last_ckpts   = 5,
)

# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    # Required
    p.add_argument("--train", type=Path, required=True, help="Train JSONL")
    p.add_argument("--val",   type=Path, required=True, help="Validation JSONL")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints, logs, plots")
    # Run naming
    p.add_argument("--run-name", type=str, default=None,
                   help="WandB run name; defaults to output_dir basename + timestamp")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="WandB project name; omit to disable WandB logging")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"])
    # Hyperparameters
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    # Smoke-test / early-exit cap (0 = disabled)
    p.add_argument("--max-steps", type=int, default=0,
                   help="If >0, stop after this many optimizer steps (smoke tests)")
    # Resume
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Checkpoint directory to resume from")
    # Local DeepSpeed rank (set automatically by launcher)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ────────────────────────────────────────────────────────────────────────

def ddp_rank() -> int:
    return int(os.environ.get("RANK", 0))

def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def is_main() -> bool:
    return ddp_rank() == 0


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
    # Slight speed cost but worth it for reproducibility
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
    """Write all hyperparameters + environment metadata to config.json."""
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
    }
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def save_data_fingerprint(args: argparse.Namespace, output_dir: Path) -> None:
    """SHA256 of first 100 lines + total line count for each split."""
    def fp(path: Path) -> dict:
        n_lines = 0
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for i, line in enumerate(f):
                n_lines += 1
                if i < 100:
                    hasher.update(line)
        return {"path": str(path), "lines": n_lines,
                "sha256_first_100": hasher.hexdigest()}

    fingerprint = {
        "train": fp(args.train),
        "val":   fp(args.val),
    }
    (output_dir / "data_fingerprint.json").write_text(json.dumps(fingerprint, indent=2))


def save_env(output_dir: Path) -> None:
    try:
        pf = subprocess.check_output(["pip", "freeze"], stderr=subprocess.DEVNULL).decode()
        (output_dir / "env.txt").write_text(pf)
    except Exception as e:
        (output_dir / "env.txt").write_text(f"pip freeze failed: {e}\n")


# ────────────────────────────────────────────────────────────────────────
# Dataset — streams JSONL line offsets (low RAM for 18 GB train file)
# ────────────────────────────────────────────────────────────────────────

class BGCTextDataset(Dataset):
    """Memory-efficient JSONL dataset yielding tokenised training_text.

    Scans the file once at init to record byte offsets of each record, so
    we can seek/read individual records on-demand without loading the full
    18 GB into memory.
    """

    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int) -> None:
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.offsets: list[int] = []

        # Build line-offset index (~seconds for 18 GB)
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
        # Re-open per call to avoid leaking file handles across workers
        with self.jsonl_path.open("rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        rec = json.loads(line)
        text: str = rec["training_text"]

        # Sanitise: retain |, :, ;, _, +, -, 0-9, A-Z; drop everything else.
        # CharLevelTokenizer encodes via ord(); any byte value in [0, 255]
        # is valid but we standardise on uppercase ACGTN in DNA.
        # The training_text is already well-formed from our pipeline.

        # Tokenise to list[int]
        tokens = self.tokenizer.tokenize(text)
        if isinstance(tokens, (list, tuple)):
            ids = list(tokens)
        else:
            # Fallback for tokenisers returning a tensor-like
            ids = [int(t) for t in tokens]

        # Truncate from the end of the sequence — preserves the prefix
        # (which contains the class + taxonomy conditioning).
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]

        return {
            "input_ids": ids,
            "length":    len(ids),
            "accession": rec.get("accession", ""),
            "class":     rec.get("compound_class", ""),
        }


def collate_pad(batch: list[dict[str, Any]], pad_id: int = PAD_TOKEN_ID
                ) -> dict[str, torch.Tensor]:
    """Pad to max length within this batch; mask padding positions in labels."""
    max_len = max(b["length"] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["length"]
        ids = torch.tensor(b["input_ids"], dtype=torch.long)
        input_ids[i, :L] = ids
        labels[i, :L]    = ids
    return {"input_ids": input_ids, "labels": labels}


# ────────────────────────────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────────────────────────────

def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token prediction loss. logits: (B, T, V), labels: (B, T)."""
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
def run_validation(engine, val_loader, local_rank: int,
                   max_batches: int) -> float:
    """Compute average CE loss over up to max_batches from val_loader."""
    engine.eval()
    total_loss = 0.0
    n_tokens = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(local_rank, non_blocking=True)
        labels    = batch["labels"].to(local_rank, non_blocking=True)

        logits, _ = engine(input_ids)
        loss = causal_lm_loss(logits, labels)

        # Weight loss by number of non-ignored target tokens this batch
        n = (labels[..., 1:] != IGNORE_INDEX).sum().item()
        total_loss += loss.item() * n
        n_tokens += n

    # Reduce across ranks
    if torch.distributed.is_initialized():
        t = torch.tensor([total_loss, n_tokens], dtype=torch.float64,
                         device=f"cuda:{local_rank}")
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        total_loss, n_tokens = t.tolist()

    engine.train()
    return total_loss / max(n_tokens, 1)


# ────────────────────────────────────────────────────────────────────────
# DeepSpeed config
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
            "stage":                   2,
            "allgather_partitions":    True,
            "reduce_scatter":          True,
            "overlap_comm":            True,
            "contiguous_gradients":    True,
            "allgather_bucket_size":   5e8,
            "reduce_bucket_size":      5e8,
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
            # DS WarmupCosineLR signature (v0.18+):
            #   total_num_steps, warmup_num_steps, warmup_min_ratio (of peak
            #   LR at warmup start), cos_min_ratio (floor as ratio of peak),
            #   warmup_type. Peak LR is taken from the optimizer.
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps":   total_steps,
                "warmup_num_steps":  args.warmup_steps,
                "warmup_min_ratio":  0.0,
                "cos_min_ratio":     args.lr_min_ratio,
                "warmup_type":       "linear",
            },
        },
        "steps_per_print": max(args.log_every, 1),
        "wall_clock_breakdown": False,
    }


# ────────────────────────────────────────────────────────────────────────
# Plot generation (runs at every checkpoint, rank 0 only)
# ────────────────────────────────────────────────────────────────────────

def generate_plots(output_dir: Path) -> None:
    """Read train_log.jsonl + val_log.jsonl and write PNG plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib optional

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

    # Panel 1: train + val loss vs step
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([e["step"] for e in train], [e["train_loss"] for e in train],
            label="train", alpha=0.8, linewidth=1)
    if val:
        ax.plot([e["step"] for e in val], [e["val_loss"] for e in val],
                label="val", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training / Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss.png", dpi=100)
    plt.close(fig)

    # Panel 2: learning rate schedule (sanity check)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], [e["lr"] for e in train], color="C2")
    ax.set_xlabel("step"); ax.set_ylabel("learning rate")
    ax.set_title("LR Schedule"); ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(plots_dir / "lr.png", dpi=100)
    plt.close(fig)

    # Panel 3: gradient norm
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot([e["step"] for e in train], [e["grad_norm"] for e in train],
            color="C3", alpha=0.7, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("gradient norm")
    ax.set_title("Gradient Norm"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "grad_norm.png", dpi=100)
    plt.close(fig)

    # Panel 4: throughput
    fig, ax = plt.subplots(figsize=(10, 3.5))
    tps = [e.get("tokens_per_sec", 0) for e in train]
    ax.plot([e["step"] for e in train], tps, color="C4", alpha=0.7, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("tokens / sec (all GPUs)")
    ax.set_title("Throughput"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "throughput.png", dpi=100)
    plt.close(fig)

    # Combined summary panel
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax = axes[0, 0]
    ax.plot([e["step"] for e in train], [e["train_loss"] for e in train],
            label="train", alpha=0.6, linewidth=1)
    if val:
        ax.plot([e["step"] for e in val], [e["val_loss"] for e in val],
                label="val", linewidth=2, marker="o", markersize=4)
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("step")

    ax = axes[0, 1]
    ax.plot([e["step"] for e in train], [e["lr"] for e in train], color="C2")
    ax.set_title("Learning rate (log)"); ax.set_yscale("log"); ax.grid(alpha=0.3)
    ax.set_xlabel("step")

    ax = axes[1, 0]
    ax.plot([e["step"] for e in train], [e["grad_norm"] for e in train],
            color="C3", alpha=0.6, linewidth=1)
    ax.set_title("Gradient norm"); ax.grid(alpha=0.3); ax.set_xlabel("step")

    ax = axes[1, 1]
    ax.plot([e["step"] for e in train], tps, color="C4", alpha=0.6, linewidth=1)
    ax.set_title("Throughput (tokens/sec)"); ax.grid(alpha=0.3); ax.set_xlabel("step")

    fig.suptitle(f"Training progress — {output_dir.name}", y=0.995)
    fig.tight_layout()
    fig.savefig(plots_dir / "summary.png", dpi=110)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def gpu_memory_gb() -> list[float]:
    """Peak allocated memory per visible GPU in GB."""
    return [torch.cuda.max_memory_allocated(i) / 1e9
            for i in range(torch.cuda.device_count())]


# ────────────────────────────────────────────────────────────────────────
# Checkpoint retention
# ────────────────────────────────────────────────────────────────────────

def cleanup_old_checkpoints(ckpt_root: Path, keep_last: int) -> None:
    """Remove old step_* directories, keeping the newest `keep_last`
    plus any tagged 'best/'. Safe to call from rank 0 only."""
    if not ckpt_root.exists():
        return
    step_dirs = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if len(step_dirs) <= keep_last:
        return
    for stale in step_dirs[:-keep_last]:
        try:
            subprocess.run(["rm", "-rf", str(stale)], check=False)
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────
# Graceful shutdown — save a final checkpoint on SIGTERM / SIGINT
# ────────────────────────────────────────────────────────────────────────

_SHOULD_STOP = False

def _set_stop_flag(signum, frame):
    global _SHOULD_STOP
    _SHOULD_STOP = True
    rank0_print(f"Signal {signum} received; will checkpoint and exit cleanly after current step")


# ────────────────────────────────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Per-rank GPU masking ─────────────────────────────────────────
    # Evo2's StripedHyena loader auto-shards layers across every visible
    # CUDA device (pipeline-style). That collides with ZeRO-2, which
    # wants one full model replica per rank. Fix: restrict each worker
    # to its own local GPU *before* any CUDA init, so Evo2 only sees
    # one device and places the whole model on it.
    local_rank_env = ddp_local_rank()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank_env)
    # After masking, the only visible device is index 0 from torch's POV.
    # DeepSpeed's init_distributed reads LOCAL_RANK to build device_id=
    # cuda:LOCAL_RANK, so we must also rewrite LOCAL_RANK to 0. RANK and
    # WORLD_SIZE stay as set by the launcher for proper rank coordination.
    os.environ["LOCAL_RANK"] = "0"
    local_rank = 0
    # DeepSpeed's DeepSpeedEngine._do_args_sanity_check compares
    # args.local_rank (set by launcher CLI) against env['LOCAL_RANK']
    # and asserts equality. Keep them in sync.
    args.local_rank = 0

    seed_everything(args.seed + ddp_rank())

    # Import heavy deps *after* argparse so `--help` is fast
    import deepspeed
    import evo2

    # Pin to the only visible device (index 0 after masking) BEFORE
    # init_distributed so NCCL/eager-init uses the correct ordinal.
    # Without this, torch.distributed tries to eager-connect to
    # device_id=LOCAL_RANK (e.g. 3) which no longer exists in the
    # masked process and fails with "invalid device ordinal".
    torch.cuda.set_device(local_rank)
    # Distributed init (uses NCCL over env://, respects LOCAL_RANK/RANK)
    deepspeed.init_distributed()

    # Run name / output setup
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

    # WandB init (rank 0 only)
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
            )
            rank0_print(f"WandB:      {wandb_run.url if wandb_run else 'disabled'}")
        except Exception as e:
            rank0_print(f"WandB init failed ({e}); continuing with local logging only")

    # Signal handlers — save checkpoint on SIGTERM/SIGINT
    signal.signal(signal.SIGTERM, _set_stop_flag)
    signal.signal(signal.SIGINT,  _set_stop_flag)

    # ── Load Evo2 ─────────────────────────────────────────────────────
    rank0_print("Loading Evo2 7B model + tokenizer...")
    t0 = time.time()
    evo_wrapper = evo2.Evo2(EVO2_MODEL_NAME)
    model = evo_wrapper.model
    tokenizer = evo_wrapper.tokenizer
    rank0_print(f"  Loaded in {time.time()-t0:.0f} s")
    rank0_print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Evo2 applies post-load `.permute()` to Wqkv blocks, leaving those
    # tensors non-contiguous. DeepSpeed's _broadcast_model then fails
    # with "Tensors must be contiguous". Force-contiguify in place.
    n_fixed = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()
                n_fixed += 1
        for b in model.buffers():
            if not b.is_contiguous():
                b.data = b.data.contiguous()
                n_fixed += 1
    rank0_print(f"  Made {n_fixed} tensors contiguous")

    # ── Datasets ─────────────────────────────────────────────────────
    rank0_print("Indexing datasets...")
    train_ds = BGCTextDataset(args.train, tokenizer, args.max_seq_len)
    val_ds   = BGCTextDataset(args.val,   tokenizer, args.max_seq_len)

    world_size = ddp_world_size()
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
    total_steps = steps_per_epoch * args.max_epochs
    rank0_print(f"Train batches/rank: {len(train_loader):,}")
    rank0_print(f"Steps per epoch:    {steps_per_epoch:,}  (after grad_accum={args.grad_accum})")
    rank0_print(f"Total steps:        {total_steps:,}  (max_epochs={args.max_epochs})")

    # ── DeepSpeed init ───────────────────────────────────────────────
    ds_config = build_ds_config(args, world_size, total_steps)
    if is_main():
        (args.output_dir / "deepspeed_config.json").write_text(json.dumps(ds_config, indent=2))

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
    )

    # ── Resume? ──────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    if args.resume_from is not None and args.resume_from.exists():
        rank0_print(f"Resuming from {args.resume_from}")
        _, client_state = model_engine.load_checkpoint(str(args.resume_from.parent),
                                                       tag=args.resume_from.name)
        if client_state:
            start_step    = int(client_state.get("step", 0))
            best_val_loss = float(client_state.get("best_val_loss", float("inf")))
            rank0_print(f"  Resumed at step {start_step}  best_val_loss={best_val_loss:.4f}")

    # ── Training loop ───────────────────────────────────────────────
    rank0_print(f"Starting training from step {start_step}")
    model_engine.train()

    step = start_step
    log_buffer_loss = 0.0
    log_buffer_count = 0
    last_log_time = time.time()
    last_log_tokens = 0

    train_log_path = args.output_dir / "train_log.jsonl"
    val_log_path   = args.output_dir / "val_log.jsonl"

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

                log_buffer_loss += loss.item()
                log_buffer_count += 1
                last_log_tokens += input_ids.numel() * world_size

                # Increment step only on optimizer step. Use DS's authoritative
                # `global_steps` counter (incremented inside `_take_model_step`
                # after optimizer.step() + scheduler.step()).
                # NOTE: `is_gradient_accumulation_boundary()` called AFTER engine.step()
                # is off-by-one at grad_accum > 1 — it returns True one micro-batch
                # before the optimizer actually steps. At ga=1 both coincide.
                if model_engine.global_steps != step:
                    step = model_engine.global_steps

                    # ── Logging (per log_every optimizer steps)
                    if step % args.log_every == 0:
                        avg_loss = log_buffer_loss / max(log_buffer_count, 1)
                        lr = scheduler.get_last_lr()[0] if scheduler else args.lr
                        grad_norm = float(model_engine.get_global_grad_norm() or 0.0)
                        elapsed = time.time() - last_log_time
                        tps = last_log_tokens / max(elapsed, 1e-3)

                        if is_main():
                            entry = {
                                "step":            step,
                                "epoch":           round(step / max(steps_per_epoch,1), 4),
                                "train_loss":      round(avg_loss, 6),
                                "lr":              lr,
                                "grad_norm":       round(grad_norm, 4),
                                "gpu_mem_gb":      [round(x, 2) for x in gpu_memory_gb()],
                                "tokens_per_sec":  int(tps),
                                "elapsed_sec":     int(time.time() - t0),
                            }
                            append_jsonl(train_log_path, entry)
                            if wandb_run:
                                wandb_run.log({k: v for k, v in entry.items()
                                              if not isinstance(v, list)}, step=step)
                            if step % (args.log_every * 10) == 0:
                                rank0_print(
                                    f"step {step:5d} | ep {entry['epoch']:.2f} | "
                                    f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                                    f"gn {grad_norm:.3f} | {int(tps):,} tok/s"
                                )

                        log_buffer_loss = 0.0
                        log_buffer_count = 0
                        last_log_time = time.time()
                        last_log_tokens = 0

                    # ── Validation (per val_every steps)
                    if step % args.val_every == 0:
                        val_loss = run_validation(model_engine, val_loader,
                                                  local_rank, args.val_max_batches)
                        if is_main():
                            ventry = {
                                "step":      step,
                                "val_loss":  round(val_loss, 6),
                                "val_ppl":   round(math.exp(min(val_loss, 20)), 4),
                            }
                            append_jsonl(val_log_path, ventry)
                            if wandb_run:
                                wandb_run.log(ventry, step=step)
                            rank0_print(
                                f"  VAL @ step {step}: loss {val_loss:.4f}  "
                                f"ppl {math.exp(min(val_loss,20)):.3f}  "
                                f"(best so far: {best_val_loss:.4f})"
                            )
                            # Save best checkpoint
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                rank0_print(f"  NEW best val_loss; saving best/ checkpoint")
                        # Broadcast best so all ranks save if needed
                        if torch.distributed.is_initialized():
                            t = torch.tensor([best_val_loss, val_loss],
                                             dtype=torch.float64, device=f"cuda:{local_rank}")
                            torch.distributed.broadcast(t, 0)
                            best_val_loss = t[0].item()
                            current_val = t[1].item()
                        else:
                            current_val = val_loss

                        if current_val <= best_val_loss + 1e-9:
                            model_engine.save_checkpoint(
                                str(args.output_dir / "checkpoints"),
                                tag="best",
                                client_state={"step": step, "best_val_loss": best_val_loss},
                            )

                    # ── Checkpoint (per save_every steps)
                    if step % args.save_every == 0:
                        if is_main():
                            rank0_print(f"  Saving checkpoint at step {step}...")
                        model_engine.save_checkpoint(
                            str(args.output_dir / "checkpoints"),
                            tag=f"step_{step}",
                            client_state={"step": step, "best_val_loss": best_val_loss},
                        )
                        if is_main():
                            cleanup_old_checkpoints(
                                args.output_dir / "checkpoints",
                                keep_last=args.keep_last_ckpts,
                            )
                            generate_plots(args.output_dir)
                            rank0_print(f"  Checkpoint saved; plots refreshed")

                    # Smoke-test early exit
                    if args.max_steps and step >= args.max_steps:
                        rank0_print(f"Reached --max-steps={args.max_steps}; exiting")
                        if is_main():
                            generate_plots(args.output_dir)
                        return

                    # Graceful shutdown
                    if _SHOULD_STOP:
                        rank0_print("Graceful shutdown requested; saving final checkpoint")
                        model_engine.save_checkpoint(
                            str(args.output_dir / "checkpoints"),
                            tag=f"step_{step}_interrupted",
                            client_state={"step": step, "best_val_loss": best_val_loss},
                        )
                        if is_main():
                            generate_plots(args.output_dir)
                        return

                    if step >= total_steps:
                        break
            if step >= total_steps:
                break

    except torch.cuda.OutOfMemoryError as e:
        rank0_print(f"CUDA OOM: {e}")
        rank0_print("Attempting to save emergency checkpoint before exit...")
        try:
            model_engine.save_checkpoint(
                str(args.output_dir / "checkpoints"),
                tag=f"step_{step}_oom",
                client_state={"step": step, "best_val_loss": best_val_loss},
            )
        except Exception as ee:
            rank0_print(f"Emergency checkpoint failed: {ee}")
        raise

    finally:
        # Final checkpoint + plots
        if step > start_step:
            rank0_print(f"Saving final checkpoint at step {step}")
            try:
                model_engine.save_checkpoint(
                    str(args.output_dir / "checkpoints"),
                    tag=f"step_{step}_final",
                    client_state={"step": step, "best_val_loss": best_val_loss},
                )
            except Exception as e:
                rank0_print(f"Final checkpoint failed: {e}")
            if is_main():
                generate_plots(args.output_dir)

        if wandb_run:
            wandb_run.finish()
        rank0_print(f"Training complete. Final step: {step}  best_val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
