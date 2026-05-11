#!/usr/bin/env bash
set -euo pipefail

# L=32768 pilot on real combined splits.
# Waits for GPU idle, launches a short production-like run (20 optimizer steps),
# then verifies that all expected artefacts were produced.
#
# This is the gating step before the full multi-day production training run.
# See PROJECT_GUIDE.md §13 (⭐ NEXT) and FINETUNE_GUIDE.md §4.

usage() {
  cat <<'EOF'
Usage:
  scripts/queue_h100_pilot.sh [options]

Runs a short L=32768 pilot with production-like settings on real combined
splits. Validates: natural-length collation, validation pipeline, checkpoint
write path, logging artefacts, memory envelope, and wall-clock throughput.

Options:
  --max-steps N           Optimizer steps to run (default: 20).
  --val-every N           Validate every N steps (default: 10).
  --save-every N          Checkpoint every N steps (default: 10).
  --max-seq-len N         Sequence length (default: 32768).
  --batch-size N          Micro-batch size (default: 4).
  --grad-accum N          Gradient accumulation steps (default: 32).
  --check-every-sec N     Poll interval while waiting for idle GPU (default: 20).
  --idle-hold-sec N       Continuous idle time required before launch (default: 60).
  --min-free-mib N        Minimum free MiB required (default: 78000).
  --gpu-index N           GPU index to check (default: 0).
  --output-root PATH      Root output dir (default: /data2/ds85/bgcmodel_runs).
  --hf-home PATH          HF cache path (default: /data2/ds85/hf_cache).
  --env-name NAME         micromamba env name (default: bgcmodel).
  --train PATH            Train JSONL (default: data/processed/splits_combined/train.jsonl).
  --val PATH              Val JSONL (default: data/processed/splits_combined/val.jsonl).
  --wandb-mode MODE       WandB mode: online|offline|disabled (default: offline).
  --dry-run               Print commands without running training.
  -h, --help              Show this help.
EOF
}

# ── Defaults ──────────────────────────────────────────────────────────────
MAX_STEPS=20
VAL_EVERY=10
SAVE_EVERY=10
MAX_SEQ_LEN=32768
BATCH_SIZE=4
GRAD_ACCUM=32
CHECK_EVERY_SEC=20
IDLE_HOLD_SEC=60
MIN_FREE_MIB=78000
GPU_INDEX=0
OUTPUT_ROOT="/data2/ds85/bgcmodel_runs"
HF_HOME_PATH="/data2/ds85/hf_cache"
ENV_NAME="bgcmodel"
TRAIN_JSONL="data/processed/splits_combined/train.jsonl"
VAL_JSONL="data/processed/splits_combined/val.jsonl"
WANDB_MODE="offline"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-steps)        MAX_STEPS="${2:?missing value}"; shift 2 ;;
    --val-every)        VAL_EVERY="${2:?missing value}"; shift 2 ;;
    --save-every)       SAVE_EVERY="${2:?missing value}"; shift 2 ;;
    --max-seq-len)      MAX_SEQ_LEN="${2:?missing value}"; shift 2 ;;
    --batch-size)       BATCH_SIZE="${2:?missing value}"; shift 2 ;;
    --grad-accum)       GRAD_ACCUM="${2:?missing value}"; shift 2 ;;
    --check-every-sec)  CHECK_EVERY_SEC="${2:?missing value}"; shift 2 ;;
    --idle-hold-sec)    IDLE_HOLD_SEC="${2:?missing value}"; shift 2 ;;
    --min-free-mib)     MIN_FREE_MIB="${2:?missing value}"; shift 2 ;;
    --gpu-index)        GPU_INDEX="${2:?missing value}"; shift 2 ;;
    --output-root)      OUTPUT_ROOT="${2:?missing value}"; shift 2 ;;
    --hf-home)          HF_HOME_PATH="${2:?missing value}"; shift 2 ;;
    --env-name)         ENV_NAME="${2:?missing value}"; shift 2 ;;
    --train)            TRAIN_JSONL="${2:?missing value}"; shift 2 ;;
    --val)              VAL_JSONL="${2:?missing value}"; shift 2 ;;
    --wandb-mode)       WANDB_MODE="${2:?missing value}"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 2
  }
}

require_cmd nvidia-smi
require_cmd python
require_cmd micromamba
require_cmd deepspeed

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Train JSONL not found: $TRAIN_JSONL" >&2
  exit 2
fi
if [[ ! -f "$VAL_JSONL" ]]; then
  echo "Val JSONL not found: $VAL_JSONL" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$HF_HOME_PATH"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/pilot_L${MAX_SEQ_LEN}_${RUN_TS}"
mkdir -p "$RUN_DIR"
META_LOG="${RUN_DIR}/queue.log"

# ── Logging / GPU helpers ─────────────────────────────────────────────────

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*" | tee -a "$META_LOG"
}

get_proc_count() {
  local count
  count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk 'NF>0' | wc -l)"
  printf "%s" "${count// /}"
}

get_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | sed -n "$((GPU_INDEX + 1))p" | tr -d ' '
}

wait_for_gpu_idle() {
  local idle_start=0
  local now proc_count free_mib

  log "Waiting for GPU ${GPU_INDEX} idle: proc_count=0, free_mib>=${MIN_FREE_MIB}, hold=${IDLE_HOLD_SEC}s"
  while true; do
    proc_count="$(get_proc_count)"
    free_mib="$(get_free_mib)"
    now="$(date +%s)"

    if [[ -z "$free_mib" ]]; then
      log "Could not read free memory for GPU index ${GPU_INDEX}; retrying in ${CHECK_EVERY_SEC}s."
      idle_start=0
      sleep "$CHECK_EVERY_SEC"
      continue
    fi

    if [[ "$proc_count" -eq 0 && "$free_mib" -ge "$MIN_FREE_MIB" ]]; then
      if [[ "$idle_start" -eq 0 ]]; then
        idle_start="$now"
        log "GPU appears idle (free=${free_mib} MiB). Starting hold timer."
      fi
      if (( now - idle_start >= IDLE_HOLD_SEC )); then
        log "Idle window confirmed."
        return 0
      fi
    else
      idle_start=0
      log "GPU busy: proc_count=${proc_count}, free_mib=${free_mib}; recheck in ${CHECK_EVERY_SEC}s."
    fi

    sleep "$CHECK_EVERY_SEC"
  done
}

# ── Print run plan ────────────────────────────────────────────────────────

log "══════════════════════════════════════════════════════════════"
log "L=${MAX_SEQ_LEN} pilot run — production-like settings on real data"
log "  Output:     ${RUN_DIR}"
log "  Train data: ${TRAIN_JSONL}"
log "  Val data:   ${VAL_JSONL}"
log "  Steps:      ${MAX_STEPS}  (val-every=${VAL_EVERY}, save-every=${SAVE_EVERY})"
log "  Batch:      ${BATCH_SIZE} × grad-accum ${GRAD_ACCUM} = effective batch $(( BATCH_SIZE * GRAD_ACCUM ))"
log "  WandB:      ${WANDB_MODE}"
log "══════════════════════════════════════════════════════════════"

# ── Wait + launch ─────────────────────────────────────────────────────────

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN — command:"
  log "  HF_HOME=$HF_HOME_PATH deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \\"
  log "    --train $TRAIN_JSONL --val $VAL_JSONL \\"
  log "    --output-dir $RUN_DIR \\"
  log "    --max-seq-len $MAX_SEQ_LEN --batch-size $BATCH_SIZE --grad-accum $GRAD_ACCUM \\"
  log "    --max-steps $MAX_STEPS --val-every $VAL_EVERY --save-every $SAVE_EVERY \\"
  log "    --log-every 10 --wandb-mode $WANDB_MODE --wandb-project bcg-evo2-phase1"
  log "DRY RUN complete."
  exit 0
fi

wait_for_gpu_idle

STARTED_AT="$(date '+%F %T')"
log "Launching pilot run at ${STARTED_AT}"

set +e
HF_HOME="$HF_HOME_PATH" \
  micromamba run -n "$ENV_NAME" \
  deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
    --train "$TRAIN_JSONL" \
    --val "$VAL_JSONL" \
    --output-dir "$RUN_DIR" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" \
    --val-every "$VAL_EVERY" \
    --save-every "$SAVE_EVERY" \
    --log-every 10 \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project bcg-evo2-phase1 \
    2>&1 | tee "${RUN_DIR}/pilot.log"
TRAIN_RC="${PIPESTATUS[0]}"
set -e

FINISHED_AT="$(date '+%F %T')"

if [[ "$TRAIN_RC" -ne 0 ]]; then
  log "❌ Pilot FAILED with exit code ${TRAIN_RC}."
  log "   Started:  ${STARTED_AT}"
  log "   Finished: ${FINISHED_AT}"
  log "   See: ${RUN_DIR}/pilot.log"
  exit 1
fi

log "Training completed successfully."

# ── Post-run verification ─────────────────────────────────────────────────

log ""
log "══════════════════════════════════════════════════════════════"
log "POST-RUN VERIFICATION"
log "══════════════════════════════════════════════════════════════"

python - "$RUN_DIR" "$MAX_STEPS" "$VAL_EVERY" "$SAVE_EVERY" <<'PYVERIFY' 2>&1 | tee -a "$META_LOG"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
max_steps = int(sys.argv[2])
val_every = int(sys.argv[3])
save_every = int(sys.argv[4])

errors = []
warnings = []

# ── 1. train_log.jsonl ────────────────────────────────────────────────
train_log = run_dir / "train_log.jsonl"
if not train_log.exists():
    errors.append("train_log.jsonl missing")
else:
    rows = [json.loads(l) for l in train_log.read_text().splitlines() if l.strip()]
    print(f"\n📊 train_log.jsonl: {len(rows)} entries")
    if rows:
        last_step = rows[-1]["step"]
        print(f"   Steps logged: {rows[0]['step']} → {last_step}")
        if last_step < max_steps:
            errors.append(f"Only reached step {last_step}, expected {max_steps}")
        else:
            print(f"   ✅ Reached target step {max_steps}")

        # Memory envelope
        peaks = [max(r["gpu_mem_gb"]) for r in rows if "gpu_mem_gb" in r]
        if peaks:
            peak_mem = max(peaks)
            avg_mem = sum(peaks) / len(peaks)
            print(f"   GPU memory: peak={peak_mem:.2f} GB, avg={avg_mem:.2f} GB")
            if peak_mem > 75:
                warnings.append(f"Peak GPU memory {peak_mem:.2f} GB is high (>75 GB)")
            else:
                print(f"   ✅ Memory within envelope (peak {peak_mem:.2f} GB < 75 GB)")

        # Throughput
        tps = [r["tokens_per_sec"] for r in rows if "tokens_per_sec" in r and r["tokens_per_sec"] > 0]
        if tps:
            avg_tps = sum(tps) / len(tps)
            print(f"   Throughput: avg={avg_tps:.0f} tok/s")

        # Collated seq lengths (natural collation check)
        seq_lens = [r["collated_seq_len"] for r in rows if "collated_seq_len" in r]
        unique_lens = set(seq_lens)
        if len(unique_lens) > 1:
            print(f"   ✅ Natural collation: {len(unique_lens)} distinct collated lengths observed")
            print(f"      Range: {min(seq_lens)} → {max(seq_lens)}")
        elif len(unique_lens) == 1:
            warnings.append(f"Only one collated_seq_len seen ({unique_lens.pop()}); natural collation may not be exercised")

        # Loss trajectory
        losses = [r["train_loss"] for r in rows]
        print(f"   Loss: first={losses[0]:.4f}, last={losses[-1]:.4f}")
    else:
        errors.append("train_log.jsonl is empty")

# ── 2. val_log.jsonl ──────────────────────────────────────────────────
val_log = run_dir / "val_log.jsonl"
if not val_log.exists():
    errors.append("val_log.jsonl missing")
else:
    val_rows = [json.loads(l) for l in val_log.read_text().splitlines() if l.strip()]
    expected_val_count = max_steps // val_every
    print(f"\n📊 val_log.jsonl: {len(val_rows)} entries (expected ≥{expected_val_count})")
    if len(val_rows) >= expected_val_count:
        print(f"   ✅ Validation ran at expected cadence")
        for vr in val_rows:
            print(f"      step={vr.get('step','?')}  val_loss={vr.get('val_loss','?')}")
    else:
        errors.append(f"Only {len(val_rows)} val entries, expected ≥{expected_val_count}")

# ── 3. config.json ────────────────────────────────────────────────────
config_path = run_dir / "config.json"
if not config_path.exists():
    errors.append("config.json missing")
else:
    print(f"\n📊 config.json: present")
    cfg = json.loads(config_path.read_text())
    print(f"   ✅ Config saved with {len(cfg)} keys")

# ── 4. Checkpoints ────────────────────────────────────────────────────
ckpt_dir = run_dir / "checkpoints"
if not ckpt_dir.exists():
    errors.append("checkpoints/ directory missing")
else:
    ckpts = sorted([d.name for d in ckpt_dir.iterdir() if d.is_dir()])
    expected_ckpts = [f"step_{s}" for s in range(save_every, max_steps + 1, save_every)]
    print(f"\n📊 Checkpoints: {ckpts}")
    print(f"   Expected: {expected_ckpts}")
    for ec in expected_ckpts:
        ec_path = ckpt_dir / ec
        if ec_path.exists():
            # Check size (should be small — LoRA only)
            model_states = list(ec_path.glob("mp_rank_*_model_states.pt"))
            adapter_dir = ec_path / "adapter"
            if model_states:
                ms_mb = model_states[0].stat().st_size / (1024 * 1024)
                print(f"   ✅ {ec}: model_states={ms_mb:.1f} MB", end="")
                if ms_mb > 500:
                    errors.append(f"{ec} model_states={ms_mb:.1f} MB — frozen params may be included")
                    print(" ⚠️")
                else:
                    print(" (frozen params excluded)")
            if adapter_dir.exists():
                print(f"      adapter/ present ✅")
            else:
                errors.append(f"{ec}/adapter/ directory missing")
        else:
            errors.append(f"Expected checkpoint {ec} not found")

# ── 5. Final adapter ──────────────────────────────────────────────────
final_adapter = run_dir / "final_adapter"
if final_adapter.exists():
    print(f"\n📊 final_adapter/: present ✅")
else:
    warnings.append("final_adapter/ not present (expected only if run completed all epochs)")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print("❌ PILOT VERIFICATION FAILED:")
    for e in errors:
        print(f"   ❌ {e}")
else:
    print("✅ ALL PILOT CHECKS PASSED")

if warnings:
    print("\n⚠️  Warnings:")
    for w in warnings:
        print(f"   ⚠️  {w}")

print()
sys.exit(1 if errors else 0)
PYVERIFY

VERIFY_RC=$?

log ""
log "Started:  ${STARTED_AT}"
log "Finished: ${FINISHED_AT}"
log "Output:   ${RUN_DIR}"

if [[ "$VERIFY_RC" -eq 0 ]]; then
  log "✅ Pilot PASSED. Ready for full production run."
else
  log "❌ Pilot verification found issues. Review output above."
fi

exit "$VERIFY_RC"
