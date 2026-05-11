#!/usr/bin/env bash
set -euo pipefail

# Resume-from-checkpoint verification test.
# Waits for GPU idle, then:
#   Phase 1: Train 5 steps with --save-every 2  (saves at steps 2, 4; final at 5)
#   Phase 2: Resume from step_2 checkpoint, train to step 5
#   Verify:  Compare loss/lr/step continuity between phase 1 and phase 2

CHECK_EVERY_SEC=20
IDLE_HOLD_SEC=60
MIN_FREE_MIB=78000
GPU_INDEX=0
OUTPUT_ROOT="/data2/ds85/bgcmodel_runs"
HF_HOME_PATH="/data2/ds85/hf_cache"
ENV_NAME="bgcmodel"
TRAIN_JSONL="data/processed/splits_combined/val.jsonl"
VAL_JSONL="data/processed/splits_combined/val.jsonl"
SEQ_LEN=1024
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  scripts/queue_h100_resume_test.sh [options]

Verifies that checkpoint save/resume works correctly with
exclude_frozen_parameters=True. Runs two short training phases at L=1024
and compares step counter, learning rate, and loss continuity.

Options:
  --check-every-sec N   Poll interval while waiting for idle GPU (default: 20).
  --idle-hold-sec N     Continuous idle time required before launch (default: 60).
  --min-free-mib N      Minimum free MiB required (default: 78000).
  --gpu-index N         GPU index to check (default: 0).
  --output-root PATH    Root output dir (default: /data2/ds85/bgcmodel_runs).
  --hf-home PATH        HF cache path (default: /data2/ds85/hf_cache).
  --env-name NAME       micromamba env name (default: bgcmodel).
  --max-seq-len N       Sequence length (default: 1024). Keep short for speed.
  --dry-run             Print commands without running.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-every-sec)  CHECK_EVERY_SEC="${2:?missing value}"; shift 2 ;;
    --idle-hold-sec)    IDLE_HOLD_SEC="${2:?missing value}"; shift 2 ;;
    --min-free-mib)     MIN_FREE_MIB="${2:?missing value}"; shift 2 ;;
    --gpu-index)        GPU_INDEX="${2:?missing value}"; shift 2 ;;
    --output-root)      OUTPUT_ROOT="${2:?missing value}"; shift 2 ;;
    --hf-home)          HF_HOME_PATH="${2:?missing value}"; shift 2 ;;
    --env-name)         ENV_NAME="${2:?missing value}"; shift 2 ;;
    --max-seq-len)      SEQ_LEN="${2:?missing value}"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

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
RUN_ROOT="${OUTPUT_ROOT}/resume_test_${RUN_TS}"
mkdir -p "$RUN_ROOT"
META_LOG="${RUN_ROOT}/queue.log"

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

PHASE1_DIR="${RUN_ROOT}/phase1_initial"
PHASE2_DIR="${RUN_ROOT}/phase2_resumed"

COMMON_ARGS=(
  --train "$TRAIN_JSONL"
  --val "$VAL_JSONL"
  --max-seq-len "$SEQ_LEN"
  --batch-size 1
  --grad-accum 1
  --warmup-steps 2
  --max-epochs 1
  --log-every 1
  --val-every 99
  --wandb-mode offline
)

# ── Phase 1: Train 5 steps, saving every 2 ─────────────────────────────

log "══════════════════════════════════════════════════════════════"
log "Resume-from-checkpoint verification test"
log "  Phase 1: 5 steps with --save-every 2  →  ${PHASE1_DIR}"
log "  Phase 2: resume from step_2, run to step 5  →  ${PHASE2_DIR}"
log "  Sequence length: ${SEQ_LEN}"
log "══════════════════════════════════════════════════════════════"

wait_for_gpu_idle
log "Phase 1: Starting initial 5-step run with --save-every 2"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN — Phase 1 command:"
  log "  deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py ${COMMON_ARGS[*]} --output-dir $PHASE1_DIR --max-steps 5 --save-every 2"
else
  set +e
  HF_HOME="$HF_HOME_PATH" \
    micromamba run -n "$ENV_NAME" \
    deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
      "${COMMON_ARGS[@]}" \
      --output-dir "$PHASE1_DIR" \
      --max-steps 5 \
      --save-every 2 \
      2>&1 | tee "${RUN_ROOT}/phase1.log"
  rc1="${PIPESTATUS[0]}"
  set -e
  if [[ "$rc1" -ne 0 ]]; then
    log "Phase 1 FAILED with exit code ${rc1}. Aborting."
    exit 1
  fi
  log "Phase 1 completed successfully."
fi

# Verify checkpoint at step_2 exists
RESUME_CKPT="${PHASE1_DIR}/checkpoints/step_2"
if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ ! -d "$RESUME_CKPT" ]]; then
    log "ERROR: Expected checkpoint directory not found: ${RESUME_CKPT}"
    log "Contents of checkpoints/:"
    ls -la "${PHASE1_DIR}/checkpoints/" 2>&1 | tee -a "$META_LOG"
    exit 1
  fi
  log "Checkpoint verified: ${RESUME_CKPT}"
  log "  Contents:"
  ls -la "$RESUME_CKPT" 2>&1 | tee -a "$META_LOG"
  ls -la "${RESUME_CKPT}/adapter/" 2>&1 | tee -a "$META_LOG" || true
fi

# ── Phase 2: Resume from step_2, run to step 5 ─────────────────────────

wait_for_gpu_idle
log "Phase 2: Resuming from ${RESUME_CKPT}, running to step 5"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN — Phase 2 command:"
  log "  deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py ${COMMON_ARGS[*]} --output-dir $PHASE2_DIR --max-steps 5 --save-every 99 --resume-from $RESUME_CKPT"
else
  set +e
  HF_HOME="$HF_HOME_PATH" \
    micromamba run -n "$ENV_NAME" \
    deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
      "${COMMON_ARGS[@]}" \
      --output-dir "$PHASE2_DIR" \
      --max-steps 5 \
      --save-every 99 \
      --resume-from "$RESUME_CKPT" \
      2>&1 | tee "${RUN_ROOT}/phase2.log"
  rc2="${PIPESTATUS[0]}"
  set -e
  if [[ "$rc2" -ne 0 ]]; then
    log "Phase 2 FAILED with exit code ${rc2}."
    exit 1
  fi
  log "Phase 2 completed successfully."
fi

# ── Verification ────────────────────────────────────────────────────────

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN complete. No verification to perform."
  exit 0
fi

log ""
log "══════════════════════════════════════════════════════════════"
log "VERIFICATION"
log "══════════════════════════════════════════════════════════════"

python - "$PHASE1_DIR" "$PHASE2_DIR" <<'PYVERIFY' 2>&1 | tee -a "$META_LOG"
import json
import sys
from pathlib import Path

phase1_dir = Path(sys.argv[1])
phase2_dir = Path(sys.argv[2])

def load_log(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

p1_log = load_log(phase1_dir / "train_log.jsonl")
p2_log = load_log(phase2_dir / "train_log.jsonl")

print(f"\n--- Phase 1 train_log.jsonl ({len(p1_log)} entries) ---")
for r in p1_log:
    print(f"  step={r['step']:>3d}  loss={r['train_loss']:.6f}  lr={r['lr']:.2e}  "
          f"grad_norm={r.get('grad_norm', 'N/A')}")

print(f"\n--- Phase 2 train_log.jsonl ({len(p2_log)} entries) ---")
for r in p2_log:
    print(f"  step={r['step']:>3d}  loss={r['train_loss']:.6f}  lr={r['lr']:.2e}  "
          f"grad_norm={r.get('grad_norm', 'N/A')}")

# Check 1: Phase 2 starts at step 3 (resumed from step 2)
errors = []
if p2_log:
    first_resumed_step = p2_log[0]["step"]
    if first_resumed_step != 3:
        errors.append(f"FAIL: Phase 2 first step = {first_resumed_step}, expected 3")
    else:
        print(f"\n✅ Step counter: Phase 2 starts at step {first_resumed_step} (correct)")
else:
    errors.append("FAIL: Phase 2 train_log.jsonl is empty")

# Check 2: Phase 2 ends at step 5
if p2_log:
    last_step = p2_log[-1]["step"]
    if last_step != 5:
        errors.append(f"FAIL: Phase 2 last step = {last_step}, expected 5")
    else:
        print(f"✅ Step counter: Phase 2 ends at step {last_step} (correct)")

# Check 3: LR continuity — phase 2 step 3 lr should match phase 1 step 3 lr
p1_step3 = [r for r in p1_log if r["step"] == 3]
p2_step3 = [r for r in p2_log if r["step"] == 3]
if p1_step3 and p2_step3:
    lr1 = p1_step3[0]["lr"]
    lr2 = p2_step3[0]["lr"]
    if abs(lr1 - lr2) < 1e-10:
        print(f"✅ LR continuity: step 3 lr = {lr1:.2e} in both runs")
    else:
        errors.append(f"FAIL: LR mismatch at step 3: phase1={lr1:.2e}, phase2={lr2:.2e}")
else:
    errors.append(f"WARN: Could not compare step 3 LR (p1 has {len(p1_step3)}, p2 has {len(p2_step3)} entries)")

# Check 4: Loss at step 3 should be similar (not identical due to data order,
# but wildly different would indicate broken resume)
if p1_step3 and p2_step3:
    loss1 = p1_step3[0]["train_loss"]
    loss2 = p2_step3[0]["train_loss"]
    diff = abs(loss1 - loss2)
    if diff < 0.01:
        print(f"✅ Loss continuity: step 3 loss phase1={loss1:.6f}, phase2={loss2:.6f} (diff={diff:.6f})")
    elif diff < 0.1:
        print(f"⚠️  Loss slightly different at step 3: phase1={loss1:.6f}, phase2={loss2:.6f} (diff={diff:.6f})")
        print(f"   This may be expected if data sampling order differs after resume.")
    else:
        errors.append(f"FAIL: Loss very different at step 3: phase1={loss1:.6f}, phase2={loss2:.6f} (diff={diff:.6f})")

# Check 5: Verify checkpoint files are small (exclude_frozen_parameters working)
ckpt_dir = phase1_dir / "checkpoints" / "step_2"
model_states = list(ckpt_dir.glob("mp_rank_*_model_states.pt"))
optim_states = list(ckpt_dir.glob("*_optim_states.pt"))
if model_states:
    ms_size_mb = model_states[0].stat().st_size / (1024 * 1024)
    if ms_size_mb < 500:
        print(f"✅ Checkpoint size: model_states = {ms_size_mb:.1f} MB (frozen params excluded)")
    else:
        errors.append(f"FAIL: model_states = {ms_size_mb:.1f} MB — frozen params may be included (expected < 500 MB)")
if optim_states:
    os_size_mb = optim_states[0].stat().st_size / (1024 * 1024)
    print(f"   Optimizer states = {os_size_mb:.1f} MB")

print()
if errors:
    print("❌ RESUME TEST FAILED:")
    for e in errors:
        print(f"   {e}")
    sys.exit(1)
else:
    print("✅ ALL CHECKS PASSED — resume-from-checkpoint is working correctly.")
    sys.exit(0)
PYVERIFY

VERIFY_RC=$?
if [[ "$VERIFY_RC" -eq 0 ]]; then
  log ""
  log "✅ Resume test PASSED. Results: ${RUN_ROOT}"
else
  log ""
  log "❌ Resume test FAILED. Check logs: ${RUN_ROOT}"
fi

exit "$VERIFY_RC"
