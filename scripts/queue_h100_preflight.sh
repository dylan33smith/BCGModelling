#!/usr/bin/env bash
set -euo pipefail

# Queue and run production-like preflight checks when GPU 0 is idle.
# Intended to map practical max_seq_len under launch settings closer to
# production than the short smoke matrix (e.g., grad_accum=32).

usage() {
  cat <<'EOF'
Usage:
  scripts/queue_h100_preflight.sh [options]

Options:
  --lengths "36864 40960 45056 49152 53248 57344 61440"
                                          Space-separated sequence lengths.
  --check-every-sec N                     Poll interval while waiting for idle GPU (default: 20).
  --idle-hold-sec N                       Continuous idle time required before launch (default: 60).
  --min-free-mib N                        Minimum free MiB required (default: 78000).
  --gpu-index N                           GPU index to check (default: 0).
  --output-root PATH                      Root output dir (default: /data2/ds85/bgcmodel_runs).
  --hf-home PATH                          HF cache path (default: /data2/ds85/hf_cache).
  --env-name NAME                         micromamba env name (default: bgcmodel).
  --train PATH                            Train JSONL (default: data/processed/splits_combined/train.jsonl).
  --val PATH                              Val JSONL (default: data/processed/splits_combined/val.jsonl).
  --grad-accum N                          Gradient accumulation for preflight (default: 32).
  --max-steps N                           Optimizer steps per run (default: 30).
  --warmup-steps N                        Warmup steps (default: 50).
  --batch-size N                          Micro-batch size (default: 1).
  --activation-checkpointing              Force enable AC (default is trainer default).
  --no-activation-checkpointing           Force disable AC.
  --no-smoke-pad-to-max-seq-len           Disable fixed-width train collation.
  --dry-run                               Print commands without running training.
  -h, --help                              Show this help.

Notes:
  - This script checks GPU idleness before each length run.
  - Each run uses production-like preflight defaults (grad_accum=32, max_steps=30).
  - It records per-length logs and writes a summary TSV.
EOF
}

CHECK_EVERY_SEC=20
IDLE_HOLD_SEC=60
MIN_FREE_MIB=78000
GPU_INDEX=0
OUTPUT_ROOT="/data2/ds85/bgcmodel_runs"
HF_HOME_PATH="/data2/ds85/hf_cache"
ENV_NAME="bgcmodel"
TRAIN_JSONL="data/processed/splits_combined/train.jsonl"
VAL_JSONL="data/processed/splits_combined/val.jsonl"
GRAD_ACCUM=32
MAX_STEPS=30
WARMUP_STEPS=50
BATCH_SIZE=1
DRY_RUN=0
SMOKE_PAD_TO_MAX=1
AC_MODE="default"  # one of: default, on, off

DEFAULT_LENGTHS=(36864 40960 45056 49152 53248 57344 61440)
LENGTHS=("${DEFAULT_LENGTHS[@]}")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lengths)
      IFS=' ' read -r -a LENGTHS <<< "${2:-}"
      shift 2
      ;;
    --check-every-sec)
      CHECK_EVERY_SEC="${2:?missing value}"
      shift 2
      ;;
    --idle-hold-sec)
      IDLE_HOLD_SEC="${2:?missing value}"
      shift 2
      ;;
    --min-free-mib)
      MIN_FREE_MIB="${2:?missing value}"
      shift 2
      ;;
    --gpu-index)
      GPU_INDEX="${2:?missing value}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing value}"
      shift 2
      ;;
    --hf-home)
      HF_HOME_PATH="${2:?missing value}"
      shift 2
      ;;
    --env-name)
      ENV_NAME="${2:?missing value}"
      shift 2
      ;;
    --train)
      TRAIN_JSONL="${2:?missing value}"
      shift 2
      ;;
    --val)
      VAL_JSONL="${2:?missing value}"
      shift 2
      ;;
    --grad-accum)
      GRAD_ACCUM="${2:?missing value}"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="${2:?missing value}"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="${2:?missing value}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?missing value}"
      shift 2
      ;;
    --activation-checkpointing)
      AC_MODE="on"
      shift
      ;;
    --no-activation-checkpointing)
      AC_MODE="off"
      shift
      ;;
    --no-smoke-pad-to-max-seq-len)
      SMOKE_PAD_TO_MAX=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${#LENGTHS[@]}" -eq 0 ]]; then
  echo "No sequence lengths configured." >&2
  exit 2
fi

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
RUN_ROOT="${OUTPUT_ROOT}/queued_preflight_${RUN_TS}"
mkdir -p "$RUN_ROOT"
SUMMARY_TSV="${RUN_ROOT}/summary.tsv"
META_LOG="${RUN_ROOT}/queue.log"

echo -e "length\tstatus\tpeak_gpu_mem_gb\ttrain_log\tstdout_log\tstarted_at\tfinished_at" > "$SUMMARY_TSV"

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

extract_peak_mem() {
  local train_log="$1"
  if [[ ! -f "$train_log" ]]; then
    echo "NA"
    return
  fi
  python - "$train_log" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
peak = None
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    arr = row.get("gpu_mem_gb")
    if isinstance(arr, list) and arr:
        val = arr[0]
        if isinstance(val, (int, float)):
            peak = val if peak is None else max(peak, val)
print("NA" if peak is None else f"{peak:.2f}")
PY
}

run_length() {
  local L="$1"
  local out_dir="${RUN_ROOT}/preflight_L${L}"
  local stdout_log="${RUN_ROOT}/preflight_L${L}.log"
  local train_log="${out_dir}/train_log.jsonl"
  local started_at finished_at status peak

  local extra_args=()
  case "$AC_MODE" in
    on)  extra_args+=("--activation-checkpointing") ;;
    off) extra_args+=("--no-activation-checkpointing") ;;
  esac
  if [[ "$SMOKE_PAD_TO_MAX" -eq 1 ]]; then
    extra_args+=("--smoke-pad-to-max-seq-len")
  fi

  mkdir -p "$out_dir"
  wait_for_gpu_idle

  started_at="$(date '+%F %T')"
  log "Starting preflight L=${L} (extra_args: ${extra_args[*]:-<none>})."

  if [[ "$DRY_RUN" -eq 1 ]]; then
    cat <<EOF | tee -a "$stdout_log"
DRY RUN:
HF_HOME="$HF_HOME_PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
micromamba run -n "$ENV_NAME" \
deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
  --train "$TRAIN_JSONL" \
  --val "$VAL_JSONL" \
  --output-dir "$out_dir" \
  --max-seq-len "$L" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
  --warmup-steps "$WARMUP_STEPS" --max-epochs 1 --max-steps "$MAX_STEPS" \
  --log-every 1 --val-every 999999 --save-every 999999 \
  --wandb-mode offline ${extra_args[*]:-}
EOF
    status="dry_run"
  else
    set +e
    HF_HOME="$HF_HOME_PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      micromamba run -n "$ENV_NAME" \
      deepspeed --num_gpus=1 scripts/finetune_evo2_lora.py \
        --train "$TRAIN_JSONL" \
        --val "$VAL_JSONL" \
        --output-dir "$out_dir" \
        --max-seq-len "$L" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
        --warmup-steps "$WARMUP_STEPS" --max-epochs 1 --max-steps "$MAX_STEPS" \
        --log-every 1 --val-every 999999 --save-every 999999 \
        --wandb-mode offline "${extra_args[@]}" 2>&1 | tee "$stdout_log"
    rc="${PIPESTATUS[0]}"
    set -e
    if [[ "$rc" -eq 0 ]]; then
      status="ok"
    else
      status="failed(${rc})"
    fi
  fi

  finished_at="$(date '+%F %T')"
  peak="$(extract_peak_mem "$train_log")"
  echo -e "${L}\t${status}\t${peak}\t${train_log}\t${stdout_log}\t${started_at}\t${finished_at}" >> "$SUMMARY_TSV"
  log "Finished L=${L}: status=${status}, peak_gpu_mem_gb=${peak}"
}

log "Queued preflight run starting."
log "Lengths: ${LENGTHS[*]}"
log "Outputs: ${RUN_ROOT}"
log "Summary TSV: ${SUMMARY_TSV}"
log "Settings: batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} max_steps=${MAX_STEPS} warmup_steps=${WARMUP_STEPS}"

for L in "${LENGTHS[@]}"; do
  run_length "$L"
done

log "All requested lengths completed."
log "Done. See summary: ${SUMMARY_TSV}"
