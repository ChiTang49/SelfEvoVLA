#!/usr/bin/env bash
set -euo pipefail

ROOT="${RIPT_VLA_LITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_ID="${1:?experiment id required}"
CHECKPOINT="${2:?checkpoint required}"
DATA_DIR="${3:?feedback rollout directory required}"
TASK_NAME="${4:?task name required}"
EPOCHS="${5:-2}"

if [[ ! "${EXP_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid EXP_ID" >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ] || [ ! -d "${DATA_DIR}" ]; then
  echo "checkpoint or data directory not found" >&2
  exit 1
fi
source "${ROOT}/scripts/env_setup.sh"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export RIPT_FEEDBACK_DATA_DIR="${DATA_DIR}"
export RIPT_FEEDBACK_OUTPUT="${ROOT}/evidence/${EXP_ID}/checkpoints/feedback.pth"
export RIPT_FEEDBACK_EPOCHS="${EPOCHS}"
mkdir -p "${ROOT}/evidence/${EXP_ID}/logs"
if [ -f "${ROOT}/evidence/${EXP_ID}/logs/train.log" ]; then
  echo "refusing to overwrite existing feedback log" >&2
  exit 1
fi
MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
set +e
NCCL_TIMEOUT=108000 torchrun --nproc_per_node=1 --master_port "${MASTER_PORT}" "${ROOT}/planning/feedback_train.py" \
  --config-name=train_prior_rl.yaml \
  task=libero_long_rl \
  exp_name="${EXP_ID}" \
  variant_name=feedback \
  checkpoint_path="${CHECKPOINT}" \
  task.env_runner.reset_type=ori \
  seed=0 \
  logging.mode=disabled \
  "task.task_names_to_use=[\"${TASK_NAME}\"]" 2>&1 | tee "${ROOT}/evidence/${EXP_ID}/logs/train.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e
printf "status: %s\nexit_code: %s\nfinished_at: %s\n" \
  "$([ "${EXIT_CODE}" -eq 0 ] && echo completed || echo failed)" "${EXIT_CODE}" "$(date -Iseconds)" \
  > "${ROOT}/evidence/${EXP_ID}/status.yaml"
exit "${EXIT_CODE}"
