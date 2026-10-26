#!/usr/bin/env bash
set -euo pipefail

ROOT="${RIPT_VLA_LITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_ID="${1:?usage: run_quest_eval.sh EXP_ID CHECKPOINT TASK_NAME|ALL [ROLLOUTS] [PARALLEL] [SEED]}"
CHECKPOINT="${2:?checkpoint required}"
TASK_NAME="${3:?task name or ALL required}"
ROLLOUTS="${4:-50}"
PARALLEL="${5:-5}"
SEED="${6:-0}"

if [[ ! "${EXP_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid EXP_ID" >&2
  exit 1
fi
if [[ ! "${ROLLOUTS}" =~ ^[0-9]+$ ]] || [[ ! "${PARALLEL}" =~ ^[0-9]+$ ]] || [ "${ROLLOUTS}" -lt 1 ] || [ "${PARALLEL}" -lt 1 ]; then
  echo "ROLLOUTS and PARALLEL must be positive integers" >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

source "${ROOT}/scripts/env_setup.sh"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"
export HF_HUB_OFFLINE=1
export RIPT_VIDEO_COUNT="${VIDEO_COUNT:-0}"
export RIPT_VIDEO_DIR="${ROOT}/evidence/${EXP_ID}/videos"
export RIPT_NUM_CANDIDATES="${NUM_CANDIDATES:-0}"
export RIPT_VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-}"
export RIPT_ROLLOUT_DIR="${ROLLOUT_DIR:-}"
export RIPT_INIT_STATE_START="${INIT_STATE_START:-0}"
EVIDENCE="${ROOT}/evidence/${EXP_ID}"
export RIPT_EPISODE_METADATA_PATH="${EVIDENCE}/episode_results.json"
mkdir -p "${EVIDENCE}/logs"
if [ -f "${EVIDENCE}/logs/eval.log" ]; then
  echo "refusing to overwrite ${EVIDENCE}/logs/eval.log" >&2
  exit 1
fi
EXPECTED_COUNT=$(( (ROLLOUTS + PARALLEL - 1) / PARALLEL * PARALLEL ))
if [ "${TASK_NAME}" = "ALL" ]; then
  EXPECTED_COUNT=$((EXPECTED_COUNT * 10))
fi
CHECKPOINT_SHA="$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)"
export RIPT_CHECKPOINT_SHA256="${CHECKPOINT_SHA}"
STARTED_AT="$(date -Iseconds)"

cat > "${EVIDENCE}/metadata.yaml" <<EOF
experiment_id: ${EXP_ID}
status: running
started_at: ${STARTED_AT}
checkpoint: ${CHECKPOINT}
checkpoint_sha256: ${CHECKPOINT_SHA}
task_name: ${TASK_NAME}
rollouts_per_env: ${ROLLOUTS}
num_parallel_envs: ${PARALLEL}
expected_rollout_count: ${EXPECTED_COUNT}
video_count: ${RIPT_VIDEO_COUNT}
seed: ${SEED}
reset_type: ori
num_candidates: ${RIPT_NUM_CANDIDATES}
value_checkpoint: ${RIPT_VALUE_CHECKPOINT}
rollout_dir: ${RIPT_ROLLOUT_DIR}
init_state_start: ${RIPT_INIT_STATE_START}
EOF

cd "${ROOT}/sources/ript-vla"
MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
if [ "${TASK_NAME}" = "ALL" ]; then
  TASK_OVERRIDE="task.task_names_to_use=null"
else
  TASK_OVERRIDE='task.task_names_to_use=["'"${TASK_NAME}"'"]'
fi

cat > "${EVIDENCE}/resolved_overrides.txt" <<EOF
task=libero_long_rl
exp_name=${EXP_ID}
variant_name=eval
rollout.rollouts_per_env=${ROLLOUTS}
task.env_runner.num_parallel_envs=${PARALLEL}
checkpoint_path=${CHECKPOINT}
task.env_runner.reset_type=ori
seed=${SEED}
logging.mode=disabled
${TASK_OVERRIDE}
EOF

set +e
NCCL_TIMEOUT=108000 torchrun --nproc_per_node=1 --master_port "${MASTER_PORT}" eval_quest.py --config-name=train_prior_rl.yaml \
  task=libero_long_rl \
  exp_name="${EXP_ID}" \
  variant_name=eval \
  rollout.rollouts_per_env="${ROLLOUTS}" \
  task.env_runner.num_parallel_envs="${PARALLEL}" \
  checkpoint_path="${CHECKPOINT}" \
  task.env_runner.reset_type=ori \
  seed="${SEED}" \
  logging.mode=disabled \
  "${TASK_OVERRIDE}" 2>&1 | tee "${EVIDENCE}/logs/eval.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

python "${ROOT}/scripts/parse_eval_log.py" \
  "${EVIDENCE}/logs/eval.log" \
  "${EVIDENCE}/summary.json" \
  "${EXP_ID}" \
  "${TASK_NAME}" \
  "${EXPECTED_COUNT}" \
  "${EXIT_CODE}"

printf "status: %s\nexit_code: %s\nfinished_at: %s\n" \
  "$([ "${EXIT_CODE}" -eq 0 ] && echo completed || echo failed)" \
  "${EXIT_CODE}" "$(date -Iseconds)" > "${EVIDENCE}/status.yaml"
exit "${EXIT_CODE}"
