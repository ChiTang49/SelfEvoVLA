#!/usr/bin/env bash
set -euo pipefail

ROOT="${RIPT_VLA_LITE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP_ID="${1:?usage: run_quest_train.sh EXP_ID TASK_NAME MODE [SEED] [CHECKPOINT]}"
TASK_NAME="${2:?task name required}"
MODE="${3:?mode required: smoke, rloo, bript}"
SEED="${4:-0}"
CHECKPOINT="${5:-${ROOT}/checkpoints/quest/LIBERO_LONG_sft.pth}"

if [[ ! "${EXP_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid EXP_ID" >&2
  exit 1
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "SEED must be numeric" >&2
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

case "${MODE}" in
  smoke|bript-smoke)
    BATCH=2
    K=4
    PPO_BATCH=2
    PPO_EPOCHS=1
    STEPS=1
    SAVE_INTERVAL=1
    ROLLOUT_STEPS=1
    ROLLOUT_ENABLED=false
    ROLLOUTS=4
    PARALLEL=4
    EARLY_STOP=1.0
    DYNAMIC=false
    LOGGING=disabled
    if [ "${MODE}" = "bript-smoke" ]; then
      DYNAMIC=true
    fi
    ;;
  rloo|bript)
    BATCH=10
    K=16
    PPO_BATCH=8
    PPO_EPOCHS=20
    STEPS=15
    SAVE_INTERVAL=5
    ROLLOUT_STEPS=5
    ROLLOUT_ENABLED=true
    ROLLOUTS=48
    PARALLEL="${PARALLEL_ENVS:-4}"
    EARLY_STOP=0.8
    DYNAMIC=false
    LOGGING=offline
    if [ "${MODE}" = "bript" ]; then
      DYNAMIC=true
    fi
    ;;
  *)
    echo "MODE must be smoke, bript-smoke, rloo, or bript" >&2
    exit 1
    ;;
esac

source "${ROOT}/scripts/env_setup.sh"
export HF_HUB_OFFLINE=1
EVIDENCE="${ROOT}/evidence/${EXP_ID}"
mkdir -p "${EVIDENCE}/logs" "${EVIDENCE}/checkpoints"
if [ -f "${EVIDENCE}/logs/train.log" ]; then
  echo "refusing to overwrite ${EVIDENCE}/logs/train.log" >&2
  exit 1
fi
STARTED_AT="$(date -Iseconds)"
CHECKPOINT_SHA="$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)"
EXTRA_OVERRIDES=()
if [ -n "${RIPT_EXTRA_OVERRIDES:-}" ]; then
  read -r -a EXTRA_OVERRIDES <<< "${RIPT_EXTRA_OVERRIDES}"
fi
for OVERRIDE in "${EXTRA_OVERRIDES[@]}"; do
  case "${OVERRIDE}" in
    training.n_steps=*) STEPS="${OVERRIDE#*=}" ;;
    training.save_interval=*) SAVE_INTERVAL="${OVERRIDE#*=}" ;;
  esac
done

cat > "${EVIDENCE}/metadata.yaml" <<EOF
experiment_id: ${EXP_ID}
status: running
mode: ${MODE}
started_at: ${STARTED_AT}
task_name: ${TASK_NAME}
seed: ${SEED}
initial_checkpoint: ${CHECKPOINT}
initial_checkpoint_sha256: ${CHECKPOINT_SHA}
train_dataloader_batch_size: ${BATCH}
rloo_batch_size: ${K}
ppo_batch_size: ${PPO_BATCH}
ppo_epochs: ${PPO_EPOCHS}
training_steps: ${STEPS}
save_interval: ${SAVE_INTERVAL}
rollout_steps: ${ROLLOUT_STEPS}
rollout_enabled: ${ROLLOUT_ENABLED}
rollouts_per_env: ${ROLLOUTS}
num_parallel_envs: ${PARALLEL}
early_stop_percentage: ${EARLY_STOP}
dynamic_sampling: ${DYNAMIC}
logging_mode: ${LOGGING}
extra_overrides: ${RIPT_EXTRA_OVERRIDES:-}
EOF

cd "${ROOT}/sources/ript-vla"
MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
TASK_OVERRIDE='task.task_names_to_use=["'"${TASK_NAME}"'"]'
cat > "${EVIDENCE}/resolved_overrides.txt" <<EOF
task=libero_long_rl
exp_name=${EXP_ID}
variant_name=${MODE}_seed${SEED}
train_dataloader.batch_size=${BATCH}
algo.rloo_batch_size=${K}
algo.ppo_batch_size=${PPO_BATCH}
algo.num_ppo_epochs=${PPO_EPOCHS}
training.n_steps=${STEPS}
training.save_interval=${SAVE_INTERVAL}
training.rollout_steps=${ROLLOUT_STEPS}
rollout.enabled=${ROLLOUT_ENABLED}
rollout.rollouts_per_env=${ROLLOUTS}
task.env_runner.num_parallel_envs=${PARALLEL}
algo.early_stop_porcentage=${EARLY_STOP}
algo.enable_dynamic_sampling=${DYNAMIC}
algo.use_token_level_loss_avg=true
algo.policy.scheduler_factory.eta_min=9e-7
checkpoint_path=${CHECKPOINT}
task.env_runner.reset_type=ori
seed=${SEED}
logging.mode=${LOGGING}
${TASK_OVERRIDE}
${EXTRA_OVERRIDES:-}
EOF

if [ "${RIPT_DRY_RUN:-0}" = "1" ]; then
  echo "dry run: training.n_steps=${STEPS}, save_interval=${SAVE_INTERVAL}"
  exit 0
fi

set +e
NCCL_TIMEOUT=108000 torchrun --nproc_per_node=1 --master_port "${MASTER_PORT}" train_ript.py --config-name=train_prior_rl.yaml \
  task=libero_long_rl \
  exp_name="${EXP_ID}" \
  variant_name="${MODE}_seed${SEED}" \
  train_dataloader.batch_size="${BATCH}" \
  algo.rloo_batch_size="${K}" \
  algo.ppo_batch_size="${PPO_BATCH}" \
  algo.num_ppo_epochs="${PPO_EPOCHS}" \
  training.n_steps="${STEPS}" \
  training.save_interval="${SAVE_INTERVAL}" \
  training.rollout_steps="${ROLLOUT_STEPS}" \
  rollout.enabled="${ROLLOUT_ENABLED}" \
  rollout.rollouts_per_env="${ROLLOUTS}" \
  task.env_runner.num_parallel_envs="${PARALLEL}" \
  algo.early_stop_porcentage=${EARLY_STOP} \
  algo.enable_dynamic_sampling="${DYNAMIC}" \
  algo.use_token_level_loss_avg=true \
  algo.policy.scheduler_factory.eta_min=9e-7 \
  checkpoint_path="${CHECKPOINT}" \
  task.env_runner.reset_type=ori \
  seed="${SEED}" \
  logging.mode="${LOGGING}" \
  "${TASK_OVERRIDE}" \
  "${EXTRA_OVERRIDES[@]}" 2>&1 | tee "${EVIDENCE}/logs/train.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [ "${EXIT_CODE}" -ne 0 ]; then
  printf "status: failed\nexit_code: %s\nfinished_at: %s\n" "${EXIT_CODE}" "$(date -Iseconds)" > "${EVIDENCE}/status.yaml"
  exit "${EXIT_CODE}"
fi

FINAL_CHECKPOINT="$(find "${ROOT}/outputs" -path "*${EXP_ID}*" -type f -name "multitask_model_step_*.pth" -printf "%T@ %p\n" | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [ -z "${FINAL_CHECKPOINT}" ] || [ ! -f "${FINAL_CHECKPOINT}" ]; then
  printf "status: failed\nerror: no checkpoint produced\nfinished_at: %s\n" "$(date -Iseconds)" > "${EVIDENCE}/status.yaml"
  exit 2
fi
cp "${FINAL_CHECKPOINT}" "${EVIDENCE}/checkpoints/final.pth"
FINAL_SHA="$(sha256sum "${EVIDENCE}/checkpoints/final.pth" | cut -d' ' -f1)"
printf "source_checkpoint: %s\nevidence_checkpoint: %s\nsha256: %s\n" \
  "${FINAL_CHECKPOINT}" "${EVIDENCE}/checkpoints/final.pth" "${FINAL_SHA}" > "${EVIDENCE}/last_checkpoint.txt"

if [ "${MODE}" = "smoke" ] || [ "${MODE}" = "bript-smoke" ]; then
  bash "${ROOT}/scripts/run_quest_eval.sh" "${EXP_ID}-reload" "${EVIDENCE}/checkpoints/final.pth" "${TASK_NAME}" 1 1 "${SEED}"
fi

printf "status: completed\nexit_code: 0\nfinished_at: %s\nfinal_checkpoint_sha256: %s\n" \
  "$(date -Iseconds)" "${FINAL_SHA}" > "${EVIDENCE}/status.yaml"
printf "completed %s\nfinal checkpoint: %s\n" "${EXP_ID}" "${EVIDENCE}/checkpoints/final.pth"
