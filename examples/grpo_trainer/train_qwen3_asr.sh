#!/usr/bin/env bash
# One-click launcher for Qwen3-ASR-1.7B GRPO training (1 GPU by default).
#
# Usage:
#   bash examples/grpo_trainer/train_qwen3_asr.sh                 # train with defaults
#   bash examples/grpo_trainer/train_qwen3_asr.sh ...extra args...  # override any knob
#
# Overridable knobs (passed through to run_qwen3_asr_1_7b_fsdp.sh):
#   CUDA_VISIBLE_DEVICES, LOG_DIR, TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE,
#   ROLLOUT_N, ACTOR_LR, KL_LOSS_COEF, MAX_PROMPT_LENGTH, ... (see the run script)

set -euo pipefail

# verl repo root (this script lives in examples/grpo_trainer/)
VERL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$VERL_ROOT"

# ---- user-adjustable ----
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LOG_DIR=${LOG_DIR:-/root/autodl-tmp/verl/logs}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-6}                       # GRPO group size (Tongyi paper)
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ACTOR_LR=${ACTOR_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
CER_WEIGHT=${CER_WEIGHT:-0.6}                   # reward = CER + ordered-pronoun sequence score
MATCH_WEIGHT=${MATCH_WEIGHT:-0.4}               # pronoun component weight (kept for CLI compatibility)
PRONOUN_EXACT_BONUS=${PRONOUN_EXACT_BONUS:-0.5}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
# disable = fresh training from MODEL_PATH weights only (no auto-resume);
# auto = resume from the last checkpoint in the experiment's output dir.
RESUME_MODE=${RESUME_MODE:-disable}
# ---- end user-adjustable ----

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/qwen3_asr_grpo_${TIMESTAMP}.log"
echo "Log: $LOG_FILE"

export VERL_ROOT CUDA_VISIBLE_DEVICES CER_WEIGHT MATCH_WEIGHT PRONOUN_EXACT_BONUS
export REWARD_FUNC=${REWARD_FUNC:-$VERL_ROOT/examples/reward_funcs/asr_cer.py}

# derive the number of GPUs from CUDA_VISIBLE_DEVICES
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')

# launch training (defaults are the full config; extra args override)
bash "$VERL_ROOT/examples/grpo_trainer/run_qwen3_asr_1_7b_fsdp.sh" \
  trainer.n_gpus_per_node=${N_GPUS} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.resume_mode=${RESUME_MODE} \
  "$@" 2>&1 | tee "$LOG_FILE"
