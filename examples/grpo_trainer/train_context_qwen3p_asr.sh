#!/usr/bin/env bash
# One-click launcher for Qwen3-ASR-1.7B GRPO training on the context-RL dataset
# (hard-negative context ASR, reward = entity match + 1-CER via
# context_robustness.py, shared with the eval scorer).
#
# The grpo_trainer run script defaults to the CMRC data and asr_cer reward, so
# this wrapper exports TRAIN_FILES / VAL_FILES / REWARD_FUNC to point at the
# context-RL parquet and the context_robustness reward.  verl is not
# pip-installed, so we cd to the repo root for `import verl`.
#
# Usage:
#   bash examples/grpo_trainer/train_context_qwen3p_asr.sh               # defaults
#   bash examples/grpo_trainer/train_context_qwen3p_asr.sh ...extra args...  # override
#
# Overridable env (passthrough to run_qwen3_asr_1_7b_fsdp.sh):
#   CUDA_VISIBLE_DEVICES, LOG_DIR, MANIFEST_DIR, MODEL_PATH, PYTHON,
#   TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE, ROLLOUT_N, ACTOR_LR, KL_LOSS_COEF,
#   MATCH_WEIGHT, TOTAL_EPOCHS, SAVE_FREQ, TEST_FREQ, EXPERIMENT_NAME, ...

set -euo pipefail

# verl repo root (this script lives in examples/grpo_trainer/)
VERL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$VERL_ROOT"

# ---- user-adjustable ----
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
LOG_DIR=${LOG_DIR:-/quark_speech_nas_zjk/users/wangzilin/verl/logs}
MANIFEST_DIR=${MANIFEST_DIR:-/quark_speech_nas_zjk/users/wangzilin/data/context_rl/manifest_20k_band_remove_same_homo}
MODEL_PATH=${MODEL_PATH:-/quark_speech_nas_zjk/users/shao.shuai/Asrproj-train/outputs/formal_1node_hardneg_scatter_ss_0726/checkpoint-2000}
PYTHON=${PYTHON:-/quark_speech_nas_zjk/users/wangzilin/miniconda3/envs/verl/bin/python}

# Training schedule (exported; run_qwen3_asr_1_7b_fsdp.sh reads these).
# NOTE: this verl version has no log-frequency knob — the console logger prints
# metrics every step; only the validation frequency is configurable.
TEST_FREQ=${TEST_FREQ:-20}                 # run validation on val_files every N steps
SAVE_FREQ=${SAVE_FREQ:-100}                 # save a model checkpoint every N steps
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}   # explicit total steps; empty -> derived from epochs

# Batch size (exported; run_qwen3_asr_1_7b_fsdp.sh reads these)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}                 # global batch: prompts sampled per step (all GPUs)
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}            # PPO update minibatch; grad steps/step = batch / mini
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}  # dynamic-bsz token cap per GPU
ROLLOUT_N=${ROLLOUT_N:-6}                                 # GRPO group size: sampled outputs per prompt

# Derive the number of GPUs from CUDA_VISIBLE_DEVICES so Ray and the trainer
# agree on trainer.n_gpus_per_node when running with fewer than 8 GPUs.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')}

# context-RL dataset + reward overrides (read by run_qwen3_asr_1_7b_fsdp.sh)
export TRAIN_FILES=${TRAIN_FILES:-$MANIFEST_DIR/context_rl_train.parquet}
export VAL_FILES=${VAL_FILES:-$MANIFEST_DIR/context_rl_test.parquet}
export REWARD_FUNC=${REWARD_FUNC:-$VERL_ROOT/examples/reward_funcs/context_robustness.py}

export CUDA_VISIBLE_DEVICES MODEL_PATH PYTHON NGPUS_PER_NODE \
    TEST_FREQ SAVE_FREQ TOTAL_EPOCHS TOTAL_TRAINING_STEPS \
    TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MAX_TOKEN_LEN_PER_GPU ROLLOUT_N

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/qwen3_asr_context_rl_grpo_${TIMESTAMP}.log"
echo "Log: $LOG_FILE"

# checkpoints go to verl's default <CWD>/checkpoints/<project>/<experiment>/global_step_*
# (CWD is $VERL_ROOT here, so they land under the project); to put them
# elsewhere, pass trainer.default_local_dir=/path/to/dir as a command-line arg
# gradient checkpointing off: GPU memory has plenty of headroom and the
# forward recomputation it avoids is worth more than the memory it frees
bash "$VERL_ROOT/examples/grpo_trainer/run_qwen3_asr_1_7b_fsdp.sh" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "$@" 2>&1 | tee "$LOG_FILE"
