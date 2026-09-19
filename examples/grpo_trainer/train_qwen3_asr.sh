#!/usr/bin/env bash
# Train Qwen3-ASR-1.7B with mixed semantic-pronoun and lexical-context GRPO.
#
# Expected dataset layout (downloaded from wanhhe/Context-ASR):
#   $RL_DATASET_ROOT/rl/train/{ta,context}_{train,test}.parquet
#   $RL_DATASET_ROOT/rl/audio/...
#
# Usage:
#   bash examples/grpo_trainer/train_qwen3_asr.sh
#   CUDA_VISIBLE_DEVICES=0,1 MODEL_PATH=/path/to/model \
#     bash examples/grpo_trainer/train_qwen3_asr.sh ...hydra overrides...

set -euo pipefail

VERL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$VERL_ROOT"

############################ paths ###################################

PYTHON=${PYTHON:-/root/miniconda3/envs/verl/bin/python}
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/Qwen3-ASR/finetuning/qwen3-asr-finetuning-out/checkpoint-1863}
RL_DATASET_ROOT=${RL_DATASET_ROOT:-/root/autodl-tmp/verl/Context-ASR}
RL_DATA_DIR=${RL_DATA_DIR:-$RL_DATASET_ROOT/rl/train}
LOG_DIR=${LOG_DIR:-/root/autodl-tmp/verl/logs}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/verl/checkpoints/qwen3_asr_mixed_grpo}

TRAIN_FILES=${TRAIN_FILES:-"['$RL_DATA_DIR/ta_train.parquet','$RL_DATA_DIR/context_train.parquet']"}
VAL_FILES=${VAL_FILES:-"['$RL_DATA_DIR/ta_test.parquet','$RL_DATA_DIR/context_test.parquet']"}
REWARD_FUNC=${REWARD_FUNC:-$VERL_ROOT/examples/reward_funcs/asr_mixed.py}
REWARD_FUNC_NAME=${REWARD_FUNC_NAME:-compute_score}

######################## training defaults ###########################

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

ROLLOUT_N=${ROLLOUT_N:-6}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

ACTOR_LR=${ACTOR_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
FREEZE_AUDIO=${FREEZE_AUDIO:-true}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2}
RESUME_MODE=${RESUME_MODE:-disable}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_qwen3_asr}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_asr_mixed_grpo}

########################### reward weights ###########################

# Semantic samples: 0.60 ASR + 0.40 ordered-pronoun score.
CER_WEIGHT=${CER_WEIGHT:-0.60}
MATCH_WEIGHT=${MATCH_WEIGHT:-0.40}
PRONOUN_EXACT_BONUS=${PRONOUN_EXACT_BONUS:-0.50}

# Lexical samples: 0.40 ASR + 0.35 entity + 0.25 gated rejection.
CONTEXT_CER_WEIGHT=${CONTEXT_CER_WEIGHT:-0.40}
ENTITY_WEIGHT=${ENTITY_WEIGHT:-0.35}
REJECT_WEIGHT=${REJECT_WEIGHT:-0.25}
ENTITY_EXACT_WEIGHT=${ENTITY_EXACT_WEIGHT:-0.75}
REJECT_GATE_START=${REJECT_GATE_START:-0.50}
REJECT_GATE_FULL=${REJECT_GATE_FULL:-0.80}

############################ validation ##############################

for parquet in ta_train.parquet ta_test.parquet context_train.parquet context_test.parquet; do
    if [[ ! -f "$RL_DATA_DIR/$parquet" ]]; then
        echo "Missing required dataset: $RL_DATA_DIR/$parquet" >&2
        echo "Download it with: hf download wanhhe/Context-ASR --repo-type dataset --include 'rl/**' --local-dir '$RL_DATASET_ROOT'" >&2
        exit 1
    fi
done

if [[ ! -d "$RL_DATASET_ROOT/rl/audio" ]]; then
    echo "Missing RL audio directory: $RL_DATASET_ROOT/rl/audio" >&2
    exit 1
fi

if [[ -z "$NGPUS_PER_NODE" ]]; then
    IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    NGPUS_PER_NODE=${#visible_gpus[@]}
fi

if ! [[ "$NGPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NGPUS_PER_NODE must be a positive integer, got: $NGPUS_PER_NODE" >&2
    exit 1
fi
if ! [[ "$ROLLOUT_TP" =~ ^[1-9][0-9]*$ ]] || (( NGPUS_PER_NODE % ROLLOUT_TP != 0 )); then
    echo "ROLLOUT_TP must be a positive divisor of NGPUS_PER_NODE" >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE < PPO_MINI_BATCH_SIZE || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE must be divisible by and no smaller than PPO_MINI_BATCH_SIZE" >&2
    exit 1
fi
if ! [[ "$MAX_CKPT_TO_KEEP" =~ ^[0-9]+$ ]]; then
    echo "MAX_CKPT_TO_KEEP must be a non-negative integer, got: $MAX_CKPT_TO_KEEP" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/qwen3_asr_grpo_${TIMESTAMP}.log"

export CUDA_VISIBLE_DEVICES
echo "Model:       $MODEL_PATH"
echo "Dataset:     $RL_DATASET_ROOT"
echo "GPUs:        $CUDA_VISIBLE_DEVICES ($NGPUS_PER_NODE per node)"
echo "Checkpoints: $OUTPUT_DIR"
echo "Keep latest: $MAX_CKPT_TO_KEEP checkpoints (0 means unlimited)"
echo "Log:         $LOG_FILE"

########################### verl config ##############################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.audio_root="$RL_DATASET_ROOT"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=8
    data.truncation=error
    data.return_raw_chat=True
    data.audio_key=audios
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.freeze_audio_tower="$FREEZE_AUDIO"
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=1000000
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL"
    actor_rollout_ref.rollout.n="$ROLLOUT_N"
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE"
    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=1000000
)

REWARD=(
    reward.custom_reward_function.path="$REWARD_FUNC"
    reward.custom_reward_function.name="$REWARD_FUNC_NAME"
    reward.custom_reward_function.reward_kwargs.cer_weight="$CER_WEIGHT"
    reward.custom_reward_function.reward_kwargs.match_weight="$MATCH_WEIGHT"
    +reward.custom_reward_function.reward_kwargs.pronoun_exact_bonus="$PRONOUN_EXACT_BONUS"
    +reward.custom_reward_function.reward_kwargs.context_cer_weight="$CONTEXT_CER_WEIGHT"
    +reward.custom_reward_function.reward_kwargs.entity_weight="$ENTITY_WEIGHT"
    +reward.custom_reward_function.reward_kwargs.reject_weight="$REJECT_WEIGHT"
    +reward.custom_reward_function.reward_kwargs.entity_exact_weight="$ENTITY_EXACT_WEIGHT"
    +reward.custom_reward_function.reward_kwargs.reject_gate_start="$REJECT_GATE_START"
    +reward.custom_reward_function.reward_kwargs.reject_gate_full="$REJECT_GATE_FULL"
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.n_gpus_per_node="$NGPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    trainer.default_local_dir="$OUTPUT_DIR"
    trainer.max_actor_ckpt_to_keep="$MAX_CKPT_TO_KEEP"
    trainer.max_critic_ckpt_to_keep="$MAX_CKPT_TO_KEEP"
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
    trainer.resume_mode="$RESUME_MODE"
)

# vLLM 0.18 must start EngineCore with spawn after CUDA initialization.
EXTRA=(
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD=spawn
)

"$PYTHON" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
