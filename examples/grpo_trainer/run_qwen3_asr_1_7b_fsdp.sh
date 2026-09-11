#!/usr/bin/env bash
# GRPO | Qwen3-ASR-1.7B | FSDP training | ASR (CMRC2019 coref context-asr)
#
# Qwen3-ASR specifics:
#   - audio encoder (thinker.audio_tower) is frozen; LLM part trains full-param
#   - rollout uses the vLLM string-prompt path (vLLM 0.18 TokensPrompt+audio bug)
#   - reward = 1 - CER (see examples/reward_funcs/asr_cer.py)
#
# Knobs:
#   NGPUS_PER_NODE  GPUs per node            (default: 8)
#   NNODES          node count               (default: 1)
#   ROLLOUT_N       GRPO group size          (default: 6, following the Tongyi paper)
#   FREEZE_AUDIO    freeze audio encoder     (default: true)

set -xeuo pipefail

########################### user-adjustable ###########################
PYTHON=${PYTHON:-/quark_speech_nas_zjk/users/wangzilin/miniconda3/envs/verl/bin/python}
MODEL_PATH=${MODEL_PATH:-/quark_speech_nas_zjk/users/wangzilin/pretrained/Qwen3-ASR-1.7B}
DATA_ROOT=${DATA_ROOT:-/quark_speech_nas_zjk/users/wangzilin/wsc}
VERL_ROOT=${VERL_ROOT:-/quark_speech_nas_zjk/users/wangzilin/verl}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}   # covers expanded audio tokens
max_response_length=${MAX_RESPONSE_LENGTH:-256}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}

rollout_n=${ROLLOUT_N:-6}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_p=${ROLLOUT_TOP_P:-1.0}
rollout_tp=${ROLLOUT_TP:-1}                    # 1.7B fits one GPU; use DP replicas for scale
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}

# Custom ASR reward weights. Reward functions ignore knobs that do not apply to
# their data source, so this run script can serve both pronoun and context-ASR.
entity_weight=${ENTITY_WEIGHT:-0.35}
reject_weight=${REJECT_WEIGHT:-0.25}
entity_exact_weight=${ENTITY_EXACT_WEIGHT:-0.75}
reject_gate_start=${REJECT_GATE_START:-0.50}
reject_gate_full=${REJECT_GATE_FULL:-0.80}
pronoun_exact_bonus=${PRONOUN_EXACT_BONUS:-0.50}

FREEZE_AUDIO=${FREEZE_AUDIO:-true}

total_epochs=${TOTAL_EPOCHS:-10}
total_training_steps=${TOTAL_TRAINING_STEPS:-null}   # explicit step count; null -> derived from epochs
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_qwen3_asr}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_asr_1_7b_grpo_fsdp_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILES:-$DATA_ROOT/test_cmrc2019_train.parquet}"
    data.val_files="${VAL_FILES:-$DATA_ROOT/test_cmrc2019_test.parquet}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=64
    data.truncation='error'
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
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.freeze_audio_tower=${FREEZE_AUDIO}
    # Qwen3-ASR layer classes live in the qwen_asr package and cannot be resolved
    # by the default _no_split_modules wrap policy; use a size-based policy instead.
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=1000000
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=1000000
)

REWARD=(
    reward.custom_reward_function.path="${REWARD_FUNC:-$VERL_ROOT/examples/reward_funcs/asr_cer.py}"
    reward.custom_reward_function.name=${REWARD_FUNC_NAME:-compute_score}
    reward.custom_reward_function.reward_kwargs.cer_weight=${CER_WEIGHT:-0.5}
    reward.custom_reward_function.reward_kwargs.match_weight=${MATCH_WEIGHT:-0.5}
    +reward.custom_reward_function.reward_kwargs.entity_weight=${entity_weight}
    +reward.custom_reward_function.reward_kwargs.reject_weight=${reject_weight}
    +reward.custom_reward_function.reward_kwargs.entity_exact_weight=${entity_exact_weight}
    +reward.custom_reward_function.reward_kwargs.reject_gate_start=${reject_gate_start}
    +reward.custom_reward_function.reward_kwargs.reject_gate_full=${reject_gate_full}
    +reward.custom_reward_function.reward_kwargs.pronoun_exact_bonus=${pronoun_exact_bonus}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

# vLLM 0.18 starts its EngineCore via fork by default, which crashes after CUDA
# init; spawn is required. Propagate through Ray's runtime env so every worker
# (rollout server, EngineCore) inherits it.
EXTRA=(
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD=spawn
)

########################### launch ###########################
$PYTHON -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
