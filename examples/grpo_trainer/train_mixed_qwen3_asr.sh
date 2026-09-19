#!/usr/bin/env bash
# Launch mixed semantic-pronoun and lexical entity/distractor GRPO training.
#
# Required:
#   RL_DATA_DIR  Directory containing ta_{train,test}.parquet and
#                context_{train,test}.parquet.
#
# The underlying verl dataset loader concatenates the parquet files and keeps
# ``data_source`` per row.  asr_mixed.py dispatches the appropriate reward and
# returns a fixed metric schema for every sample.

set -euo pipefail

VERL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$VERL_ROOT"

: "${RL_DATA_DIR:?Set RL_DATA_DIR to the directory containing the four RL parquet files}"
RL_DATASET_ROOT=${RL_DATASET_ROOT:-$(cd "$RL_DATA_DIR/../.." && pwd)}

for parquet in ta_train.parquet ta_test.parquet context_train.parquet context_test.parquet; do
    if [[ ! -f "$RL_DATA_DIR/$parquet" ]]; then
        echo "Missing required dataset: $RL_DATA_DIR/$parquet" >&2
        exit 1
    fi
done

export TRAIN_FILES=${TRAIN_FILES:-"['$RL_DATA_DIR/ta_train.parquet','$RL_DATA_DIR/context_train.parquet']"}
export VAL_FILES=${VAL_FILES:-"['$RL_DATA_DIR/ta_test.parquet','$RL_DATA_DIR/context_test.parquet']"}
export REWARD_FUNC=${REWARD_FUNC:-$VERL_ROOT/examples/reward_funcs/asr_mixed.py}

# Semantic reward: 0.60 ASR + 0.40 ordered-pronoun score.
export CER_WEIGHT=${CER_WEIGHT:-0.60}
export MATCH_WEIGHT=${MATCH_WEIGHT:-0.40}
export PRONOUN_EXACT_BONUS=${PRONOUN_EXACT_BONUS:-0.50}

# Lexical reward: 0.40 ASR + 0.35 entity + 0.25 gated rejection.
CONTEXT_CER_WEIGHT=${CONTEXT_CER_WEIGHT:-0.40}
export ENTITY_WEIGHT=${ENTITY_WEIGHT:-0.35}
export REJECT_WEIGHT=${REJECT_WEIGHT:-0.25}
export ENTITY_EXACT_WEIGHT=${ENTITY_EXACT_WEIGHT:-0.75}
export REJECT_GATE_START=${REJECT_GATE_START:-0.50}
export REJECT_GATE_FULL=${REJECT_GATE_FULL:-0.80}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_asr_mixed_grpo}

bash "$VERL_ROOT/examples/grpo_trainer/train_qwen3_asr.sh" \
    data.audio_root="$RL_DATASET_ROOT" \
    +reward.custom_reward_function.reward_kwargs.context_cer_weight=${CONTEXT_CER_WEIGHT} \
    "$@"
