#!/usr/bin/env bash
# Convert a verl FSDP checkpoint to an official HF checkpoint directory.
#
# Usage:
#   bash scripts/convert_fsdp_ckpt_to_hf.sh \
#       --ckpt  <global_step_XX/actor> \
#       --model <original HF model dir> \
#       --output <output HF dir>
#
# Environment overrides:
#   PYTHON   python interpreter (default: python from the active environment)

set -euo pipefail

PYTHON=${PYTHON:-python}

# ---- parse args ----
CKPT=""
MODEL=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)  CKPT="$2";  shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CKPT" || -z "$MODEL" || -z "$OUTPUT" ]]; then
    echo "Usage: bash $0 --ckpt <ckpt/actor> --model <hf model dir> --output <out dir>" >&2
    exit 1
fi

echo "==> converting checkpoint: $CKPT"
echo "==> model config/tokenizer source: $MODEL"
echo "==> output: $OUTPUT"

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # verl repo root

"$PYTHON" scripts/convert_fsdp_ckpt_to_hf.py \
    --ckpt "$CKPT" \
    --model "$MODEL" \
    --output "$OUTPUT"

echo "==> done: $OUTPUT"
