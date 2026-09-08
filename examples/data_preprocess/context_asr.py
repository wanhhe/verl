# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess context-RL hard-negative manifest with audio into verl RLHF parquet format.

Input: JSONL with fields:
    - id: sample identifier
    - source / neg_type: data source and negative type (partial_hit / all_swap / positive)
    - audio_path: OSS object key of the WAV audio
    - context: preceding text context
    - ref: ground-truth transcription
    - must_emit: list of tokens the model must emit
    - must_reject: list of confusable tokens the model must avoid
    - user_prompt: optional user instruction (currently always empty)

Audio must be downloaded first with ``download_oss_audio.py`` into the cache
root, which mirrors the OSS key structure.  This script maps
``audio_path`` -> ``<cache-root>/<audio_path>`` with the same derivation.

Output: parquet with columns:
    - prompt: OpenAI chat messages with <audio> placeholder
    - audios: list of local audio file paths
    - answer: ground-truth transcription (ref)
    - data_source: "context_rl"
    - must_emit / must_reject: for reward computation
    - context: preceding text context (for reward computation)
    - extra_info: {id, source, neg_type, must_emit, must_reject}
    - reward_model: {"style": "rule", "ground_truth": ref}

Usage:
    /quark_speech_nas_zjk/users/wangzilin/miniconda3/envs/verl/bin/python \
        examples/data_preprocess/context_asr.py \
        --input data/context_rl/manifest_20k_band_remove_same_homo/rl_train.jsonl \
        --val_manifest data/context_rl/manifest_20k_band_remove_same_homo/rl_heldout.jsonl \
        --output /quark_speech_nas_zjk/users/wangzilin/wsc/context_rl.parquet
"""

import argparse
import json
import os

import datasets
import pandas as pd


def build_pure_asr_messages(audio_path: str) -> list[dict]:
    """Prompt: transcribe audio without any context.

    The instruction lives in the system message because Qwen3-ASR's chat
    template discards user-message text; the `<audio>` placeholder in the
    user content is replaced by the audios column payload in RLHFDataset.
    """
    return [
        {"role": "system", "content": "请转写这段语音。"},
        {"role": "user", "content": "<audio>"},
    ]


def build_context_asr_messages(context: str, audio_path: str) -> list[dict]:
    """Prompt: transcribe audio with preceding text context."""
    return [
        {"role": "system", "content": f"请根据上下文转写这段语音。\n上下文：{context}"},
        {"role": "user", "content": "<audio>"},
    ]


MESSAGE_BUILDERS = {
    "pure_asr": build_pure_asr_messages,
    "context_asr": build_context_asr_messages,
}


def default_cache_root(input_path: str) -> str:
    """<manifest-dir>/../audio_cache, matching download_oss_audio.py's default."""
    return os.path.abspath(os.path.join(os.path.dirname(input_path), "..", "audio_cache"))


def main():
    parser = argparse.ArgumentParser(description="Convert context-RL manifest JSONL to verl parquet format")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file (rl_train.jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Path to output parquet file")
    parser.add_argument(
        "--val_manifest",
        type=str,
        default=None,
        help="Optional held-out JSONL used as the test split; when absent, --train_ratio is used",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=None,
        help="Local audio cache dir downloaded by download_oss_audio.py "
        "(default: <manifest-dir>/../audio_cache)",
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        default="context_asr",
        choices=["pure_asr", "context_asr"],
        help="ASR task variant (default: context_asr)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Train split ratio, used only when --val_manifest is absent (default: 0.95)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max samples to keep (-1 = all)",
    )
    parser.add_argument(
        "--check_audio_exists",
        action="store_true",
        default=False,
        help="Verify audio files exist in the local cache (slow on network filesystems)",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="HF datasets cache for round-trip verification; defaults to $HF_DATASETS_CACHE "
        "or <output-dir>/.hf_cache (the root partition is usually full)",
    )
    args = parser.parse_args()

    if args.cache_root is None:
        args.cache_root = default_cache_root(args.input)
    if args.hf_cache_dir is None:
        args.hf_cache_dir = os.getenv("HF_DATASETS_CACHE") or os.path.join(
            os.path.dirname(os.path.abspath(args.output)), ".hf_cache"
        )
    build_messages = MESSAGE_BUILDERS[args.task_mode]

    def load_manifest(path: str) -> list[dict]:
        records = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                audio_key = rec.get("audio_path", "")
                if not audio_key:
                    skipped += 1
                    continue
                audio_path = os.path.join(args.cache_root, audio_key)
                if args.check_audio_exists and not os.path.exists(audio_path):
                    skipped += 1
                    continue

                context = rec.get("context", "")
                ref = rec.get("ref", "")
                must_emit = rec.get("must_emit") or []
                must_reject = rec.get("must_reject") or []

                messages = build_messages(context, audio_path) if "context" in args.task_mode else build_messages(audio_path)

                records.append({
                    "prompt": messages,
                    "audios": [audio_path],
                    "answer": ref,
                    "data_source": "context_rl",
                    "must_emit": must_emit,
                    "must_reject": must_reject,
                    "context": context,
                    "extra_info": {
                        "id": rec.get("id"),
                        "source": rec.get("source"),
                        "neg_type": rec.get("neg_type"),
                        "must_emit": must_emit,
                        "must_reject": must_reject,
                        "language": rec.get("language") or "zh",
                    },
                    "reward_model": {"style": "rule", "ground_truth": ref},
                })
        print(f"{path}: {len(records)} records, skipped {skipped} (missing audio / bad JSON)")
        return records

    # ---- read JSONL ----
    train_records = load_manifest(args.input)
    if args.val_manifest is not None:
        test_records = load_manifest(args.val_manifest)
    else:
        n_train = int(len(train_records) * args.train_ratio)
        test_records = train_records[n_train:]
        train_records = train_records[:n_train]

    if args.max_samples > 0 and args.max_samples < len(train_records):
        train_records = train_records[: args.max_samples]
        print(f"Truncated to {args.max_samples} samples")

    # ---- convert to parquet ----
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        if not split_records:
            continue

        df = pd.DataFrame(split_records)
        out_path = args.output.replace(".parquet", f"_{split_name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(split_records)} samples)")

    # ---- verify round-trip ----
    print("\n--- Round-trip verification (train) ---")
    ds = datasets.load_dataset(
        "parquet",
        data_files=args.output.replace(".parquet", "_train.parquet"),
        cache_dir=args.hf_cache_dir,
    )["train"]
    for i in range(min(3, len(ds))):
        sample = ds[i]
        print(f"\nSample {i}:")
        print(f"  prompt:    {json.dumps(sample['prompt'], ensure_ascii=False)[:200]}...")
        print(f"  audios:    {sample['audios']}")
        print(f"  answer:    {sample['answer']}")
        print(f"  must_emit: {sample['must_emit']}")
        print(f"  must_reject: {sample['must_reject']}")
        print(f"  context:   {sample['context'][:80]}...")


if __name__ == "__main__":
    main()
