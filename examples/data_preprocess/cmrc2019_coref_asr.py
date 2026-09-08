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
Preprocess CMRC2019 coreference dataset with audio into verl RLHF parquet format.

Input: JSONL with fields:
    - context: preceding text context
    - id: sample identifier
    - text: ground-truth transcription (sentence containing pronouns)
    - pronoun: list of target pronouns in the sentence
    - audio_path: path to the WAV audio file

Output: parquet with columns:
    - prompt: OpenAI chat messages with <audio> placeholder
    - audios: list of audio file paths
    - answer: ground-truth transcription
    - data_source: "cmrc2019_coref"
    - pronoun: list of target pronouns (for reward computation)
    - context: preceding text context (for reward computation)

Task variants (controlled by --task_mode):
    - "pure_asr": transcribe audio only, no context given
    - "context_asr": transcribe audio with context text provided
    - "coref_asr": transcribe audio + identify pronoun referents

Usage:
    python examples/data_preprocess/cmrc2019_coref_asr.py \
        --input /path/to/cmrc2019_coref_dev_with_audio.jsonl \
        --output /path/to/output.parquet \
        --task_mode context_asr
"""

import argparse
import json
import os
import sys

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


def build_coref_asr_messages(context: str, pronoun: list[str], audio_path: str) -> list[dict]:
    """Prompt: transcribe and resolve pronouns."""
    pronouns_str = "、".join(pronoun)
    return [
        {"role": "system", "content": (
            f"请转写这段语音，并指出代词「{pronouns_str}」分别指代什么。\n上下文：{context}"
        )},
        {"role": "user", "content": "<audio>"},
    ]


MESSAGE_BUILDERS = {
    "pure_asr": build_pure_asr_messages,
    "context_asr": build_context_asr_messages,
    "coref_asr": build_coref_asr_messages,
}


def main():
    parser = argparse.ArgumentParser(description="Convert CMRC2019 coref + audio JSONL to verl parquet format")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output parquet file")
    parser.add_argument(
        "--task_mode",
        type=str,
        default="context_asr",
        choices=["pure_asr", "context_asr", "coref_asr"],
        help="ASR task variant (default: context_asr)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.95,
        help="Train split ratio (default: 0.95, i.e. last 5% as validation)",
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
        help="Verify audio files exist on disk (slow on network filesystems)",
    )
    args = parser.parse_args()

    build_messages = MESSAGE_BUILDERS[args.task_mode]

    # ---- read JSONL ----
    records = []
    skipped = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # ---- format detection ----
            # Legacy flat format: {context, text, pronoun, audio_path}
            # Chat format:       {messages: [...], audios: [...], pronoun: [...]}
            if "messages" in rec:
                # extract audio path(s) from the audios column
                audio_paths = rec.get("audios") or []
                audio_path = audio_paths[0] if audio_paths else ""
                # ground truth: assistant message text
                text = ""
                for msg in rec.get("messages", []):
                    if msg.get("role") == "assistant":
                        for item in msg.get("content", []):
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("value", "")
                # context: first audio item's extra.context (list) joined
                context = ""
                for msg in rec.get("messages", []):
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "audio":
                            ctx = (item.get("extra") or {}).get("context") or []
                            context = "，".join(ctx)
                pronoun = rec.get("pronoun", [])
            else:
                audio_path = rec.get("audio_path", "")
                context = rec.get("context", "")
                text = rec.get("text", "")
                pronoun = rec.get("pronoun", [])

            if not audio_path:
                skipped += 1
                continue
            if args.check_audio_exists and not os.path.exists(audio_path):
                skipped += 1
                continue

            messages = build_messages(context, audio_path) if "context" in args.task_mode else build_messages(audio_path)

            records.append({
                "prompt": messages,
                "audios": [audio_path],
                "answer": text,
                "data_source": "cmrc2019_coref",
                "pronoun": pronoun,
                "context": context,
                "extra_info": {"pronoun": pronoun},
                "reward_model": {"style": "rule", "ground_truth": text},
            })

    print(f"Loaded {len(records)} records, skipped {skipped} (missing audio / bad JSON)")

    if args.max_samples > 0 and args.max_samples < len(records):
        records = records[: args.max_samples]
        print(f"Truncated to {args.max_samples} samples")

    # ---- split train / test ----
    n_train = int(len(records) * args.train_ratio)
    train_records = records[:n_train]
    test_records = records[n_train:]

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
    ds = datasets.load_dataset("parquet", data_files=args.output.replace(".parquet", "_train.parquet"))["train"]
    for i in range(min(3, len(ds))):
        sample = ds[i]
        print(f"\nSample {i}:")
        print(f"  prompt:   {json.dumps(sample['prompt'], ensure_ascii=False)[:200]}...")
        print(f"  audios:   {sample['audios']}")
        print(f"  answer:   {sample['answer']}")
        print(f"  pronoun:  {sample['pronoun']}")
        print(f"  context:  {sample['context'][:80]}...")

    # ---- simulate verl RLHFDataset._extract_audio_info ----
    print("\n--- Simulate verl _extract_audio_info ---")
    for i in range(min(3, len(ds))):
        sample = ds[i]
        audios = []
        for message in sample["prompt"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                if "audio" in item:
                    audios.append(item["audio"])
                elif "audio_url" in item:
                    audios.append(item["audio_url"])
                else:
                    audios.append({k: v for k, v in item.items() if k != "type"})
        print(f"  extracted audios: {audios}")


if __name__ == "__main__":
    main()
