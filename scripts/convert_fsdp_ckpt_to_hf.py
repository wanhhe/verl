#!/usr/bin/env python
# coding=utf-8
"""Convert a verl FSDP checkpoint (per-rank DTensor shards) to an official HF checkpoint dir.

The per-rank shards saved by verl are plain DTensors sharded on dim 0
(placements=(Shard(dim=0),)). This script merges them on CPU without needing
an FSDP context, then writes config.json + tokenizer + model.safetensors.

Usage (single process, no torchrun):
    python scripts/convert_fsdp_ckpt_to_hf.py \
        --ckpt  <global_step_60/actor> \
        --model <original HF model dir> \
        --output <output HF dir>
"""
import argparse
import glob
import os
import shutil

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint dir containing model_world_size_*_rank_*.pt")
    parser.add_argument("--model", required=True, help="original HF model dir (config/tokenizer source)")
    parser.add_argument("--output", required=True, help="output HF checkpoint dir")
    args = parser.parse_args()

    # 1. discover shard files and world size
    shard_files = sorted(glob.glob(os.path.join(args.ckpt, "model_world_size_*_rank_*.pt")))
    assert shard_files, f"no model_world_size_*_rank_*.pt found in {args.ckpt}"
    world_size = len(shard_files)
    print(f"==> found {world_size} shards in {args.ckpt}")

    # 2. merge shards (each value is a DTensor sharded on dim 0)
    shards = [torch.load(f, map_location="cpu", weights_only=False) for f in shard_files]
    keys = list(shards[0].keys())
    print(f"==> merging {len(keys)} parameters")

    full_state = {}
    for key in keys:
        tensors = []
        for shard in shards:
            v = shard[key]
            if isinstance(v, torch.Tensor) and hasattr(v, "placements"):
                local = v.to_local()  # (shard, ...) on dim 0
            elif isinstance(v, torch.Tensor):
                local = v
            else:
                raise TypeError(f"unexpected value type for {key}: {type(v)}")
            tensors.append(local.contiguous())
        merged = torch.cat(tensors, dim=0)
        # sanity check against the DTensor's global shape
        global_shape = tuple(shards[0][key].shape)
        assert tuple(merged.shape) == global_shape, (
            f"{key}: merged {tuple(merged.shape)} != global {global_shape}"
        )
        full_state[key] = merged
        if len(full_state) <= 3:
            print(f"  {key}: {tuple(merged.shape)}")

    # 3. write the official HF checkpoint
    os.makedirs(args.output, exist_ok=True)
    # config + tokenizer: prefer the checkpoint's huggingface/ subdir, fall back to the original model dir
    hf_cfg_dir = os.path.join(args.ckpt, "huggingface")
    src_dir = hf_cfg_dir if os.path.isdir(hf_cfg_dir) else args.model
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if os.path.isfile(src) and not name.startswith("model"):
            shutil.copy2(src, os.path.join(args.output, name))

    # weights
    from safetensors.torch import save_file

    save_file({k: v.to(torch.bfloat16) for k, v in full_state.items()}, os.path.join(args.output, "model.safetensors"))
    print(f"==> saved weights to {args.output}/model.safetensors")

    # 4. quick verification: load with transformers and check a couple of tensors
    from transformers import AutoConfig, AutoModel

    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )
    from transformers import AutoProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

    model = AutoModel.from_pretrained(args.output, torch_dtype=torch.bfloat16, trust_remote_code=True)
    sd = model.state_dict()
    for key in list(full_state.keys())[:3]:
        got = sd[key]
        assert got.shape == full_state[key].shape, f"{key}: {tuple(got.shape)} != {tuple(full_state[key].shape)}"
    print("==> verification OK: transformers loaded the converted checkpoint")
    print(f"==> done: {args.output}")


if __name__ == "__main__":
    main()
