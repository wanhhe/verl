#!/usr/bin/env python
# coding=utf-8
"""Download audio files referenced by a context-RL manifest from Aliyun OSS.

Reads a JSONL manifest whose ``audio_path`` field holds an OSS object key
(e.g. ``primus/biz_datasets/59/external/TTS/.../RECOGNIZE_ORIGINAL/1.wav``)
and downloads each object into ``--cache-root`` preserving the key's directory
structure, so the local path is deterministic: ``<cache-root>/<oss-key>``.
``context_asr.py`` maps the manifest to the same local layout.

Credentials are read from the environment (same as test_oss.py):
    AccessKeyId, AccessKeySecret, Endpoint, Bucket
loaded from ``.env`` in the working directory or ``~/.env``.  Region defaults
to ``cn-zhangjiakou`` (override with ``OSS_REGION``).

Usage (run with the oss2 environment, Python 3.6):
    /quark_speech_nas_zjk/users/wangzilin/miniconda3/envs/oss2/bin/python \
        examples/data_preprocess/download_oss_audio.py \
        --manifest data/context_rl/manifest_20k_band_remove_same_homo/rl_train.jsonl

Non-empty files that are already present are skipped, so re-running is a cheap
resume.  Try ``--max_samples N`` / ``--check_only`` on a small subset first.
"""

import argparse
import concurrent.futures
import json
import os
import sys

import oss2
from dotenv import load_dotenv

DEFAULT_REGION = "cn-zhangjiakou"


def load_credentials():
    """Load .env from cwd first, then ~/.env (never overrides existing env vars)."""
    load_dotenv()
    load_dotenv(os.path.expanduser("~/.env"))
    required = ("AccessKeyId", "AccessKeySecret", "Endpoint", "Bucket")
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        sys.exit("Missing OSS credentials (env or .env): %s" % ", ".join(missing))


def read_audio_keys(manifest_path):
    """Extract unique OSS object keys from the manifest's audio_path field."""
    keys = []
    seen = set()
    skipped = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            key = rec.get("audio_path") or ""
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
            elif not key:
                skipped += 1
    print("Manifest: %d unique audio keys (%d lines skipped)" % (len(keys), skipped))
    return keys


def main():
    parser = argparse.ArgumentParser(description="Download context-RL manifest audio from OSS")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the JSONL manifest (rl_train.jsonl / rl_heldout.jsonl)",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Local audio cache dir; defaults to <manifest-dir>/../audio_cache "
        "(mirrors the OSS key structure)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download threads (default: 8)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit keys to download (-1 = all)")
    parser.add_argument(
        "--check_only",
        action="store_true",
        default=False,
        help="Only check object existence on OSS, do not download",
    )
    args = parser.parse_args()

    load_credentials()
    if args.cache_root is None:
        args.cache_root = os.path.abspath(os.path.join(os.path.dirname(args.manifest), "..", "audio_cache"))
    keys = read_audio_keys(args.manifest)
    if args.max_samples > 0:
        keys = keys[: args.max_samples]

    auth = oss2.Auth(os.getenv("AccessKeyId"), os.getenv("AccessKeySecret"))
    bucket = oss2.Bucket(
        auth,
        os.getenv("Endpoint"),
        os.getenv("Bucket"),
        region=os.getenv("OSS_REGION", DEFAULT_REGION),
    )

    # keep only keys that are not already cached
    pending = []
    already = 0
    for key in keys:
        local = os.path.join(args.cache_root, key)
        if os.path.exists(local) and os.path.getsize(local) > 0:
            already += 1
            continue
        pending.append((key, local))
    print("Already cached: %d, to download: %d (cache-root: %s)" % (already, len(pending), args.cache_root))

    def download_one(pair):
        key, local = pair
        try:
            if args.check_only:
                return key, "exists" if bucket.object_exists(key) else "MISSING-ON-OSS"
            os.makedirs(os.path.dirname(local), exist_ok=True)
            bucket.get_object_to_file(key, local)
            if os.path.getsize(local) <= 0:
                os.remove(local)
                return key, "FAILED(empty-download)"
            return key, "ok"
        except Exception as exc:  # per-file error: keep going, report at the end
            if os.path.exists(local):
                try:
                    os.remove(local)
                except OSError:
                    pass
            return key, "FAILED(%s)" % exc

    ok = 0
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_one, pair) for pair in pending]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            key, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                failed.append((key, status))
            if i % 500 == 0:
                print("  %d/%d done" % (i, len(pending)))

    print("\nDone: %d downloaded, %d already cached, %d failed" % (ok, already, len(failed)))
    for key, status in failed[:20]:
        print("  %s -> %s" % (key, status))
    if len(failed) > 20:
        print("  ... and %d more" % (len(failed) - 20))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
