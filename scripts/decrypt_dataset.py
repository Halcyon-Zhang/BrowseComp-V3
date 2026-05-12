#!/usr/bin/env python3
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Decrypt BrowseComp-V3 train.jsonl")
    parser.add_argument("--input", default="data/raw/data/train.jsonl")
    parser.add_argument("--key-file", default="key.txt")
    parser.add_argument("--output-dir", default="data/samples")
    parser.add_argument("--copy-images", default="data/raw")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cmd = [sys.executable, "scripts/decrypt_batch.py", "--input", args.input, "--key-file", args.key_file, "--output-dir", args.output_dir, "--copy-images", args.copy_images]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
