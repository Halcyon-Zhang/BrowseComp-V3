#!/usr/bin/env python3
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download BrowseComp-V3 from Hugging Face")
    parser.add_argument("--repo-id", default="Halcyon-Zhang/BrowseComp-V3")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=str(out), local_dir_use_symlinks=False)
    print(path)


if __name__ == "__main__":
    main()
