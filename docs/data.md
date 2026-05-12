# Data Download And Processing

## Download

The dataset is hosted on Hugging Face as `Halcyon-Zhang/BrowseComp-V3`. Download it into `data/raw/`:

```bash
python scripts/download_dataset.py --output-dir data/raw
```

You can also download manually from Hugging Face. The expected raw layout is:

```text
data/raw/
└── data/
    ├── train.jsonl
    └── images/
```

## Decryption Key

Use the public release passphrase:

```text
A_Visual_Vertical_Verifiable_Benchmark_for_Multimodal_Browsing_Agents
```

For convenience, store it in a local file. The passphrase is public, but generated local files such as `key.txt` are ignored so they do not clutter release commits:

```bash
printf '%s\n' 'A_Visual_Vertical_Verifiable_Benchmark_for_Multimodal_Browsing_Agents' > key.txt
```

## Decrypt Per-Sample Files

For a small smoke test:

```bash
python scripts/decrypt_batch.py \
  --input data/raw/data/train.jsonl \
  --key-file key.txt \
  --output-dir data/samples \
  --copy-images data/raw \
  --limit 2
```

For the full dataset, remove `--limit`.

The output layout is:

```text
data/samples/
├── <sample-id>.json
└── images/
    └── <copied-image-file>
```

Each decrypted sample contains `question`, `answer`, image references, metadata, and optional sub-goals. The `--copy-images` option copies referenced assets into `data/samples/images/` and rewrites image references to relative paths such as `images/<file>.jpg`.

## Convenience Wrapper

`scripts/decrypt_dataset.py` uses the same defaults as the full per-sample command above:

```bash
python scripts/decrypt_dataset.py \
  --input data/raw/data/train.jsonl \
  --key-file key.txt \
  --output-dir data/samples \
  --copy-images data/raw
```

Generated decrypted files are ignored by git.
