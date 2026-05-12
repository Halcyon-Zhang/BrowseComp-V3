# Setup And Running

## Environment

The default release does not require torch. Use one conda environment and one requirements file:

```bash
conda create -n browsecomp-v3 python=3.11
conda activate browsecomp-v3
pip install -r requirements.txt
```

## Keys

Copy `.env.template` to `.env` and fill in the values you need. The baseline, evaluator, and OmniSeeker runner all load `.env` from the repository root.

Baseline rollout needs an OpenAI-compatible chat completions endpoint:

```bash
export OPENAI_API_KEY='replace-with-your-key'
export OPENAI_API_BASE='https://api.openai.com/v1'  # optional for OpenAI; required for compatible providers
```

OmniSeeker search tools also need search and reader keys:

```bash
export SERPER_API_KEY='replace-with-serper-key'
export JINA_API_KEY='replace-with-jina-key'
```

Reverse image search needs a public image URL. If the input is already an HTTP(S) image URL, OmniSeeker can send it directly to Serper Lens. If the input is a local image, a base64 image, or a cropped image produced by `CropImage`, OmniSeeker uploads it first and then uses the returned public URL.

The built-in uploader uses Cloudflare R2:

```bash
export SEARCH_STORAGE_MODE='cloud'
export CLOUDFLARE_R2_ACCOUNT_ID='replace-with-account-id'
export CLOUDFLARE_R2_ACCESS_KEY_ID='replace-with-access-key-id'
export CLOUDFLARE_R2_SECRET_ACCESS_KEY='replace-with-secret-access-key'
export CLOUDFLARE_R2_BUCKET_NAME='replace-with-bucket-name'
export CLOUDFLARE_R2_PUBLIC_DOMAIN='https://your-public-domain.example.com'
# optional, useful to isolate runs
export CLOUDFLARE_R2_PREFIX='browsecomp-v3'
```

The bucket or public domain must serve uploaded objects publicly; otherwise reverse image search cannot fetch them.

Local logs and cache files are generated under ignored paths by default:

```text
omniseeker/logs/
omniseeker/logs/image_downloads/
.cache/search_cache.db
```

## Baseline Rollout

Dry-run first to verify data and image resolution without calling the model:

```bash
python examples/baseline_rollout.py \
  --data_dir data/samples \
  --data_root data/samples \
  --output_dir results/baseline \
  --model_name gpt-4o \
  --dry_run
```

Run a real baseline after configuring `OPENAI_API_KEY`:

```bash
python examples/baseline_rollout.py \
  --data_dir data/samples \
  --data_root data/samples \
  --output_dir results/baseline \
  --model_name gpt-4o
```

Evaluate and summarize baseline results:

```bash
python examples/eval_rollout_results.py \
  --input_dir results/baseline/gpt-4o \
  --judge_model gpt-4o

python scripts/summarize_eval_scores.py results/baseline/gpt-4o/eval_gpt-4o.jsonl
```

## OmniSeeker MCP Agent

Check command construction without starting the MCP server or calling the model:

```bash
bash omniseeker/run_bench.sh \
  --provider custom \
  --model-name gpt-4o \
  --data-dir data/samples \
  --example 1 \
  --dry-run
```

Run one real sample after configuring model and search keys:

```bash
bash omniseeker/run_bench.sh \
  --provider custom \
  --model-name gpt-4o \
  --data-dir data/samples \
  --example 1 \
  --num-rollouts 1 \
  --max-turns 20
```

The default gateway config is `omniseeker/configs/gateway_searchtools.example.json`. Copy and edit it if you need a different port or tool group.

## Local Verification

```bash
python -m py_compile \
  scripts/decrypt_batch.py \
  scripts/decrypt_dataset.py \
  examples/baseline_rollout.py \
  examples/eval_rollout_results.py \
  omniseeker/src/run_rollout.py \
  omniseeker/src/run_parallel_rollout.py

python -m pytest tests
```
