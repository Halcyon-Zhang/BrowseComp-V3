#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BrowseComp-V3 Parallel Rollout Runner (MCP Native)

This script mirrors src/run_parallel_rollout.py but specializes in
loading BrowseComp-style datasets (new JSON structure with images, sub_goals, metadata)
and running the same parallel rollout + evaluation pipeline.

Key features preserved from run_parallel_rollout.py:
- Parallel worker processes
- Environment factory usage (MCP Http env by default)
- Metrics computation and score export
- Trajectory saving

Usage examples:
  python omniseeker/src/run_rollout.py \
    --data_dir data/samples \
    --num_rollouts 2 \
    --output_dir results


  # Or point to a single file
  python omniseeker/src/run_rollout.py --data_path data/samples/001_....json
"""

import json
import os
import sys
import glob
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ensure this 'src' directory is importable for `import benchmark`
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from benchmark import Benchmark
from envs.factory import get_environment_class

try:
    from run_parallel_rollout import run_parallel_rollout, ParallelRolloutConfig
except ModuleNotFoundError as exc:
    if exc.name != "mcp":
        raise

    run_parallel_rollout = None

    @dataclass
    class ParallelRolloutConfig:
        """Small local config used when MCP dependencies are not installed."""

        num_rollouts: int = 1
        env_mode: str = "http_mcp"
        env_kwargs: Dict[str, Any] = field(default_factory=dict)
        agent_config_dict: Dict[str, Any] = field(default_factory=dict)
        output_dir: str = "results"


def run_rollout_sequential(config: ParallelRolloutConfig, benchmark: Benchmark):
    """Fallback sequential runner when multiprocessing is unavailable."""
    import logging
    import json as _json

    logger = logging.getLogger("rollout_seq")
    logger.setLevel(logging.INFO)

    EnvClass = get_environment_class(config.env_mode)
    logger.info(f"[SEQ] Using environment class: {EnvClass.__name__}")

    environment = EnvClass(parallel_degree=1, **config.env_kwargs)
    env_start = getattr(environment, "env_start", None)
    if callable(env_start):
        try:
            environment.env_start()
        except Exception:
            pass

    agent_config = dict(config.agent_config_dict)
    agent_config["output_dir"] = config.output_dir

    results = []
    for item in benchmark.get_items():
        task = {
            "id": item.id,
            "question": item.question,
            "answer": item.answer,
            "metadata": item.metadata or {},
        }
        res = environment.run_task(task, agent_config, logger)
        results.append(res)

    # Save trajectories
    os.makedirs(config.output_dir, exist_ok=True)
    traj_path = os.path.join(config.output_dir, "trajectory.jsonl")
    with open(traj_path, "w", encoding="utf-8") as f:
        for r in results:
            _json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"[SEQ] Saved trajectories to {traj_path}")

    # Evaluate
    predictions = {
        r["task_id"]: r.get("answer", "") for r in results if r.get("success")
    }
    evaluation_cfg = agent_config.get("evaluation_metric", "exact_match")
    metrics = (
        [evaluation_cfg] if isinstance(evaluation_cfg, str) else list(evaluation_cfg)
    )

    stats = {}
    for m in metrics:
        metric_results = benchmark.evaluate(predictions, metric=m, concurrent=False)
        total = len(metric_results)
        avg = sum(r.score for r in metric_results) / total if total else 0.0
        stats[m] = {
            "total_items": total,
            "average_score": avg,
        }
    summary_path = os.path.join(config.output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        _json.dump(
            {"metrics": stats, "benchmark": benchmark.get_summary()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[SEQ] Saved evaluation summary to {summary_path}")


def load_benchmark_from_dir(
    data_dir: str, pattern: str = "*.json"
) -> Benchmark:
    """Aggregate all benchmark JSON files in a directory into a single Benchmark."""
    files: List[str] = sorted(glob.glob(os.path.join(data_dir, pattern)))
    agg = Benchmark(name="BrowseComp-V3 Benchmark", description=f"Aggregated from {data_dir}")
    count = 0
    for f in files:
        try:
            b = Benchmark(data_path=f)
            if b.items:
                agg.items.extend(b.items)
                count += len(b.items)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            continue
    print(f"Loaded {count} item(s) from {len(files)} file(s) under {data_dir}")
    return agg


def load_benchmark(
    data_dir: str = None, data_path: str = None, pattern: str = "*.json"
) -> Benchmark:
    """Load benchmark either from a directory (aggregate) or a single file."""
    if data_dir:
        return load_benchmark_from_dir(data_dir, pattern)
    if data_path:
        return Benchmark(data_path=data_path, name="BrowseComp-V3 Benchmark")
    raise ValueError("Must provide either data_dir or data_path")


def _safe_name(val: str) -> str:
    """Make a string filesystem-safe."""
    return "".join([c if c.isalnum() or c in "-_." else "_" for c in str(val)])


def _extract_domain_d1(metadata) -> str:
    """Extract domain d1 from metadata; default to 'unknown'."""
    if not isinstance(metadata, dict):
        return "unknown"
    dom = metadata.get("domain") or {}
    if isinstance(dom, dict):
        return str(dom.get("d1") or "unknown")
    return "unknown"
'''
def filter_completed_tasks(
    benchmark: Benchmark, output_dir: str, model_name: str
) -> int:
    """
    Remove items whose rollout result already exists to avoid duplicate execution.
    Path convention: results/<model>/<domain>/<qid>.json (domain uses d1).
    Rules:
      - no file: run
      - file.status == success: skip
      - file.status != success (failed/timeout/unknown/read error): re-run
    """
    if not benchmark.items:
        return 0

    safe_model = _safe_name(model_name)
    filtered = []
    skipped_success = 0
    retry_failed = 0
    for item in benchmark.items:
        qid = getattr(item, "id", None)
        meta = getattr(item, "metadata", None)
        domain = _safe_name(_extract_domain_d1(meta))

        base_dir_name = os.path.basename(os.path.normpath(output_dir))
        if base_dir_name == safe_model:
            base_path = output_dir
        else:
            base_path = os.path.join(output_dir, safe_model)

        task_dir = os.path.join(base_path, domain)
        task_path = os.path.join(task_dir, f"{_safe_name(qid)}.json")

        if qid and os.path.exists(task_path):
            status = None
            try:
                with open(task_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                status = (
                    data.get("execution", {}).get("status")
                    or ("success" if data.get("success") else None)
                )
            except Exception:
                # 读失败当作需要重跑
                status = None

            if status == "success":
                skipped_success += 1
                continue
            else:
                retry_failed += 1

        filtered.append(item)

    benchmark.items = filtered
    if skipped_success:
        print(
            f"[INFO] Skipped {skipped_success} tasks already successful under {base_path}"
        )
    if retry_failed:
        print(f"[INFO] Re-queued {retry_failed} failed/unknown tasks for retry")
    return skipped_success
'''

def filter_completed_tasks(
    benchmark: Benchmark,
    output_dir: str,
    model_name: str,
    dataset: Optional[str] = None,
) -> int:
    """
    Remove items whose rollout result already exists to avoid duplicate execution.
    Path convention (与 http_mcp_env.get_task_output_dir 一致):
      results/<model>/<dataset>/<qid>.json
      若未传 dataset，则为 results/<model>/<qid>.json（兼容旧目录）。
    Rules:
      - no file: run
      - file.status == success: skip
      - file.status != success (failed/timeout/unknown/read error): re-run
    """
    if not benchmark.items:
        return 0

    safe_model = _safe_name(model_name)
    safe_dataset = _safe_name(dataset) if dataset else None
    filtered = []
    skipped_success = 0
    retry_failed = 0
    for item in benchmark.items:
        qid = getattr(item, "id", None)

        base_dir_name = os.path.basename(os.path.normpath(output_dir))
        if base_dir_name == safe_model:
            base_path = output_dir
        else:
            base_path = os.path.join(output_dir, safe_model)

        if safe_dataset:
            base_path = os.path.join(base_path, safe_dataset)

        task_path = os.path.join(base_path, f"{_safe_name(qid)}.json")

        if qid and os.path.exists(task_path):
            status = None
            try:
                with open(task_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                status = (
                    data.get("execution", {}).get("status")
                    or ("success" if data.get("success") else None)
                )
            except Exception:
                # 读失败当作需要重跑
                status = None

            if status == "success":
                skipped_success += 1
                continue
            else:
                retry_failed += 1

        filtered.append(item)

    benchmark.items = filtered
    if skipped_success:
        print(
            f"[INFO] Skipped {skipped_success} tasks already successful under {base_path}"
        )
    if retry_failed:
        print(f"[INFO] Re-queued {retry_failed} failed/unknown tasks for retry")
    return skipped_success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Science dataset parallel rollout (MCP Native)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(SRC_DIR, "data", "Science"),
        help="Directory containing Science JSON files",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional single JSON file to run instead of a directory",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files in data_dir",
    )

    parser.add_argument(
        "--num_rollouts", type=int, default=5, help="Number of parallel workers"
    )
    parser.add_argument(
        "--env_mode",
        type=str,
        default="http_mcp",
        help="Environment mode (for example: http_mcp or http_mcp_search)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results)",
    )

    # MCP / Gateway config
    parser.add_argument(
        "--mcp_server_url",
        type=str,
        default="http://localhost:8080",
        help="MCP Server URL",
    )
    parser.add_argument(
        "--gateway_config_path",
        type=str,
        default="gateway_config.json",
        help="Path to gateway config file",
    )

    # Agent config
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-5-mini",
        help="Agent model name",
    )
    parser.add_argument("--max_turns", type=int, default=20, help="Max turns per task")
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="generic",
        choices=["generic", "no_tool"],
        help="Prompt variant for compatible environments",
    )
    parser.add_argument(
        "--evaluation_metric",
        type=str,
        nargs="+",
        default=["exact_match"],
        help="Evaluation metric(s) to use. Can specify multiple: --evaluation_metric exact_match f1_score",
    )

    parser.add_argument(
        "--tool_choice",
        type=str,
        default="auto",
        help='Default tool selection mode for the API (e.g. "auto" or "required").',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name used for result path grouping (e.g. test, Science)",
    )
    parser.add_argument(
        "--example",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N samples from the loaded benchmark (by file load order). Omit to run all.",
    )

    args = parser.parse_args()

    # Default output goes to results; environments create model/domain subdirectories.
    output_dir = args.output_dir if args.output_dir is not None else "results"
    dataset_name = (
        args.dataset
        if args.dataset is not None
        else os.path.basename(os.path.normpath(args.data_dir))
    )

    # Load benchmark from directory or single file
    if args.data_path:
        benchmark = load_benchmark(data_path=args.data_path)
    else:
        benchmark = load_benchmark(data_dir=args.data_dir, pattern=args.pattern)

    if args.example is not None:
        if args.example <= 0:
            print("[WARN] --example must be a positive integer; ignoring, using all loaded items.")
        else:
            total_loaded_items = len(benchmark.items)
            benchmark.items = benchmark.items[: args.example]
            print(
                f"[INFO] Limited to first {len(benchmark.items)} samples (--example={args.example}, "
                f"had {total_loaded_items} after load)"
            )

    # Remove already-completed tasks to avoid duplicate rollout
    total_loaded = len(benchmark.items)
    skipped = filter_completed_tasks(
        benchmark, output_dir, args.model_name, dataset=dataset_name
    )
    remaining = len(benchmark.items)
    print(
        f"[INFO] Tasks loaded: {total_loaded}, skipped (already done): {skipped}, to run: {remaining}"
    )
    if remaining == 0:
        print("[INFO] No tasks to run after de-duplication. Exiting.")
        sys.exit(0)

    # Prepare environment kwargs (same as run_parallel_rollout)
    env_kwargs = {
        "observation_type": "screenshot_a11y_tree",
        "mcp_server_url": args.mcp_server_url,
        "gateway_config_path": args.gateway_config_path,
        "prompt_type": args.prompt_type,
        "tool_choice": args.tool_choice,
    }

    config = ParallelRolloutConfig(
        num_rollouts=args.num_rollouts,
        env_mode=args.env_mode,
        output_dir=output_dir,
        env_kwargs=env_kwargs,
        agent_config_dict={
            "model_name": args.model_name,
            "dataset": dataset_name,
            "evaluation_metric": (
                args.evaluation_metric
                if len(args.evaluation_metric) > 1
                else args.evaluation_metric[0]
            ),
            "max_turns": args.max_turns,
            "max_retries": 2,
        },
    )

    # Run with the same parallel rollout logic, fallback to sequential when
    # multiprocessing or optional MCP dependencies are unavailable.
    if run_parallel_rollout is None:
        print("[WARN] MCP dependencies unavailable. Falling back to sequential rollout.")
        run_rollout_sequential(config, benchmark)
        print("Science sequential rollout completed successfully")
    else:
        try:
            run_parallel_rollout(config, benchmark)
            print("Science parallel rollout completed successfully")
        except Exception as e:
            print(f"[WARN] Parallel rollout unavailable ({e}). Falling back to sequential.")
            run_rollout_sequential(config, benchmark)
            print("Science sequential rollout completed successfully")
