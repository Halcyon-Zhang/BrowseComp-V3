# src/run_parallel_rollout.py
# -*- coding: utf-8 -*-
"""
并行 Rollout 框架 - MCP 纯净版
已移除所有本地重资源管理器（Resource Manager）的遗留逻辑。
完全依赖 MCP 协议与远程环境（Gateway/Server）进行资源交互。
"""
import time
import os
import sys
import json
import logging
import signal
from datetime import datetime
from multiprocessing import Manager, Process
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional dotenv support
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

from benchmark import Benchmark
from envs.factory import get_environment_class
from envs.http_mcp_env import _task_session_id

# 导入超时异常
from utils.task_timeout import TaskTimeoutError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 预加载环境变量
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if HAS_DOTENV and os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    logger.info(f"Loaded environment variables from {ENV_PATH}")


def _register_main_signal_handlers():
    """
    注册主进程信号处理
    仅负责日志记录和优雅退出，不再负责清理全局资源对象（已移除）。
    """
    def handle_signal(signum, frame):
        logger.info(f"Main process received signal {signum}. Exiting...")
        # 在这里可以添加其他必要的轻量级清理
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


@dataclass
class ParallelRolloutConfig:
    """并行 Rollout 配置"""
    num_rollouts: int = 5          # 并行度（Worker 数量）
    env_mode: str = "http_mcp"     # 默认为 MCP 模式
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    agent_config_dict: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results"


def run_parallel_rollout(
    config: ParallelRolloutConfig,
    benchmark: Benchmark
):
    """
    运行并行 Rollout 框架 (MCP 纯净版)
    """
    # [新增] 1. 开始计时
    benchmark_start_time = time.time()

    logger.info("=" * 60)
    logger.info("Starting Parallel Rollout Framework (MCP Native)")
    logger.info(f"Rollouts: {config.num_rollouts} | Env: {config.env_mode} | Items: {len(benchmark.get_items())}")
    logger.info("=" * 60)

    # 为本次运行生成唯一前缀，避免跨脚本/实例的 worker_id 冲突
    run_id = f"rollout_{int(time.time())}_{os.getpid()}"
    logger.info(f"Run ID: {run_id}")
    
    # 1. 获取环境类
    EnvClass = get_environment_class(config.env_mode)
    logger.info(f"Using environment class: {EnvClass.__name__}")
    
    # [变更] 不再调用 setup_global_resources
    # MCP 模式下，资源由 Gateway/Server 管理，Client 端无需初始化全局池。

    _register_main_signal_handlers()
    
    # 2. 创建跨进程共享的数据结构
    with Manager() as manager:
        task_queue = manager.Queue()
        shared_results = manager.list()
        worker_instance_map = manager.dict()
        worker_instance_events = manager.list()
        
        # 将所有基准测试项放入任务队列
        for item in benchmark.get_items():
            task_dict = {
                "id": item.id,
                "question": item.question,
                "answer": item.answer,
                "metadata": item.metadata or {}
            }
            task_queue.put(task_dict)

        # 添加哨兵值 (Poison Pill)
        for _ in range(config.num_rollouts):
            task_queue.put(None)
        
        # 3. 启动 Worker 进程
        env_class_name = EnvClass.__module__ + "." + EnvClass.__name__
        processes = []
        for i in range(config.num_rollouts):
            worker_id = f"{run_id}_w{i+1:02d}"
            
            proc = Process(
                target=run_rollout_worker,
                args=(
                    worker_id,
                    task_queue,
                    env_class_name,
                    config.env_kwargs,
                    config.agent_config_dict,
                    config.output_dir,
                    config.num_rollouts,
                    shared_results,
                    worker_instance_map,
                    worker_instance_events,
                )
            )
            proc.start()
            processes.append(proc)
            logger.info(f"Started worker process: {worker_id}")
        
        # 等待所有 Worker 进程执行完毕
        try:
            for proc in processes:
                proc.join()
        except KeyboardInterrupt:
            logger.info("Main process interrupted. Terminating workers...")
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
        
        # [新增] 2. 结束计时并计算时长
        benchmark_end_time = time.time()
        total_duration_seconds = benchmark_end_time - benchmark_start_time
        
        # [新增] 3. 格式化时间显示 (HH:MM:SS)
        m, s = divmod(total_duration_seconds, 60)
        h, m = divmod(m, 60)
        duration_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

        # 4. 收集结果并评测
        results = list(shared_results)
        
        logger.info(f"All workers completed. Total results: {len(results)}")
        
        predictions = {
            result["task_id"]: result.get("answer", "")
            for result in results
            if result.get("success", False)
        }

        # 支持多个评分指标：可以是单个指标字符串或指标列表
        evaluation_config = config.agent_config_dict.get("evaluation_metric", "exact_match")
        if isinstance(evaluation_config, str):
            evaluation_metrics = [evaluation_config]
        else:
            evaluation_metrics = list(evaluation_config)

        # 对每个指标分别进行评测
        all_benchmark_results = {}  # {metric_name: [EvaluationResult]}
        metrics_statistics = {}     # {metric_name: {total, successful, failed, avg_score, ...}}

        logger.info(f"Evaluating with {len(evaluation_metrics)} metric(s): {', '.join(evaluation_metrics)}")

        for metric in evaluation_metrics:
            logger.info(f"  Computing metric: {metric}...")
            metric_results = benchmark.evaluate(
                predictions=predictions,
                metric=metric
            )
            all_benchmark_results[metric] = metric_results

            # 计算该指标的统计信息
            total_items = len(metric_results)
            successful_items = len([r for r in metric_results if r.score > 0])
            failed_items = total_items - successful_items
            avg_score = sum(r.score for r in metric_results) / total_items if total_items > 0 else 0.0

            metrics_statistics[metric] = {
                "total_items": total_items,
                "successful_items": successful_items,
                "failed_items": failed_items,
                "average_score": avg_score,
                "success_rate": successful_items / total_items if total_items > 0 else 0.0
            }

        # 输出所有指标的评测结果
        logger.info("=" * 60)
        logger.info("Benchmark Evaluation Results")
        logger.info(f"  Metrics: {', '.join(evaluation_metrics)}")
        logger.info(f"  Total Tasks: {len(results)}")
        logger.info(f"  Successful Predictions: {len(predictions)}")
        logger.info("")
        for metric, stats in metrics_statistics.items():
            logger.info(f"  [{metric}]")
            logger.info(f"    Total Items: {stats['total_items']}")
            logger.info(f"    Successful: {stats['successful_items']}")
            logger.info(f"    Failed: {stats['failed_items']}")
            logger.info(f"    Average Score: {stats['average_score']:.4f}")
            logger.info(f"    Success Rate: {stats['success_rate']:.2%}")
        logger.info("=" * 60)

        # [新增] 4. 在日志中输出总耗时（更显眼的格式）
        logger.info("")
        logger.info("=" * 60)
        logger.info("⏱️  TOTAL EXECUTION TIME")
        logger.info(f"  Duration: {duration_str} ({total_duration_seconds:.2f}s)")
        logger.info(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(benchmark_start_time))}")
        logger.info(f"  End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(benchmark_end_time))}")
        logger.info("=" * 60)
        
        # 不再保存多余的聚合文件，直接返回结果
        return {
            "worker_results": results,
            "benchmark_evaluation": all_benchmark_results,  # 现在包含所有指标的结果
            "metrics_statistics": metrics_statistics,       # 添加统计信息
            "total_duration": total_duration_seconds
        }


def get_active_resource_configs(environment, task_item):
    """
    根据环境实际启用的资源类型筛选任务中的资源配置
    """
    raw_configs = task_item.get("metadata", {}).get("resource_configs", {})
    active_types = getattr(environment, "active_resources", [])
    
    active_configs = {}
    for res_type in active_types:
        if res_type in raw_configs:
            active_configs[res_type] = raw_configs[res_type]
            
    return active_configs


def run_rollout_worker(
    worker_id: str,
    task_queue,
    env_class_name: str,
    env_kwargs: Dict[str, Any],
    agent_config_dict: Dict[str, Any],
    output_dir: str,
    parallel_degree: int, # [变更] 移除了 global_resources 参数
    shared_results,
    worker_instance_map=None,
    worker_instance_events=None,
):
    """
    Rollout Worker 进程函数 (MCP 纯净版)
    """
    logger = logging.getLogger(f"worker.{worker_id}")
    environment = None

    def worker_signal_handler(signum, frame):
        logger.info(f"Worker {worker_id} received signal {signum}. Cleaning up...")
        try:
            if environment:
                # 尝试调用环境的 cleanup 或 env_close
                cleanup_fn = getattr(environment, "cleanup", None)
                if callable(cleanup_fn):
                    cleanup_fn(worker_id)
                else:
                    env_close = getattr(environment, "env_close", None)
                    if callable(env_close):
                        env_close()
        except Exception as e:
            logger.error(f"Error during signal cleanup: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, worker_signal_handler)
    signal.signal(signal.SIGTERM, worker_signal_handler)

    try:
        # 1. 动态导入环境类
        module_name, class_name = env_class_name.rsplit(".", 1)
        env_module = __import__(module_name, fromlist=[class_name])
        EnvClass = getattr(env_module, class_name)
        
        # 2. 注入 worker_id
        local_env_kwargs = env_kwargs.copy()
        local_env_kwargs["worker_id"] = worker_id

        # 3. 初始化环境实例
        # [变更] 彻底移除 resource_manager 参数
        environment = EnvClass(
            parallel_degree=parallel_degree,
            **local_env_kwargs, 
        )
        
        # 4. 启动环境 (建立 MCP 连接等)
        env_start = getattr(environment, "env_start", None)
        if callable(env_start):
            try:
                environment.env_start()
            except Exception as exc:
                logger.warning(f"Worker {worker_id} env_start() failed: {exc}")

        # 调用可选的 init 方法
        init_fn = getattr(environment, "init", None)
        if callable(init_fn):
            init_fn()

        logger.info(f"Worker {worker_id} started")

        # 5. 检查环境功能特性
        task_config_fn = getattr(environment, "initialize_with_task_config", None)
        env_supports_task_config = callable(task_config_fn)
        
        allocate_fn = getattr(environment, "allocate_resource", None)
        release_fn = getattr(environment, "release_resource", None)
        
        # 检查是否需要每任务资源分配 (MCP 模式下的资源隔离)
        env_has_heavy_resource = bool(getattr(environment, "has_heavy_resource", False) and callable(allocate_fn))

        # 6. 准备 Agent 配置
        agent_config = dict(agent_config_dict)
        agent_config["output_dir"] = output_dir

        # 7. 主任务循环
        while True:
            try:
                task = task_queue.get()
                if task is None: # 哨兵值
                    logger.info(f"Worker {worker_id} received sentinel. Stopping loop.")
                    break
            except Exception as e:
                logger.error(f"Worker {worker_id} error getting task: {e}")
                break

            task_id = task.get("id", "unknown")
            resource_allocated = False
            current_resource_id = None
            task_start_time = time.time()
            logger.info(f"▶️ Worker {worker_id} START Task {task_id}")

            # 为每个任务生成并设置逻辑会话 ID，底层环境会自动注入给相关工具
            tsid = f"{worker_id}_rollout_{task_id}_{int(time.time() * 1000)}"
            _task_session_id.set(tsid)

            try:
                # 补充 metadata
                if "metadata" not in task or not isinstance(task.get("metadata"), dict):
                    task["metadata"] = task.get("metadata") or {}
                
                metadata = task.get("metadata", {})
                for key in ("config", "evaluator"):
                    if key not in task and key in metadata:
                        task[key] = metadata[key]

                # 环境特定配置
                if env_supports_task_config:
                    task_env_config = (
                        task.get("env_config")
                        or task.get("metadata", {}).get("env_config")
                    )
                    if task_env_config:
                        task_config_fn(task_env_config)

                # [资源分配]
                # 在 MCP 模式下，这里实际上是调用 allocate_fn (即 env.allocate_resource)
                # The environment may prepare MCP-managed resources before task execution.
                if env_has_heavy_resource:
                    logger.info(f"[worker {worker_id}] requesting resource via MCP...")
                    
                    # 获取需要激活的资源配置
                    active_resource_configs = get_active_resource_configs(environment, task)
                    
                    # 调用环境的分配方法 (不再依赖本地 manager)
                    if not allocate_fn(worker_id, active_resource_configs):
                        raise RuntimeError("Failed to allocate resource via MCP")
                    
                    resource_allocated = True
                    
                    get_allocated_fn = getattr(environment, "get_allocated_resource_id", None)
                    current_resource_id = get_allocated_fn() if callable(get_allocated_fn) else None
                    logger.info(f"[worker {worker_id}] acquired resource context: {current_resource_id}")
                    
                    # 记录分配状态用于调试
                    if worker_instance_map is not None and current_resource_id:
                        worker_instance_map[worker_id] = {
                            "instance_id": current_resource_id,
                            "task_id": task_id,
                            "assigned_time": datetime.now().isoformat(),
                        }

                # 8. 执行任务 (Run Task)
                result = environment.run_task(task, agent_config, logger)

                if shared_results is not None:
                    shared_results.append(result)

                duration = time.time() - task_start_time
                status_icon = "✅" if result.get("success") else "❌"
                logger.info(f"{status_icon} Worker {worker_id} FINISH Task {task_id} in {duration:.1f}s")

            except TaskTimeoutError as e:
                logger.error(f"⏰ Task {task_id} timeout: {e}")
                if shared_results is not None:
                    shared_results.append({
                        "task_id": task_id,
                        "success": False,
                        "error": f"Timeout: {e}",
                        "messages": []
                    })
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                if shared_results is not None:
                    shared_results.append({
                        "task_id": task_id,
                        "success": False,
                        "error": str(e),
                        "messages": []
                    })
            finally:
                # [资源释放]
                # 任务结束（无论成功失败），释放远端资源
                if env_has_heavy_resource and resource_allocated and callable(release_fn):
                    logger.info(f"[worker {worker_id}] releasing resource via MCP...")
                    release_fn(worker_id, reset=True)
                    
                    if worker_instance_map is not None:
                        worker_instance_map.pop(worker_id, None)

    finally:
        # Worker 退出清理
        if worker_instance_map is not None:
            worker_instance_map.pop(worker_id, None)
        
        # 关闭环境连接
        env_close = getattr(environment, "env_close", None)
        if callable(env_close):
            env_close()
            
        logger.info(f"Worker {worker_id} stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parallel rollout (MCP Native)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to benchmark data file")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--env_mode", type=str, default="http_mcp", help="Environment mode")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    # MCP 相关配置
    parser.add_argument("--mcp_server_url", type=str, default="http://localhost:8080", help="MCP Server URL")
    parser.add_argument("--gateway_config_path", type=str, default="gateway_config.json", help="Path to gateway config file")
    
    # 额外配置 (Agent)
    parser.add_argument("--model_name", type=str, default="gpt-4.1-2025-04-14", help="Agent model name")
    parser.add_argument("--max_turns", type=int, default=15, help="Max turns per task")
    parser.add_argument("--prompt_type", type=str, default="generic",
                        choices=["generic", "no_tool"],
                        help="Prompt variant for compatible environments")
    parser.add_argument(
        "--evaluation_metric",
        type=str,
        nargs='+',
        default=["exact_match"],
        help="Evaluation metric(s) to use. Can specify multiple: --evaluation_metric exact_match f1_score"
    )

    args = parser.parse_args()
    
    benchmark = Benchmark(data_path=args.data_path)
    
    # 环境参数传递给 HttpMCPEnv
    env_kwargs = {
        "observation_type": "screenshot_a11y_tree",
        "mcp_server_url": args.mcp_server_url,
        "gateway_config_path": args.gateway_config_path,
        "prompt_type": args.prompt_type,
    }
    
    config = ParallelRolloutConfig(
        num_rollouts=args.num_rollouts,
        env_mode=args.env_mode,
        output_dir=args.output_dir,
        env_kwargs=env_kwargs,
        agent_config_dict={
            "model_name": args.model_name,
            "evaluation_metric": args.evaluation_metric if len(args.evaluation_metric) > 1 else args.evaluation_metric[0],
            "max_turns": args.max_turns,
            "max_retries": 2
        }
    )
    
    run_parallel_rollout(config, benchmark)
    logger.info("Parallel rollout completed successfully")
