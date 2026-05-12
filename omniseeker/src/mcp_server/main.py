import argparse
import json
import logging
import os
import sys
import time
import signal
import atexit
from typing import List, Callable, Any, Optional
from functools import wraps

# 确保 src 目录在路径中
cwd = os.getcwd()
sys.path.append(cwd)
if os.path.join(cwd, "src") not in sys.path:
    sys.path.append(os.path.join(cwd, "src"))

from mcp.server.fastmcp import FastMCP
from mcp_server.core.registry import ToolRegistry
from mcp_server.core.tool_stats import get_stats_collector
from mcp_server.core.fastmcp_patch import apply_fastmcp_patch

# 配置日志：必须为 root 设为 INFO，否则子 logger（如 text_search 的 jina_reader.* INFO）
# 在 propagate 到 root 时会被 WARNING 阈值丢弃，导致 MCP 日志里看不到 Jina 请求/响应细节。
_LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
try:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, force=True)
except TypeError:
    # Python < 3.8 无 force 参数
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("GatewayServer")

# 应用 FastMCP 补丁以修复 Pydantic v2 兼容性问题
apply_fastmcp_patch()

# 全局统计收集器
stats_collector = None


def create_tool_wrapper(func: Callable, task_id_extractor: Optional[Callable] = None) -> Callable:
    """
    创建工具函数包装器，用于统计调用情况

    Args:
        func: 原始工具函数
        task_id_extractor: 从参数中提取 task_id 的函数，如果为 None 则使用 'unknown'

    Returns:
        包装后的函数
    """
    import asyncio
    import inspect

    # 检查函数是否为协程函数
    is_async = inspect.iscoroutinefunction(func)

    if is_async:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            global stats_collector

            # 提取 task_id
            task_id = "unknown"
            if task_id_extractor:
                try:
                    task_id = task_id_extractor(*args, **kwargs)
                except Exception:
                    pass
            elif "task_id" in kwargs:
                task_id = kwargs.get("task_id", "unknown")

            # 记录开始时间
            start_time = time.time()
            success = False
            error_message = None

            try:
                # 调用原始异步函数
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_message = f"{type(e).__name__}: {str(e)}"
                
                raise
            finally:
                # 计算耗时
                duration_ms = (time.time() - start_time) * 1000

                # 记录统计信息
                if stats_collector:
                    stats_collector.record_call(
                        tool_name=func.__name__,
                        task_id=task_id,
                        success=success,
                        error_message=error_message,
                        duration_ms=duration_ms,
                        args={"args": args, "kwargs": kwargs} if not success else None
                    )

        return async_wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            global stats_collector

            # 提取 task_id
            task_id = "unknown"
            if task_id_extractor:
                try:
                    task_id = task_id_extractor(*args, **kwargs)
                except Exception:
                    pass
            elif "task_id" in kwargs:
                task_id = kwargs.get("task_id", "unknown")

            # 记录开始时间
            start_time = time.time()
            success = False
            error_message = None

            try:
                # 调用原始函数
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_message = f"{type(e).__name__}: {str(e)}"
                raise
            finally:
                # 计算耗时
                duration_ms = (time.time() - start_time) * 1000

                # 记录统计信息
                if stats_collector:
                    stats_collector.record_call(
                        tool_name=func.__name__,
                        task_id=task_id,
                        success=success,
                        error_message=error_message,
                        duration_ms=duration_ms,
                        args={"args": args, "kwargs": kwargs} if not success else None
                    )

        return wrapper

def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    global stats_collector

    parser = argparse.ArgumentParser(description="Unified MCP Gateway Server")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the server configuration file")
    parser.add_argument("--port", type=int, help="Override port defined in config")
    parser.add_argument("--enable-stats", action="store_true", default=False, help="Enable tool call statistics (disabled by default)")
    parser.add_argument("--stats-dir", type=str, default="tool_stats", help="Directory for statistics output")
    args = parser.parse_args()

    # 初始化统计收集器
    if args.enable_stats:
        stats_collector = get_stats_collector(args.stats_dir)
        logger.info(f"📊 Tool statistics enabled, output directory: {args.stats_dir}")

        # 注册退出时的清理函数
        def cleanup_and_report():
            if stats_collector:
                logger.info("\n🔄 Generating final statistics report...")
                try:
                    report_path = stats_collector.export_report()
                    stats_collector.print_summary()
                    logger.info(f"✅ Final report saved to: {report_path}")
                except Exception as e:
                    logger.error(f"Failed to generate final report: {e}")

        atexit.register(cleanup_and_report)
        signal.signal(signal.SIGINT, lambda s, f: (cleanup_and_report(), sys.exit(0)))
        signal.signal(signal.SIGTERM, lambda s, f: (cleanup_and_report(), sys.exit(0)))

    # 1. 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # 如果没有配置文件，提供一个默认的回退配置或退出
        logger.warning("Using default empty config")
        config = {"server_name": "Default Gateway", "modules": []}

    server_name = config.get("server_name", "Unified Gateway")
    
    # 2. 初始化 FastMCP Server
    # 注意: dependencies 参数用于管理生命周期，如果工具需要共享状态，可以在这里传入
    mcp = FastMCP(server_name)
    
    logger.info(f"🚀 Initializing {server_name}...")

    # [新增步骤] 自动发现工具
    # 扫描当前包下的所有文件，触发 @register_tool
    ToolRegistry.autodiscover("mcp_server") 

    # 3. 动态注册工具
    registered_tools_count = 0
    modules = config.get("modules", [])

    for module in modules:
        r_type = module.get("resource_type", "unknown")

        logger.info(f"Loading module: Type={r_type}")

        # 通过注册表获取该模块对应的所有 Python 函数
        tool_functions = ToolRegistry.get_tools_by_config(module)

        for func in tool_functions:
            try:
                # 如果启用了统计，包装工具函数
                if args.enable_stats:
                    wrapped_func = create_tool_wrapper(func)
                else:
                    wrapped_func = func

                # 将函数注册为 MCP Tool
                # FastMCP 会自动解析函数的 type hints 和 docstrings 作为工具描述
                mcp.tool()(wrapped_func)
                logger.info(f"  + Registered tool: {func.__name__}")
                registered_tools_count += 1
            except Exception as e:
                logger.error(f"  - Failed to register tool {func.__name__}: {e}")

    logger.info(f"✅ Total tools registered: {registered_tools_count}")

    # 4. 启动服务器
    port = args.port if args.port else config.get("port", 8080)
    host = config.get("host", "0.0.0.0")
    debug = config.get("debug", False)

    logger.info(f"Starting SSE server on {host}:{port}")
    mcp.settings.host = host
    mcp.settings.port = port
    # 使用 SSE 传输模式 (与 Claude Desktop 等客户端兼容性最好)
    mcp.run(transport='sse')

if __name__ == "__main__":
    main()