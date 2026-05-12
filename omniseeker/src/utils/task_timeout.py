# src/utils/task_timeout.py
"""
任务超时监控工具
实现第2层超时：任务执行超时（Task Level Timeout）
"""
import os
import time
import signal
import logging
import threading
from typing import Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class TaskTimeoutError(Exception):
    """任务超时异常"""
    pass


class TaskTimeoutMonitor:
    """
    任务超时监控器

    使用线程计时器实现任务级别的超时控制，当任务执行时间超过限制时：
    1. 记录超时日志
    2. 抛出TaskTimeoutError异常
    3. 由调用方负责清理资源
    """

    def __init__(self, timeout: float, task_id: str, worker_id: str):
        """
        初始化超时监控器

        Args:
            timeout: 超时时间（秒）
            task_id: 任务ID
            worker_id: Worker ID
        """
        self.timeout = timeout
        self.task_id = task_id
        self.worker_id = worker_id
        self.timer: Optional[threading.Timer] = None
        self.start_time: float = 0
        self.is_cancelled = False

    def _timeout_handler(self):
        """超时处理函数"""
        if not self.is_cancelled:
            elapsed = time.time() - self.start_time
            error_msg = (
                f"⏰ [TaskTimeout] Worker={self.worker_id} Task={self.task_id} "
                f"exceeded timeout limit ({self.timeout}s). Elapsed: {elapsed:.1f}s"
            )
            logger.error(error_msg)
            # 注意：这里不能直接中断主线程，只能记录日志
            # 实际的超时处理由run_task_with_timeout装饰器完成

    def start(self):
        """启动超时监控"""
        self.start_time = time.time()
        self.is_cancelled = False
        self.timer = threading.Timer(self.timeout, self._timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        logger.info(
            f"⏱️  [TaskTimeout] Started monitoring for Worker={self.worker_id} "
            f"Task={self.task_id}, timeout={self.timeout}s"
        )

    def cancel(self):
        """取消超时监控（任务正常完成时调用）"""
        self.is_cancelled = True
        if self.timer:
            self.timer.cancel()
            elapsed = time.time() - self.start_time
            logger.info(
                f"✅ [TaskTimeout] Task completed in {elapsed:.1f}s "
                f"(Worker={self.worker_id}, Task={self.task_id})"
            )

    def __enter__(self):
        """支持with语句"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句，自动取消监控"""
        self.cancel()
        return False


def run_task_with_timeout(timeout: Optional[float] = None):
    """
    任务超时装饰器

    用法:
        @run_task_with_timeout(timeout=600)
        def run_task(self, task, agent_config, logger):
            # 任务执行逻辑
            pass

    Args:
        timeout: 超时时间（秒），如果为None则从环境变量读取

    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, task, agent_config, logger, *args, **kwargs):
            # 获取超时配置
            actual_timeout = timeout
            if actual_timeout is None:
                actual_timeout = float(
                    agent_config.get("task_timeout",
                    os.environ.get("TASK_EXECUTION_TIMEOUT", "600"))
                )

            task_id = task.get("id", "unknown")
            worker_id = getattr(self, "worker_id", "unknown")

            # 创建超时监控器
            monitor = TaskTimeoutMonitor(actual_timeout, task_id, worker_id)

            # 记录开始时间
            start_time = time.time()

            try:
                # 启动监控
                monitor.start()

                # 执行原始函数
                result = func(self, task, agent_config, logger, *args, **kwargs)

                # 检查是否超时
                elapsed = time.time() - start_time
                if elapsed > actual_timeout:
                    raise TaskTimeoutError(
                        f"Task {task_id} exceeded timeout ({actual_timeout}s), "
                        f"elapsed: {elapsed:.1f}s"
                    )

                return result

            except TaskTimeoutError:
                # 超时异常，向上抛出
                logger.error(
                    f"❌ [TaskTimeout] Task {task_id} timeout after {actual_timeout}s, "
                    "resource will be released"
                )
                raise

            except Exception as e:
                # 其他异常
                elapsed = time.time() - start_time
                logger.error(
                    f"❌ [TaskError] Task {task_id} failed after {elapsed:.1f}s: {e}"
                )
                raise

            finally:
                # 取消监控
                monitor.cancel()

        return wrapper
    return decorator


def check_execution_timeout(start_time: float, timeout: float, task_id: str, worker_id: str) -> bool:
    """
    检查任务是否超时（手动检查模式）

    用于在任务执行过程中定期检查是否超时，适合长时间运行的循环任务。

    Args:
        start_time: 任务开始时间戳
        timeout: 超时时间（秒）
        task_id: 任务ID
        worker_id: Worker ID

    Returns:
        True表示已超时，False表示未超时

    示例:
        start_time = time.time()
        while turn < max_turns:
            if check_execution_timeout(start_time, timeout, task_id, worker_id):
                raise TaskTimeoutError("Task timeout")
            # 继续执行任务
    """
    elapsed = time.time() - start_time
    if elapsed > timeout:
        logger.warning(
            f"⏰ [TaskTimeout] Worker={worker_id} Task={task_id} "
            f"timeout check failed: {elapsed:.1f}s > {timeout}s"
        )
        return True
    return False
