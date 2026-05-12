"""
工具调用统计模块
用于统计每个 task 调用工具的成功与失败情况，并生成报告
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock
import traceback

logger = logging.getLogger("ToolStats")


@dataclass
class ToolCallRecord:
    """单次工具调用记录"""
    tool_name: str
    task_id: str
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    args: Optional[Dict[str, Any]] = None


class ToolStatsCollector:
    """工具调用统计收集器"""

    def __init__(self, output_dir: str = "tool_stats", enable_realtime_save: bool = True, save_interval: int = 10):
        """
        初始化统计收集器

        Args:
            output_dir: 输出目录
            enable_realtime_save: 是否启用实时保存（每次调用后追加到文件）
            save_interval: 实时保存的间隔（每N次调用保存一次），默认10次
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 统计数据结构
        self._records: List[ToolCallRecord] = []
        self._task_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "total_calls": 0,
            "success_calls": 0,
            "failed_calls": 0
        })
        self._tool_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "total_calls": 0,
            "success_calls": 0,
            "failed_calls": 0
        })

        # 线程安全锁
        self._lock = Lock()

        # 实时保存配置
        self.enable_realtime_save = enable_realtime_save
        self.save_interval = save_interval
        self._call_count = 0

        # 待写入的记录缓冲区（锁外写入）
        self._pending_writes: List[ToolCallRecord] = []

        # 实时记录文件（追加模式）
        self._realtime_log_file = self.output_dir / "realtime_calls.jsonl"

        # 如果启用实时保存，创建/清空日志文件（锁外操作）
        if self.enable_realtime_save:
            try:
                # 检查是否存在旧的实时日志
                if self._realtime_log_file.exists():
                    # 备份旧文件
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = self.output_dir / f"realtime_calls_backup_{timestamp}.jsonl"
                    self._realtime_log_file.rename(backup_file)
                    logger.info(f"Backed up previous realtime log to: {backup_file}")

                # 创建新的实时日志文件
                self._realtime_log_file.touch()
                logger.info(f"Realtime logging enabled: {self._realtime_log_file}")
            except Exception as e:
                logger.error(f"Failed to initialize realtime log file: {e}")
                self.enable_realtime_save = False

        logger.info(f"ToolStatsCollector initialized with output directory: {self.output_dir}")

    def record_call(
        self,
        tool_name: str,
        task_id: str,
        success: bool,
        error_message: Optional[str] = None,
        duration_ms: Optional[float] = None,
        args: Optional[Dict[str, Any]] = None
    ):
        """记录一次工具调用"""
        # 准备需要写入的记录（锁外）
        records_to_write = []
        should_write = False

        with self._lock:
            # 创建记录
            record = ToolCallRecord(
                tool_name=tool_name,
                task_id=task_id,
                timestamp=datetime.now().isoformat(),
                success=success,
                error_message=error_message,
                duration_ms=duration_ms,
                args=args
            )
            self._records.append(record)

            # 更新任务级别统计
            task_stat = self._task_stats[task_id]
            task_stat["total_calls"] += 1
            if success:
                task_stat["success_calls"] += 1
            else:
                task_stat["failed_calls"] += 1

            # 更新工具级别统计
            tool_stat = self._tool_stats[tool_name]
            tool_stat["total_calls"] += 1
            if success:
                tool_stat["success_calls"] += 1
            else:
                tool_stat["failed_calls"] += 1

            # 记录日志
            status = "✓" if success else "✗"
            duration_str = f"Duration: {duration_ms:.2f}ms" if duration_ms else "Duration: N/A"
            logger.info(
                f"{status} Tool Call | Task: {task_id} | Tool: {tool_name} | {duration_str}"
            )
            if not success and error_message:
                logger.error(f"  Error: {error_message}")

            # 添加到待写入缓冲区
            self._call_count += 1
            if self.enable_realtime_save:
                self._pending_writes.append(record)
                # 达到保存间隔时，准备批量写入
                if self._call_count % self.save_interval == 0:
                    records_to_write = self._pending_writes.copy()
                    self._pending_writes.clear()
                    should_write = True

        # 锁外执行文件 I/O（不阻塞其他线程）
        if should_write and records_to_write:
            self._batch_append_to_realtime_log(records_to_write)

    def _append_to_realtime_log(self, record: ToolCallRecord):
        """追加单条记录到实时日志文件（JSONL 格式）- 已废弃，使用批量写入"""
        self._batch_append_to_realtime_log([record])

    def _batch_append_to_realtime_log(self, records: List[ToolCallRecord]):
        """批量追加记录到实时日志文件（JSONL 格式）- 锁外执行"""
        if not records:
            return

        try:
            # 批量写入，减少文件打开次数
            with open(self._realtime_log_file, 'a', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
            logger.debug(f"Batch wrote {len(records)} records to realtime log")
        except Exception as e:
            logger.error(f"Failed to batch append to realtime log: {e}")
            # 文件写入失败不应影响主流程，只记录错误

    def get_task_report(self, task_id: str) -> Dict[str, Any]:
        """获取单个任务的统计报告"""
        with self._lock:
            if task_id not in self._task_stats:
                return {"error": f"Task {task_id} not found"}

            stats = self._task_stats[task_id]
            success_rate = (
                stats["success_calls"] / stats["total_calls"] * 100
                if stats["total_calls"] > 0 else 0
            )

            # 获取该任务的所有工具调用
            task_records = [r for r in self._records if r.task_id == task_id]
            tool_breakdown = defaultdict(lambda: {"success": 0, "failed": 0})

            for record in task_records:
                if record.success:
                    tool_breakdown[record.tool_name]["success"] += 1
                else:
                    tool_breakdown[record.tool_name]["failed"] += 1

            return {
                "task_id": task_id,
                "total_calls": stats["total_calls"],
                "success_calls": stats["success_calls"],
                "failed_calls": stats["failed_calls"],
                "success_rate": round(success_rate, 2),
                "tool_breakdown": dict(tool_breakdown)
            }

    def get_all_tasks_report(self) -> Dict[str, Any]:
        """获取所有任务的汇总报告"""
        with self._lock:
            tasks_report = []
            total_calls = 0
            total_success = 0
            total_failed = 0

            for task_id in self._task_stats.keys():
                task_report = self.get_task_report(task_id)
                tasks_report.append(task_report)
                total_calls += task_report["total_calls"]
                total_success += task_report["success_calls"]
                total_failed += task_report["failed_calls"]

            overall_success_rate = (
                total_success / total_calls * 100 if total_calls > 0 else 0
            )

            return {
                "summary": {
                    "total_tasks": len(self._task_stats),
                    "total_calls": total_calls,
                    "total_success": total_success,
                    "total_failed": total_failed,
                    "overall_success_rate": round(overall_success_rate, 2)
                },
                "tasks": tasks_report
            }

    def get_tool_report(self) -> Dict[str, Any]:
        """获取工具级别的统计报告"""
        with self._lock:
            tools_report = []

            for tool_name, stats in self._tool_stats.items():
                success_rate = (
                    stats["success_calls"] / stats["total_calls"] * 100
                    if stats["total_calls"] > 0 else 0
                )

                tools_report.append({
                    "tool_name": tool_name,
                    "total_calls": stats["total_calls"],
                    "success_calls": stats["success_calls"],
                    "failed_calls": stats["failed_calls"],
                    "success_rate": round(success_rate, 2)
                })

            # 按调用次数排序
            tools_report.sort(key=lambda x: x["total_calls"], reverse=True)

            return {
                "total_tools": len(tools_report),
                "tools": tools_report
            }

    def get_failed_calls(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取失败的工具调用记录"""
        with self._lock:
            failed_records = [
                asdict(r) for r in self._records
                if not r.success and (task_id is None or r.task_id == task_id)
            ]
            return failed_records

    def export_report(self, filename: Optional[str] = None) -> str:
        """导出完整报告到 JSON 文件"""
        # 先刷新所有待写入的记录到文件
        self._flush_pending_writes()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tool_stats_report_{timestamp}.json"

        filepath = self.output_dir / filename

        report = {
            "generated_at": datetime.now().isoformat(),
            "all_tasks_report": self.get_all_tasks_report(),
            "tool_report": self.get_tool_report(),
            "failed_calls": self.get_failed_calls(),
            "detailed_records": [asdict(r) for r in self._records]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report exported to: {filepath}")
        return str(filepath)

    def _flush_pending_writes(self):
        """刷新所有待写入的记录到文件"""
        records_to_write = []
        with self._lock:
            if self._pending_writes:
                records_to_write = self._pending_writes.copy()
                self._pending_writes.clear()

        if records_to_write:
            self._batch_append_to_realtime_log(records_to_write)
            logger.info(f"Flushed {len(records_to_write)} pending records to realtime log")

    def print_summary(self):
        """打印统计摘要到控制台"""
        all_tasks = self.get_all_tasks_report()
        tool_report = self.get_tool_report()

        print("\n" + "="*60)
        print("📊 Tool Call Statistics Summary")
        print("="*60)

        summary = all_tasks["summary"]
        print(f"\n🎯 Overall Statistics:")
        print(f"  Total Tasks: {summary['total_tasks']}")
        print(f"  Total Calls: {summary['total_calls']}")
        print(f"  Success: {summary['total_success']} ({summary['overall_success_rate']}%)")
        print(f"  Failed: {summary['total_failed']}")

        print(f"\n🔧 Tool Statistics (Top 10):")
        for tool in tool_report["tools"][:10]:
            print(f"  {tool['tool_name']}:")
            print(f"    Calls: {tool['total_calls']}, "
                  f"Success Rate: {tool['success_rate']}%")

        print(f"\n📋 Task Breakdown:")
        for task in all_tasks["tasks"][:10]:  # 只显示前10个任务
            print(f"  {task['task_id']}:")
            print(f"    Calls: {task['total_calls']}, "
                  f"Success Rate: {task['success_rate']}%")

        print("\n" + "="*60 + "\n")

    def load_from_realtime_log(self):
        """从实时日志文件中恢复数据"""
        if not self._realtime_log_file.exists():
            logger.warning(f"Realtime log file not found: {self._realtime_log_file}")
            return 0

        loaded_count = 0
        with self._lock:
            try:
                with open(self._realtime_log_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record_dict = json.loads(line)
                            record = ToolCallRecord(**record_dict)

                            # 重建内存数据结构
                            self._records.append(record)

                            # 更新任务统计
                            task_stat = self._task_stats[record.task_id]
                            task_stat["total_calls"] += 1
                            if record.success:
                                task_stat["success_calls"] += 1
                            else:
                                task_stat["failed_calls"] += 1

                            # 更新工具统计
                            tool_stat = self._tool_stats[record.tool_name]
                            tool_stat["total_calls"] += 1
                            if record.success:
                                tool_stat["success_calls"] += 1
                            else:
                                tool_stat["failed_calls"] += 1

                            loaded_count += 1

                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse line {line_num} in realtime log: {e}")
                        except Exception as e:
                            logger.error(f"Failed to process line {line_num} in realtime log: {e}")

                logger.info(f"Loaded {loaded_count} records from realtime log")

            except Exception as e:
                logger.error(f"Failed to load realtime log: {e}")

        return loaded_count


# 全局单例
_global_collector: Optional[ToolStatsCollector] = None


def get_stats_collector(output_dir: str = "tool_stats") -> ToolStatsCollector:
    """获取全局统计收集器单例"""
    global _global_collector
    if _global_collector is None:
        _global_collector = ToolStatsCollector(output_dir)
    return _global_collector


def reset_stats_collector():
    """重置全局统计收集器"""
    global _global_collector
    _global_collector = None
