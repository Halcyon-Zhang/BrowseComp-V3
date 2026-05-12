# src/envs/http_mcp_env.py
import sys
import os
import json
import logging
import asyncio
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Union, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextvars import ContextVar
from io import BytesIO
import base64

# 引入 MCP SDK
from mcp.types import CallToolResult
# 引入 MCP SSE 客户端
from utils.mcp_sse_client import MCPSSEClient

# 引入任务超时监控工具
from utils.task_timeout import TaskTimeoutMonitor, TaskTimeoutError, check_execution_timeout

# 引入 system prompt 函数
from prompts.system_prompts import get_system_prompt

import httpx
import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from PIL import Image

logger = logging.getLogger(__name__)

# [新增] 上下文变量：用于在协程间传递 task_session_id
_task_session_id: ContextVar[Optional[str]] = ContextVar('task_session_id', default=None)

@dataclass
class ToolMetadata:
    """用于适配 GenericTrajectorySampler 的简单工具包装类"""
    name: str
    description: str
    parameters: List[Dict[str, Any]]

class HttpMCPEnv:
    """
    配置驱动的 MCP 环境适配器 (MCP 纯净版)
    
    完全基于 Model Context Protocol (MCP) 与远程 Gateway/Server 交互。
    负责 Agent 执行循环、工具调用转发以及通过 MCP 工具进行资源生命周期管理。
    """
    
    # 开启重型资源模式，通知框架在 run_task 前后调用 allocate/release
    has_heavy_resource = True 

    def __init__(self,
                 model_name: str = "gpt-4.1-2025-04-14",
                 parallel_degree=1,
                 **kwargs):
        
        self.model_name = model_name
        self.config = kwargs
        
        # 工具元数据缓存
        self.tool_schemas: List[Dict[str, Any]] = []
        self.tool_descriptions: str = ""
        # [新增] 本地工具缓存，用于支持 get_tool
        self.local_tools: Dict[str, ToolMetadata] = {}

        # 1. 基础配置
        self.server_url = kwargs.get("mcp_server_url", "http://localhost:8080")
        
        # 2. 获取 worker_id
        if "worker_id" in kwargs:
            self.worker_id = kwargs["worker_id"]
        else:
            import multiprocessing
            self.worker_id = multiprocessing.current_process().name

        # 3. 实例化 MCP 客户端
        self.mcp_client = MCPSSEClient(f"{self.server_url}/sse")

        # 4. 加载 Gateway 配置 (确定需要申请哪些资源)
        config_path = kwargs.get("gateway_config_path", "gateway_config.json")
        self.modules_config = self._load_gateway_config(config_path)

        # [修复] 解析活动资源类型时，过滤掉不需要后端分配的 'system' 类型
        # 'system' 通常指代无状态的系统工具集，不需要向 Resource API 申请锁定
        self.active_resources = [
            m.get("resource_type")
            for m in self.modules_config.get("modules", [])
            if m.get("resource_type") and m.get("resource_type") != "system"
        ]
        
        # 初始化状态变量
        self.initial_observation = None
        self.allocated_resources = {}
        self._tools_initialized = False
        
        # [新增] 心跳后台线程池（避免阻塞主流程）
        self._heartbeat_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"heartbeat-{self.worker_id}")
        self._last_activity_time = time.time()  # 本地记录最后活动时间
        # 保存每个任务内的原始 chat completion 响应
        self._raw_chat_responses: List[Dict[str, Any]] = []

        # 5. 工具白名单（优先策略）
        # 优先使用传入的 tool_whitelist；否则尝试从环境变量 MCP_TOOL_WHITELIST 读取（逗号分隔）
        whitelist_arg = kwargs.get("tool_whitelist")
        if isinstance(whitelist_arg, (list, tuple, set)):
            self._tool_whitelist = {str(x).strip() for x in whitelist_arg if str(x).strip()}
        elif isinstance(whitelist_arg, str):
            self._tool_whitelist = {x.strip() for x in whitelist_arg.split(',') if x.strip()}
        else:
            env_wl = os.environ.get("MCP_TOOL_WHITELIST", "")
            self._tool_whitelist = {x.strip() for x in env_wl.split(',') if x.strip()} if env_wl else set()

        # 初始化持久事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        logger.info(f"HttpMCPEnv initialized: {self.worker_id} -> {self.server_url}, resources: {self.active_resources}")
        
        # 初始化远程工具列表
        self._initialize_tools()

    @property
    def mode(self) -> str:
        return "http_mcp"

    # =========================================================================
    # [新增/修复] 核心适配接口：get_tool 和 list_tools
    # =========================================================================

    def list_tools(self) -> List[str]:
        """返回所有可用工具的名称列表"""
        return list(self.local_tools.keys())

    def get_tool(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        获取工具对象（适配器模式）。
        GenericTrajectorySampler 需要访问 tool.name, tool.description, tool.parameters
        """
        return self.local_tools.get(tool_name)

    # =========================================================================
    # 核心 Agent 执行逻辑
    # =========================================================================

    def run_task(self, task: Dict[str, Any], agent_config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """
        执行完整的 Agent 任务循环
        """
        task_id = task.get("id", "unknown")
        question = task.get("question", "")
        start_time = datetime.utcnow()

        model_name = agent_config.get("model_name", self.model_name)
        max_turns = agent_config.get("max_turns", 3)
        max_retries = agent_config.get("max_retries", 3)

        # 获取任务超时配置
        task_timeout = float(
            agent_config.get("task_timeout",
            os.environ.get("TASK_EXECUTION_TIMEOUT", "600"))
        )

        task_output_dir = None
        if hasattr(self, "get_task_output_dir") and callable(self.get_task_output_dir):
            output_dir_cfg = agent_config.get("output_dir", "results")
            # dataset 来源优先级：
            # 1) agent_config["dataset"]（如果调用方显式传入）
            # 2) 环境变量 DATASET
            # 3) output_dir 最后一段目录名（例如 results/<model>/test -> test）
            dataset_name = (
                agent_config.get("dataset")
                or os.environ.get("DATASET")
                or os.path.basename(os.path.normpath(output_dir_cfg))
                or "unknown"
            )
            task_output_dir = self.get_task_output_dir(
                output_dir_cfg,
                task_id,
                model_name,
                dataset_name
            )

        monitor = TaskTimeoutMonitor(task_timeout, task_id, self.worker_id)
        messages: List[Dict[str, Any]] = []
        self._reset_raw_chat_responses()

        try:
            monitor.start()

            messages = self._run_conversation(
                question, model_name, max_turns, max_retries, logger,
                task_timeout=task_timeout,
                task_start_time=time.time()
            )

            final_answer = self._extract_final_answer(messages)

            end_time = datetime.utcnow()

            result = {
                "task_id": task_id,
                "question": question,
                "answer": final_answer,
                "messages": messages,
                "success": True,
                "error": None,
            }

            if task_output_dir:
                self._save_conversation_log(
                    task_output_dir,
                    task,
                    model_name,
                    messages,
                    result,
                    start_time,
                    end_time,
                )

            return result

        except TaskTimeoutError as e:
            logger.error(f"❌ [TaskTimeout] Task {task_id} timeout: {e}")
            end_time = datetime.utcnow()
            messages = getattr(e, "_partial_messages", messages)
            result = {
                "task_id": task_id,
                "question": question,
                "answer": "",
                "messages": [],
                "success": False,
                "error": f"Task execution timeout: {e}",
            }
            if messages:
                result["messages"] = messages
            if task_output_dir:
                self._save_conversation_log(
                    task_output_dir,
                    task,
                    model_name,
                    result.get("messages", []),
                    result,
                    start_time,
                    end_time,
                )
            return result

        except Exception as e:
            logger.error(f"❌ [TaskError] Task {task_id} failed: {e}")
            end_time = datetime.utcnow()
            messages = getattr(e, "_partial_messages", messages)
            result = {
                "task_id": task_id,
                "question": question,
                "answer": "",
                "messages": [],
                "success": False,
                "error": str(e),
            }
            if messages:
                result["messages"] = messages
            if task_output_dir:
                self._save_conversation_log(
                    task_output_dir,
                    task,
                    model_name,
                    result.get("messages", []),
                    result,
                    start_time,
                    end_time,
                )
            return result

        finally:
            monitor.cancel()

    def _reset_raw_chat_responses(self) -> None:
        """在任务开始时重置原始响应缓存。"""
        self._raw_chat_responses = []

    def _record_chat_response(
        self,
        response: Any,
        model_name: str,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录完整 API 响应结构，供任务结束后落盘。"""
        try:
            if hasattr(response, "model_dump"):
                response_payload = response.model_dump()
            else:
                response_payload = str(response)
            self._raw_chat_responses.append(
                {
                    "captured_at": datetime.utcnow().isoformat() + "Z",
                    "model": model_name,
                    "request_kwargs": request_kwargs or {},
                    "response": response_payload,
                }
            )
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Failed to record raw chat response: {e}")

    def _run_conversation(self,
                         question: str,
                         model_name: str,
                         max_turns: int,
                         max_retries: int,
                         logger: logging.Logger,
                         task_timeout: Optional[float] = None,
                         task_start_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        
        system_prompt = self.get_system_prompt(question)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
        ]

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"Question: {question}\n"}]

        # 注入初始观察
        initial_obs = getattr(self, "initial_observation", None)
        
        # === 减少日志：移除初始观察状态检查 ===
        # if initial_obs:
        #     logger.info(f"[{self.worker_id}] [LLM_INJECT_LOG] Initial Observation Status: Present for injection.")
        # else:
        #     logger.info(f"[{self.worker_id}] [LLM_INJECT_LOG] Initial Observation Status: Not present for injection.")

        if initial_obs and isinstance(initial_obs, dict):
            if initial_obs.get("screenshot"):
                user_content.append({
                    "type": "text",
                    "text": "Here is the initial screen state of the computer:"
                })
                compressed = self._compress_base64_image(initial_obs["screenshot"])
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{compressed}",
                        "detail": "high"
                    }
                })

            if initial_obs.get("accessibility_tree"):
                user_content.append({
                    "type": "text",
                    "text": f"Accessibility Tree:\n{initial_obs['accessibility_tree']}"
                })

        # 钩子：由子类决定是否以及如何注入任务输入图片
        self._append_input_images(user_content)

        messages.append({"role": "user", "content": user_content})

        # === 减少日志：移除用户消息内容检查 ===
        # safe_msg = self._truncate_data(messages[1], max_len=200)
        # logger.info(f"[{self.worker_id}] [LLM_INJECT_LOG] First User Message Content (Check Injection): {json.dumps(safe_msg, indent=2, ensure_ascii=False)}")

        client = self._get_openai_client()
        turn_count = 0

        try:
            while turn_count < max_turns:
                if task_timeout and task_start_time:
                    if check_execution_timeout(task_start_time, task_timeout, "current_task", self.worker_id):
                        raise TaskTimeoutError(
                            f"Task timeout after {time.time() - task_start_time:.1f}s "
                            f"(limit: {task_timeout}s) at turn {turn_count}"
                        )

                retry = 0
                while retry < max_retries:
                    try:
                        # 减少日志：仅在需要时输出
                        # logger.info(f"Turn {turn_count}: Calling LLM...")
                        request_kwargs: Dict[str, Any] = {}
                        tool_choice = self.config.get("tool_choice")
                        if tool_choice is not None:
                            request_kwargs["tool_choice"] = tool_choice
                        max_tool_calls = self.config.get("max_tool_calls")
                        if max_tool_calls is not None:
                            try:
                                request_kwargs["max_tool_calls"] = int(max_tool_calls)
                            except (ValueError, TypeError):
                                request_kwargs["max_tool_calls"] = max_tool_calls

                        tools = self.get_tool_schemas()
                        response = self._call_chat_completion(
                            client,
                            model_name,
                            messages,
                            tools,
                            request_kwargs,
                        )

                        if not hasattr(response, "choices") or not response.choices:
                            raise ValueError("OpenAI API returned empty response")

                        assistant_message = response.choices[0].message
                        messages.append(assistant_message.model_dump())

                        if assistant_message.tool_calls:
                            if messages[-1]['content'] is None:
                                 messages[-1]['content'] = ""

                            for idx, tool_call in enumerate(assistant_message.tool_calls):
                                tool_name = tool_call.function.name
                                tool_args = json.loads(tool_call.function.arguments)
                                logger.info(f"🔧 {tool_name}")

                                tool_output = None
                                tool_exception = None
                                try:
                                    if idx == 0:
                                        tool_output = self.execute_tool(tool_name, tool_args)
                                    else:
                                        tool_output = "[Skipped: only the first tool_call is executed]"
                                except Exception as exc:
                                    tool_exception = exc
                                    tool_output = f"[Error executing {tool_name}: {exc}]"
                                finally:
                                    if tool_output is None:
                                        tool_output = "[No output from tool]"

                                    if isinstance(tool_output, dict) and "images" in tool_output:
                                        content_str = tool_output.get("text", "")
                                        image_list = tool_output.get("images", [])
                                    else:
                                        content_str = str(tool_output)
                                        image_list = []

                                    messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "name": tool_name,
                                            "content": content_str,
                                        }
                                    )

                                    if image_list:
                                        image_msgs = self._build_tool_image_message(image_list)
                                        if image_msgs:
                                            messages.extend(image_msgs)

                                if tool_exception:
                                    raise tool_exception

                        else:
                            # 减少日志：最终答案产生时不再输出
                            # logger.info(f"Turn {turn_count}: final answer produced")
                            return messages
                        
                        break # 成功则跳出重试循环

                    except Exception as exc:
                        retry += 1
                        logger.warning(f"Retry {retry}/{max_retries} due to error: {exc}")
                        if retry >= max_retries:
                            raise
                turn_count += 1
                
            logger.warning("Max turns reached without final answer")
            return messages
        except TaskTimeoutError as e:
            setattr(e, "_partial_messages", messages)
            raise
        except Exception as e:
            setattr(e, "_partial_messages", messages)
            raise

    def _call_chat_completion(
        self,
        client: openai.OpenAI,
        model_name: str,
        messages: List[ChatCompletionMessageParam],
        tools: List[ChatCompletionToolParam],
        request_kwargs: Dict[str, Any],
    ):
        try:
            logger.info(
                f"[{self.worker_id}] _call_chat_completion request: "
                f"model={model_name}, message_count={len(messages)}, "
                f"tools_count={len(tools) if tools else 0}, "
                f"request_kwargs_keys={list(request_kwargs.keys())}, "
                f"tool_choice={request_kwargs.get('tool_choice', None)!r}, "
                f"max_tool_calls={request_kwargs.get('max_tool_calls', None)!r}"
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                **request_kwargs,
            )
            self._record_chat_response(response, model_name, request_kwargs)
            return response
        except TypeError as exc:
            msg = str(exc)
            unsupported_keys = []
            logger.warning(
                f"[{self.worker_id}] TypeError in _call_chat_completion: {msg}; "
                f"request_kwargs={request_kwargs}"
            )
            for key in ("max_tool_calls", "tool_choice"):
                if key in msg and key in request_kwargs:
                    unsupported_keys.append(key)

            if unsupported_keys:
                for key in unsupported_keys:
                    request_kwargs.pop(key, None)
                logger.warning(
                    f"[{self.worker_id}] Removed unsupported chat kwargs {unsupported_keys} ({msg})."
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    **request_kwargs,
                )
                self._record_chat_response(response, model_name, request_kwargs)
                return response
            raise

    def _extract_final_answer(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract final answer from messages.
        First tries to extract content within <FINAL_ANSWER> tags.
        Falls back to returning the last assistant message if tags not found.
        """
        if not messages:
            return None

        # Search for messages with FINAL_ANSWER tags (from newest to oldest)
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    # Try to extract answer from special tokens
                    match = re.search(r'<FINAL_ANSWER>(.*?)</FINAL_ANSWER>', content, re.DOTALL)
                    if match:
                        answer = match.group(1).strip()
                        if answer:
                            return answer

        # Fallback: return last assistant message if no tags found
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if content is not None:
                    return str(content)

        return None
    
    def _build_key_info_patterns(self, key_info: str) -> List[str]:
        """
        根据 key_info 生成匹配模式列表（优先级从高到低）
        
        示例：
        - "University of Michigan" -> ["university of michigan", "michigan"]
        - "4" -> ["4", "four"]
        - "Martha Schwartz" -> ["martha schwartz", "schwartz"]
        """
        patterns = []
        key_lower = key_info.lower().strip()
        
        if not key_lower:
            return patterns
        
        patterns.append(key_lower)
        
        if key_lower.isdigit():
            num = int(key_lower)
            number_words = {
                0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
                5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
                14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
                18: "eighteen", 19: "nineteen", 20: "twenty"
            }
            if num in number_words:
                patterns.append(number_words[num])
        
        words = key_lower.split()
        if len(words) > 1:
            stopwords = {"of", "the", "a", "an", "in", "at", "on", "for", "to"}
            for word in reversed(words):
                if word not in stopwords and len(word) > 2:
                    patterns.append(word)
                    break
        
        return patterns

    def _extract_sentence_containing(
        self, 
        text: str, 
        patterns: List[str],
        from_end: bool = True
    ) -> Optional[str]:
        """
        从文本中提取包含指定模式的句子
        
        Args:
            text: 要搜索的文本
            patterns: 匹配模式列表（按优先级排序）
            from_end: 是否从后往前搜索
        
        Returns:
            匹配到的句子，或 None
        """
        if not text or not patterns:
            return None
        
        sentence_pattern = r'(?<=[.!?。！？\n])\s*'
        sentences = re.split(sentence_pattern, text)
        
        if from_end:
            sentences = list(reversed(sentences))
        
        for pattern in patterns:
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if not sentence_clean:
                    continue
                if pattern in sentence_clean.lower():
                    return sentence_clean
        
        return None
    
    def _extract_subgoals_from_messages(
        self, 
        messages: List[Dict[str, Any]],
        ground_truth_subgoals: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        从助手消息中提取 subgoals
        
        策略：
        1. 首先提取所有 <SUBGOAL>...</SUBGOAL> 标签内容
        2. 如果提供了 ground_truth_subgoals，检查每个 key_info：
           - 若已在标签内容中包含，跳过
           - 若未包含，从后往前搜索 assistant 消息，提取包含该 key_info 的句子
        """
        subgoals: List[str] = []
        seen_sentences: Set[str] = set()
        
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            matches = re.findall(r"<SUBGOAL>(.*?)</SUBGOAL>", content, flags=re.DOTALL)
            for m in matches:
                text = m.strip()
                if text and text not in seen_sentences:
                    seen_sentences.add(text)
                    subgoals.append(text)
        
        if ground_truth_subgoals:
            assistant_contents = []
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue
                content = msg.get("content")
                if isinstance(content, str):
                    assistant_contents.append(content)
            
            for sg in ground_truth_subgoals:
                key_info = sg.get("key_info", "")
                if not key_info:
                    continue
                
                key_lower = key_info.lower()
                patterns = self._build_key_info_patterns(key_info)
                
                already_found = False
                for existing in subgoals:
                    existing_lower = existing.lower()
                    for pattern in patterns:
                        if pattern in existing_lower:
                            already_found = True
                            break
                    if already_found:
                        break
                
                if already_found:
                    continue
                
                for content in assistant_contents:
                    sentence = self._extract_sentence_containing(content, patterns, from_end=True)
                    if sentence and sentence not in seen_sentences:
                        seen_sentences.add(sentence)
                        subgoals.append(sentence)
                        break
        
        return subgoals

    def _build_tool_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """统计工具调用次数（按 assistant 消息里的 tool_calls）"""
        stats: Dict[str, Dict[str, int]] = {}
        total = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                name = None
                if isinstance(call, dict):
                    func = call.get("function")
                    if isinstance(func, dict):
                        name = func.get("name")
                if not name:
                    continue
                stats.setdefault(name, {"calls": 0, "ok": 0, "failed": 0})
                stats[name]["calls"] += 1
                stats[name]["ok"] += 1  # 未跟踪失败，默认为成功
                total += 1
        return {"by_tool": stats, "total_calls": total}

    def _safe_list(self, value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    def _safe_dict(self, value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _count_tool_results(self, text_payload: str) -> Optional[int]:
        """Best-effort count of structured results in tool text output."""
        if not text_payload:
            return None
        try:
            data = json.loads(text_payload)
        except Exception:
            return None
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            for key in ("results", "items", "data", "search_results", "output"):
                value = data.get(key)
                if isinstance(value, list):
                    return len(value)
        return None

    def _extract_ground_truth(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """提取 ground_truth（final_answer, sub_goals）"""
        metadata = self._safe_dict(task.get("metadata"))
        sub_goals = metadata.get("sub_goals") or task.get("sub_goals") or []
        return {
            "final_answer": task.get("answer", ""),
            "sub_goals": sub_goals if isinstance(sub_goals, list) else [],
        }

    def _extract_item_tags(self, task: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self._safe_dict(task.get("metadata"))
        return {
            "level": metadata.get("level"),
            "difficulty": metadata.get("difficulty"),
            "domain": metadata.get("domain"),
            "fc_num": metadata.get("fc_num"),
        }

    def _extract_images(self, task: Dict[str, Any]) -> List[Any]:
        metadata = self._safe_dict(task.get("metadata"))
        images = metadata.get("images") or task.get("images") or []
        return images if isinstance(images, list) else [images]

    def _extract_reasoning_contents_from_raw(self) -> List[Optional[str]]:
        """
        从已缓存的原始 chat completion 响应中提取每轮 assistant 的 reasoning 字段。
        返回顺序与 API 调用顺序一致，用于回填到 process.messages。
        """
        reasonings: List[Optional[str]] = []
        for record in self._raw_chat_responses:
            try:
                resp = record.get("response", {})
                choices = resp.get("choices", []) if isinstance(resp, dict) else []
                msg = {}
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message", {}) or {}
                if not isinstance(msg, dict):
                    reasonings.append(None)
                    continue

                reasoning = msg.get("reasoning_content")
                if reasoning is None:
                    # 兼容不同供应商的字段命名
                    reasoning = msg.get("reasoning")
                if reasoning is None:
                    reasoning = msg.get("thinking")

                if isinstance(reasoning, str) and reasoning.strip():
                    reasonings.append(reasoning)
                elif reasoning is None:
                    reasonings.append(None)
                else:
                    reasonings.append(str(reasoning))
            except Exception:
                reasonings.append(None)
        return reasonings

    def _save_conversation_log(self, output_dir, task: Dict[str, Any], model: str, messages: List[Dict[str, Any]], result: Dict[str, Any], start_time: datetime, end_time: datetime):
        import os
        import json
        try:
            os.makedirs(output_dir, exist_ok=True)
            task_id = task.get("id", "unknown")
            question = task.get("question", "")
            safe_task_id = "".join([c if c.isalnum() or c in "-_." else "_" for c in str(task_id)])
            file_path = os.path.join(output_dir, f"{safe_task_id}.json")
            assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")
            assistant_reasonings = self._extract_reasoning_contents_from_raw()
            assistant_idx = 0

            # 生成消息列表（带 msg_id 和时间戳）
            msg_list = []
            for idx, msg in enumerate(messages, start=1):
                ts = (start_time + timedelta(seconds=idx - 1)).isoformat() + "Z"
                msg_id = f"m_{idx:04d}"
                # 直接保存原始 content，以保留工具/图像信息
                entry = {
                    "msg_id": msg_id,
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "timestamp": ts
                }
                # 记录工具相关信息，便于事后复盘
                if msg.get("tool_calls"):
                    entry["tool_calls"] = msg.get("tool_calls")
                if msg.get("tool_call_id"):
                    entry["tool_call_id"] = msg.get("tool_call_id")
                if msg.get("name"):
                    entry["name"] = msg.get("name")

                # 将 raw response 中的 reasoning 回填到主轨迹文件
                if msg.get("role") == "assistant":
                    reasoning = (
                        assistant_reasonings[assistant_idx]
                        if assistant_idx < len(assistant_reasonings)
                        else None
                    )
                    if reasoning:
                        entry["reasoning_content"] = reasoning
                    assistant_idx += 1
                msg_list.append(entry)

            tool_stats = self._build_tool_stats(messages)
            ground_truth = self._extract_ground_truth(task)
            subgoals_pred = self._extract_subgoals_from_messages(
                messages, 
                ground_truth.get("sub_goals")
            )

            log_content = {
                "execution": {
                    "q_id": task_id,
                    "model_name": model,
                    "timestamp": {
                        "start": start_time.isoformat() + "Z",
                        "end": end_time.isoformat() + "Z"
                    },
                    "status": "success" if result.get("success") else "failed",
                    "output_file": file_path
                },
                "item": {
                    "question": question,
                    "images": self._extract_images(task),
                    "tags": self._extract_item_tags(task)
                },
                "ground_truth": ground_truth,
                "prediction": {
                    "final_answer": result.get("answer"),
                    "subgoals": subgoals_pred
                },
                "process": {
                    "messages": msg_list,
                    "tool_stats": tool_stats,
                    "raw_chat_responses_count": len(self._raw_chat_responses)
                }
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(log_content, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"[{self.worker_id}] ✅ Conversation log saved to: {file_path}")
        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Failed to save conversation log: {e}")

    # =========================================================================
    # OpenAI Client 管理
    # =========================================================================

    def _get_openai_client(self) -> openai.OpenAI:
        if not hasattr(self, '_openai_client') or self._openai_client is None:
            api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
            base_url = self.config.get("openai_api_url") or os.environ.get("OPENAI_API_BASE")
            # 单次 chat completion 总超时（秒）。默认 120s 足够让 LLM + 代理出结果；
            # 旧默认 1200s × (1+5 retries) = 7200s，会把 worker 挂死接近 2 小时，远超 task_timeout。
            timeout = float(self.config.get("openai_timeout", os.environ.get("OPENAI_TIMEOUT", "120")))
            # OpenAI 内置重试。默认 1 次，给上层 _call_chat_completion 的 app 级 retry 留快速兜底空间。
            max_retries = int(self.config.get("openai_max_retries", os.environ.get("OPENAI_MAX_RETRIES", "1")))

            # 显式构造一个带细粒度超时的 httpx.Client，避免单纯传 timeout 标量
            # 在通过 HTTP 代理（如 Mihomo/Clash）连接上游时，因 read 阶段无独立超时
            # 而长时间挂在 socket.recv 的情况。
            try:
                http_client = httpx.Client(
                    timeout=httpx.Timeout(
                        connect=15.0,
                        read=timeout,
                        write=30.0,
                        pool=15.0,
                    ),
                    # trust_env=True 默认会读 HTTP_PROXY/HTTPS_PROXY
                )
            except Exception as exc:
                logger.warning(
                    f"[{self.worker_id}] Failed to build custom httpx.Client for OpenAI: {exc}; "
                    f"falling back to default."
                )
                http_client = None

            openai.api_key = api_key
            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "timeout": timeout,
                "max_retries": max_retries,
            }
            if http_client is not None:
                kwargs["http_client"] = http_client
            if base_url:
                openai.base_url = base_url
                kwargs["base_url"] = base_url
            self._openai_client = openai.OpenAI(**kwargs)
        return self._openai_client

    def _compress_base64_image(
        self,
        img_b64: str,
        max_bytes: int = 1 * 1024 * 1024,
        max_side: int = 1024,
        qualities: Tuple[int, ...] = (85, 75, 65, 55),
    ) -> str:
        """
        将 base64 图片在发送前按需压缩到 1MB 限制以内。
        - 始终解码并按 max_side 缩放，再重新编码为 JPEG。
        - 若仍超限，逐步降低质量直到满足要求。
        - 压缩失败则返回原图，保证不中断。
        """
        try:
            raw = base64.b64decode(img_b64, validate=False)
            img = Image.open(BytesIO(raw)).convert("RGB")
            img.thumbnail((max_side, max_side))
            compressed_data = None
            for q in qualities:
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=q, optimize=True)
                data = buf.getvalue()
                compressed_data = data
                if len(data) <= max_bytes:
                    break
            return base64.b64encode(compressed_data).decode("utf-8")
        except Exception:
            return img_b64

    def _build_tool_image_message(self, image_list: List[str]) -> List[Dict[str, Any]]:
        """
        将工具返回的图片转换为消息块。
        基类仅简单注入图片，不添加特殊 Token，子类可覆盖添加自定义标记。
        """
        blocks = []
        for img_b64 in image_list:
            img_b64 = self._compress_base64_image(img_b64)
            blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }
            })
        return [{"role": "user", "content": blocks}] if blocks else []

    def _append_input_images(self, user_content: List[Dict[str, Any]]) -> None:
        """
        钩子：注入任务输入图片。基类默认不做额外打标。
        子类可覆盖以添加特殊 Token 或自定义结构。
        """
        input_images = getattr(self, "input_images", None)
        if not isinstance(input_images, list) or not input_images:
            return

        for img in input_images:
            if not isinstance(img, dict):
                continue
            b64 = img.get("b64")
            url = img.get("url")
            if b64:
                b64 = self._compress_base64_image(b64)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high"
                    }
                })
            elif url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": "high"
                    }
                })

    # =========================================================================
    # 工具管理与执行 (适配 MCP)
    # =========================================================================

    def execute_tool(self, tool_name: str, params: Union[str, dict], **kwargs) -> Union[str, Dict[str, Any]]:
        """执行工具：直接代理到 MCP"""
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                pass
        return self._call_tool_sync(tool_name, params)

    async def execute_tool_async(self, tool_name: str, params: Union[str, dict], **kwargs) -> Union[str, Dict[str, Any]]:
        """
        异步执行工具：直接代理到 MCP（用于异步采样/并发工具调用场景）。
        注意：必须在与本实例 `self._loop` 绑定的同一事件循环中运行。
        """
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                pass
        return await self._call_tool_async(tool_name, params)

    def get_tool_schemas(self) -> List[ChatCompletionToolParam]:
        return self.tool_schemas  # type: ignore

    def get_tool_descriptions(self) -> str:
        return self.tool_descriptions

    # =========================================================================
    # Prompt 工程
    # =========================================================================

    def get_action_space(self) -> Optional[str]:
        mode_config = self.config.get(self.mode)
        if isinstance(mode_config, dict) and "action_space" in mode_config:
            return mode_config.get("action_space")
        return self.config.get("action_space")

    def get_system_prompt(self, task_question: Optional[str] = None, **kwargs) -> str:
        action_space = self.get_action_space()
        if action_space is None:
            prompt_template = get_system_prompt(environment_mode=self.mode)
        else:
            prompt_template = get_system_prompt(
                environment_mode=self.mode, action_space=action_space
            )

        prompt = prompt_template.replace("{tool_descriptions}", self.get_tool_descriptions())

        if task_question:
            prompt += f"\nYou are asked to complete the following task: {task_question}"
        
        return prompt

    # =========================================================================
    # MCP 专用逻辑
    # =========================================================================

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {"modules": []}
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"modules": []}

    def _initialize_tools(self):
        """从 MCP Server 获取工具列表并进行本地适配"""
        if self._tools_initialized:
            return

        try:
            # 减少日志：移除工具获取日志
            # logger.info(f"[{self.worker_id}] Fetching tools from MCP Server...")
            mcp_tools = self._list_tools_sync()

            # 1. 默认为白名单优先（如未配置白名单，则回退到黑名单与隐藏标记过滤）
            default_blacklist = {
                "get_observation", "evaluate_task",
                "allocate_batch_resources", "setup_batch_resources",
                "get_batch_initial_observations", "setup_vm_session",
                "setup_rag_session", "teardown_rag_session", "teardown_environment", "release_rag_session",
            }

            valid_tools = []
            self.local_tools = {}

            for t in mcp_tools:
                # 1) 白名单优先：若配置了白名单，则仅允许白名单内的工具
                if self._tool_whitelist:
                    if t.name not in self._tool_whitelist:
                        continue
                else:
                    # 2) 未设置白名单时，使用黑名单 + [HIDDEN] 过滤作为兜底策略
                    if t.name in default_blacklist:
                        continue
                    description = t.description or ""
                    if description.startswith("[HIDDEN]"):
                        continue
                
                valid_tools.append(t)
                
                # [核心适配] 将 MCP Schema 转换为 ToolMetadata (List[Dict] 格式)
                # 这允许 GenericTrajectorySampler 能够读取 tool.parameters
                converted_params = self._convert_schema_to_params(t.inputSchema)
                self.local_tools[t.name] = ToolMetadata(
                    name=t.name,
                    description=t.description or "",
                    parameters=converted_params
                )

            # 生成 Schema 和描述字符串给 LLM
            self.tool_schemas = [self._convert_mcp_tool_to_openai(t) for t in valid_tools]

            descriptions = [f"- {t.name}: {t.description or 'No description.'}" for t in valid_tools]
            self.tool_descriptions = "\n".join(descriptions)
            self._tools_initialized = True

            # 减少日志：移除工具数量日志
            # logger.info(f"[{self.worker_id}] {len(valid_tools)} tools initialized")

        except Exception as e:
            # 在 __init__ 阶段调用时,客户端未连接是预期行为,降低日志级别避免误导
            if self._tools_initialized:
                # 如果之前已成功初始化过,现在失败才是真正的错误
                logger.error(f"[{self.worker_id}] Failed to re-initialize tools: {e}")
            else:
                # 首次初始化失败(客户端未连接)只记录 debug 级别
                logger.debug(f"[{self.worker_id}] Tools not initialized yet (client not connected): {e}")

    def _convert_schema_to_params(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将 JSON Schema properties 转换为参数列表格式"""
        def _extract_constraints(node: Dict[str, Any]) -> Dict[str, Any]:
            """提取常见约束信息，便于在 UI/agent 侧透出格式要求。"""
            constraints: Dict[str, Any] = {}
            for key in ("enum", "const", "pattern", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "default"):
                if key in node:
                    constraints[key] = node[key]
            return constraints

        params = []
        if not schema or "properties" not in schema:
            return params
            
        required_set = set(schema.get("required", []))
        properties = schema.get("properties", {})
        
        for name, prop in properties.items():
            if name == "worker_id": 
                continue
            
            param_def = {
                "name": name,
                "type": prop.get("type", "string"),
                "description": prop.get("description", ""),
                "required": name in required_set
            }
            # 追加约束信息（enum、pattern、长度/数值范围等）
            param_def.update(_extract_constraints(prop))

            # 为数组类型补充 items 约束（如 enum、pattern）
            if param_def["type"] == "array" and "items" in prop and isinstance(prop["items"], dict):
                item_constraints = _extract_constraints(prop["items"])
                item_type = prop["items"].get("type", "string")
                param_def["array_type"] = item_type
                if item_constraints:
                    param_def["array_constraints"] = item_constraints
            
            params.append(param_def)
        return params

    def _convert_mcp_tool_to_openai(self, mcp_tool) -> ChatCompletionToolParam:
        """
        将 MCP 工具定义转换为 OpenAI 工具格式。
        自动移除 worker_id 和 task_session_id 参数，因为它们由环境自动注入。
        """
        # 深拷贝避免修改原始 schema
        parameters = {}
        if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
            parameters = mcp_tool.inputSchema.copy()
        
        # 隐藏系统参数，避免 Agent 感知到这些由环境注入的标识
        system_params = ["worker_id", "task_session_id"]

        if "properties" in parameters:
            for param in system_params:
                if param in parameters["properties"]:
                    del parameters["properties"][param]

        if "required" in parameters and isinstance(parameters["required"], list):
            parameters["required"] = [p for p in parameters["required"] if p not in system_params]

        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or "No description provided.",
                "parameters": parameters
            }
        }

    def env_start(self):
        logger.info(f"Worker [{self.worker_id}] connecting to MCP...")
        self._run_sync(self.mcp_client.connect())
        # 连接建立后重新初始化工具列表
        self._tools_initialized = False
        self._initialize_tools()

    def _close_client_safely(self):
        """
        [新增] 安全关闭客户端连接的统一方法
        
        处理 AsyncExitStack 在不同任务上下文中的问题，提供同步关闭选项。
        可以被所有环境子类复用。
        """
        if not hasattr(self, 'mcp_client') or not self.mcp_client:
            return
        
        # 优先使用同步关闭方法（避免 AsyncExitStack 任务上下文问题）
        if hasattr(self.mcp_client, 'close_sync'):
            try:
                self.mcp_client.close_sync()
                logger.debug(f"[{self.worker_id}] MCP client closed (sync)")
                return
            except Exception as e:
                logger.warning(f"[{self.worker_id}] Sync close failed, trying async: {e}")
        
        # 回退到异步关闭（如果 loop 可用且未关闭）
        if hasattr(self, '_loop') and self._loop and not self._loop.is_closed():
            if hasattr(self.mcp_client, 'close'):
                try:
                    self._loop.run_until_complete(self.mcp_client.close())
                    logger.debug(f"[{self.worker_id}] MCP client closed (async)")
                except Exception as e:
                    logger.warning(f"[{self.worker_id}] Error closing MCP client: {e}")
    
    def _cleanup_event_loop(self):
        """
        [新增] 清理事件循环的统一方法
        
        可以被所有环境子类复用。
        """
        if not hasattr(self, '_loop') or not self._loop or self._loop.is_closed():
            return
        
        try:
            # 取消所有待处理的任务
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            # 等待任务取消完成
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Error cancelling tasks: {e}")
        
        try:
            self._loop.close()
            logger.debug(f"[{self.worker_id}] Event loop closed")
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Error closing loop: {e}")

    def env_close(self):
        """
        关闭MCP连接，优雅处理各种异常
        
        [优化] 使用统一的辅助方法，简化代码并提高可维护性。
        所有环境子类（HttpMCPRagEnv, HttpMCPSearchEnv 等）都可以复用此实现。
        """
        try:
            # 1. 关闭客户端连接（使用安全方法）
            self._close_client_safely()
            
            # 2. 清理事件循环
            self._cleanup_event_loop()
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error in env_close: {e}")

    def _run_sync(self, awaitable):
        return self._loop.run_until_complete(awaitable)

    def _list_tools_sync(self):
        return self._run_sync(self.mcp_client.list_tools())

    async def _call_tool_with_retry(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        max_attempts: int = 2,
    ) -> CallToolResult:
        """
        对底层 mcp_client.call_tool 做"失败 → reset → 再试一次"。

        触发场景（高并发下高频）：
        - SDK 内部 `post_writer` 因 ConnectTimeout 静默死亡 → 之前是永久挂起，
          现在会在 ClientSession.read_timeout 后抛 McpError，本函数自动重连重试。
        - SSE 流被中间代理切断（`anyio.EndOfStream` 等） → 同样重连重试。
        - server 端短暂阻塞 5xx → 重试。

        仅重试一次：避免雪崩；失败仍由上层 try/except 兜底返回 error JSON。
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.mcp_client.call_tool(name, arguments)
            except Exception as e:
                last_exc = e
                if attempt >= max_attempts:
                    break
                logger.warning(
                    f"[{self.worker_id}] tool {name} attempt {attempt}/{max_attempts} "
                    f"failed: {type(e).__name__}: {e}; resetting MCP session and retrying..."
                )
                try:
                    await self.mcp_client.reset()
                except Exception as reset_err:
                    logger.error(
                        f"[{self.worker_id}] MCP reset before retry failed: {reset_err}"
                    )
                    # 即便 reset 失败，下一轮 call_tool 内部会再尝试 reset
        assert last_exc is not None
        raise last_exc

    def _call_tool_sync(self, name: str, arguments: Union[Dict[str, Any], str]):
        """
        同步调用 MCP 工具
        
        自动注入 worker_id 参数（如果缺失）以确保工具调用的一致性。
        特殊处理资源管理类工具的返回值。
        """
        # 确保参数是字典格式
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON arguments for tool {name}: {arguments}")
                raise ValueError(f"Invalid JSON arguments for tool {name}")

        # 自动注入 worker_id（如果缺失）
        if isinstance(arguments, dict) and "worker_id" not in arguments:
            arguments["worker_id"] = self.worker_id
        
        # [新增] 自动注入 task_session_id（如果存在且缺失）
        if isinstance(arguments, dict):
            task_session_id = _task_session_id.get(None)
            if task_session_id and "task_session_id" not in arguments:
                arguments["task_session_id"] = task_session_id

        # 记录调用日志（截断长参数）
        try:
            if isinstance(arguments, dict):
                safe_args = dict(arguments)
                if "messages" in safe_args:
                    msgs = safe_args["messages"]
                    safe_args["messages"] = f"[len={len(msgs)}]"
            else:
                safe_args = arguments
            logger.info(f"[{self.worker_id}] 🔧 Tool call -> {name} args={safe_args}")
        except Exception:
            pass

        # 发起同步工具调用（失败时尝试重连重试一次）
        try:
            res: CallToolResult = self._run_sync(
                self._call_tool_with_retry(name, arguments if isinstance(arguments, dict) else {})
            )
        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ Tool call failed -> {name}: {e}")
            # 返回标准化错误结构，避免上层崩溃
            return {"text": json.dumps({"status": "error", "tool": name, "message": str(e)}, ensure_ascii=False), "images": []}

        # 特殊处理资源管理类工具（直接返回原始结果）
        resource_management_tools = {
            "allocate_batch_resources", "setup_batch_resources",
            "get_batch_initial_observations", "teardown_environment",
            "release_batch_resources"
        }
        if name in resource_management_tools:
            return res

        # 标准化输出格式（文本+图像）
        output = {
            "text": "",
            "images": []
        }

        # 解析 MCP 响应内容
        texts = []
        if hasattr(res, 'content') and res.content:
            for item in res.content:
                # 文本内容累积
                if item.type == 'text':
                    texts.append(item.text)
                # 图像内容收集
                elif item.type == 'image':
                    # 支持 Data URI 和纯 Base64 两种格式
                    image_data = item.data
                    if ',' in image_data:
                        # Data URI 格式: data:image/png;base64,...
                        image_data = image_data.split(',', 1)[1]
                    output["images"].append(image_data)
        else:
            # 无内容时返回默认成功消息
            texts.append(str(res) if res else "Success")

        # 合并所有文本内容
        output_text = "\n".join(texts)
        output["text"] = output_text

        # 记录返回摘要（截断）
        try:
            preview = output_text[:200].replace("\n", " ") if output_text else ""
            result_count = self._count_tool_results(output_text)
            count_suffix = ""
            if result_count is not None:
                count_suffix = f" results={result_count} llm_results={result_count}"
            logger.info(
                f"[{self.worker_id}] ✅ Tool result <- {name} text='{preview}' "
                f"images={len(output['images'])}{count_suffix}"
            )
            if "Image token" in output_text and "not found" in output_text:
                logger.warning(f"[{self.worker_id}] ⚠️ Tool {name} reported missing image token. Check whether input images were injected (input_images={len(getattr(self, 'input_images', []) )}).")
            # 检测结构化错误并打印
            try:
                data = json.loads(output_text)
                if isinstance(data, dict) and data.get("status") == "error":
                    logger.error(f"[{self.worker_id}] ❗ Tool error <- {name}: {data.get('message')}")
            except Exception:
                pass
        except Exception:
            pass
        return output

    async def _call_tool_async(self, name: str, arguments: Union[Dict[str, Any], str]):
        """
        异步调用 MCP 工具（与 `_call_tool_sync` 语义对齐）。
        """
        # 确保参数是字典格式
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON arguments for tool {name}: {arguments}")
                raise ValueError(f"Invalid JSON arguments for tool {name}")

        # 自动注入 worker_id（如果缺失）
        if isinstance(arguments, dict) and "worker_id" not in arguments:
            arguments["worker_id"] = self.worker_id
        
        # [新增] 自动注入 task_session_id（如果存在且缺失）
        if isinstance(arguments, dict):
            task_session_id = _task_session_id.get(None)
            if task_session_id and "task_session_id" not in arguments:
                arguments["task_session_id"] = task_session_id

        # 记录调用日志（截断长参数）
        try:
            if isinstance(arguments, dict):
                safe_args = dict(arguments)
                if "messages" in safe_args:
                    msgs = safe_args["messages"]
                    safe_args["messages"] = f"[len={len(msgs)}]"
            else:
                safe_args = arguments
            logger.info(f"[{self.worker_id}] 🔧 (async) Tool call -> {name} args={safe_args}")
        except Exception:
            pass

        # 发起异步工具调用（失败时尝试重连重试一次）
        try:
            res: CallToolResult = await self._call_tool_with_retry(
                name, arguments if isinstance(arguments, dict) else {}
            )
        except Exception as e:
            logger.error(f"[{self.worker_id}] ❌ (async) Tool call failed -> {name}: {e}")
            return {"text": json.dumps({"status": "error", "tool": name, "message": str(e)}, ensure_ascii=False), "images": []}

        # 特殊处理资源管理类工具（直接返回原始结果）
        resource_management_tools = {
            "allocate_batch_resources", "setup_batch_resources",
            "get_batch_initial_observations", "teardown_environment",
            "release_batch_resources"
        }
        if name in resource_management_tools:
            return res

        # 标准化输出格式（文本+图像）
        output = {
            "text": "",
            "images": []
        }

        texts = []
        if hasattr(res, 'content') and res.content:
            for item in res.content:
                if item.type == 'text':
                    texts.append(item.text)
                elif item.type == 'image':
                    image_data = item.data
                    if ',' in image_data:
                        image_data = image_data.split(',', 1)[1]
                    output["images"].append(image_data)
        else:
            texts.append(str(res) if res else "Success")

        output_text = "\n".join(texts)
        output["text"] = output_text

        # 记录返回摘要（截断）
        try:
            preview = output_text[:200].replace("\n", " ") if output_text else ""
            result_count = self._count_tool_results(output_text)
            count_suffix = ""
            if result_count is not None:
                count_suffix = f" results={result_count} llm_results={result_count}"
            logger.info(
                f"[{self.worker_id}] ✅ (async) Tool result <- {name} text='{preview}' "
                f"images={len(output['images'])}{count_suffix}"
            )
            try:
                data = json.loads(output_text)
                if isinstance(data, dict) and data.get("status") == "error":
                    logger.error(f"[{self.worker_id}] ❗ (async) Tool error <- {name}: {data.get('message')}")
            except Exception:
                pass
        except Exception:
            pass

        return output

    def _parse_mcp_response(self, response: CallToolResult) -> Dict[str, Any]:
        try:
            if response.content and len(response.content) > 0:
                content_item = response.content[0]
                text_content = getattr(content_item, 'text', None)
                if not text_content and hasattr(content_item, 'resource'):
                    text_content = getattr(content_item.resource, 'text', None)
                
                if text_content:
                    try:
                        data = json.loads(text_content)
                        if isinstance(data, dict) and data.get("status") == "error":
                            logger.error(f"[{self.worker_id}] Tool returned error payload: {data}")
                        return data
                    except Exception as e:
                        logger.error(f"[{self.worker_id}] Failed to parse MCP response JSON: {e}")
                        return {"status": "error", "message": str(e), "raw": text_content}
            return {"status": "unknown"}
        except Exception as e:
            logger.error(f"[{self.worker_id}] Exception parsing MCP response: {e}")
            return {"status": "error", "message": str(e)}

    def fetch_initial_observations(self) -> Dict[str, Any]:
        """
        标准化的初始观察获取入口。
        基类默认不抓取任何观察；需要观察能力的环境请在子类中覆盖。
        """
        return {}

    def allocate_resource(self, worker_id: str, resource_init_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        统一的资源分配入口函数 (MCP 模式)
        """
        resource_init_data = resource_init_data or {}
        # 减少日志：简化资源分配开始日志
        logger.info(f"[{worker_id}] Allocating resources...")
        self.initial_observation = None

        try:
            if not self.active_resources:
                logger.info(f"[{self.worker_id}] Running in stateless mode (no heavy resources required). Initializing tools only.")
                # 即使没有需要分配的资源，也调用获取观察值，因为可能有无状态工具可用
                self.fetch_initial_observations()
                return True

            # 1. 申请资源
            # 减少日志：移除批量资源分配详细日志
            # logger.info(f"[{self.worker_id}] Allocating batch resources: {self.active_resources}...")
            res = self._call_tool_sync("allocate_batch_resources", {
                "resource_types": self.active_resources,
                "timeout": 600
            })
            data = self._parse_mcp_response(res)
            if isinstance(data, dict) and data.get("status") == "error":
                 logger.error(f"Alloc failed: {data.get('message')}")
                 return False

            self.allocated_resources = data

            # 2. 初始化资源（总是调用以确保会话同步）
            # 即使没有 resource_init_data，也需要调用 setup_batch_resources 来同步会话
            # 减少日志：移除资源设置日志
            # logger.info(f"[{self.worker_id}] Setting up resources...")
            setup_res = self._call_tool_sync("setup_batch_resources", {
                "resource_init_configs": resource_init_data,  # 可以为空 dict，不影响会话同步
                "allocated_resources": data  # 关键：传递已分配的资源信息用于 _sync_resource_sessions
            })
            setup_result = self._parse_mcp_response(setup_res)
            if setup_result.get("status") not in ["success", "partial_error"]:
                logger.error(f"Setup failed: {setup_result}")
                self.release_resource(self.worker_id)
                return False

            # 3. 获取初始观察（由子类决定是否实现）
            self.fetch_initial_observations()
            return True

        except Exception as e:
            logger.error(f"Allocate resource exception: {e}")
            return False

    async def allocate_resource_async(self, worker_id: str, resource_init_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        异步资源分配入口（适配 async worker，避免在运行的 loop 上 run_until_complete）。
        """
        resource_init_data = resource_init_data or {}
        logger.info(f"[{worker_id}] (async) Allocating resources...")
        self.initial_observation = None

        try:
            if not self.active_resources:
                logger.info(f"[{self.worker_id}] (async) Stateless mode, initializing tools only.")
                self.fetch_initial_observations()
                return True

            res = await self._call_tool_async("allocate_batch_resources", {
                "resource_types": self.active_resources,
                "timeout": 600
            })
            data = self._parse_mcp_response(res)
            if isinstance(data, dict) and data.get("status") == "error":
                logger.error(f"(async) Alloc failed: {data.get('message')}")
                return False

            self.allocated_resources = data

            setup_res = await self._call_tool_async("setup_batch_resources", {
                "resource_init_configs": resource_init_data,
                "allocated_resources": data
            })
            setup_result = self._parse_mcp_response(setup_res)
            if setup_result.get("status") not in ["success", "partial_error"]:
                logger.error(f"(async) Setup failed: {setup_result}")
                await self.release_resource_async(worker_id)
                return False

            self.fetch_initial_observations()
            return True
        except Exception as e:
            logger.error(f"Allocate resource async exception: {e}")
            return False

    def release_resource(self, worker_id: str, reset: bool = True) -> None:
        """
        统一释放所有已分配的资源 (MCP 模式)
        调用 system_resource 组的 release_batch_resources 工具
        """
        # 减少日志：简化资源释放开始日志
        logger.info(f"[{worker_id}] Releasing resources...")
        
        # 收集所有已分配资源的 ID
        resource_ids = []
        for res_type, res_data in self.allocated_resources.items():
            if isinstance(res_data, dict) and "id" in res_data:
                resource_ids.append(res_data["id"])
        
        if not resource_ids:
            # 减少日志：移除无资源释放日志
            # logger.info(f"Worker [{worker_id}] has no resources to release.")
            return

        try:
            # [核心修改] 调用 MCP 工具进行批量释放
            self._call_tool_sync("release_batch_resources", {
                "worker_id": worker_id,
                "resource_ids": resource_ids
            })

            # 清空本地记录
            self.allocated_resources.clear()
            # 减少日志：移除释放完成日志
            # logger.info(f"Worker [{worker_id}] release completed.")

        except Exception as e:
            logger.error(f"Failed to release resources via MCP: {e}")
        finally:
            # [新增] 关闭心跳线程池（避免资源泄漏）
            if hasattr(self, '_heartbeat_executor'):
                try:
                    self._heartbeat_executor.shutdown(wait=False)  # 不等待，快速关闭
                except Exception as e:
                    logger.debug(f"[{worker_id}] Error shutting down heartbeat executor: {e}")

    async def release_resource_async(self, worker_id: str, reset: bool = True) -> None:
        """
        异步释放资源，适配 async worker。
        """
        logger.info(f"[{worker_id}] (async) Releasing resources...")
        resource_ids = []
        for res_type, res_data in self.allocated_resources.items():
            if isinstance(res_data, dict) and "id" in res_data:
                resource_ids.append(res_data["id"])

        if not resource_ids:
            return

        try:
            await self._call_tool_async("release_batch_resources", {
                "worker_id": worker_id,
                "resource_ids": resource_ids
            })
            self.allocated_resources.clear()
        except Exception as e:
            logger.error(f"Failed to release resources via MCP (async): {e}")
        finally:
            if hasattr(self, '_heartbeat_executor'):
                try:
                    self._heartbeat_executor.shutdown(wait=False)
                except Exception as e:
                    logger.debug(f"[{worker_id}] Error shutting down heartbeat executor: {e}")

    def get_allocated_resource_id(self) -> str:
        return self.worker_id
    
    def send_heartbeat(self, progress_info: Optional[Dict[str, Any]] = None) -> None:
        """Update local activity time for MCP-managed resources.

        The open-source search path does not ship a separate resource API.
        Search environments disable heavy-resource allocation, so this is normally a no-op.
        """
        if self.allocated_resources:
            self._last_activity_time = time.time()

    def get_last_activity_time(self) -> float:
        """
        获取最后活动时间（用于探活检查）
        
        Returns:
            最后活动时间戳
        """
        return self._last_activity_time
    # =========================================================================
    # [新增] 缺失的辅助功能函数 (适配框架调用)
    # =========================================================================

    def get_task_output_dir(self, base_dir: str, task_id: str, model_name: str, dataset: Optional[str] = None) -> str:
        """
        生成任务特定的输出目录路径。
        
        Args:
            base_dir: 基础输出目录 (如 'results' 或 'results/science')
            task_id: 任务 ID
            model_name: 模型名称
            dataset: 数据集名称（如 test/Science），用于按模型后再分层保存
            
        Returns:
            完整的输出目录路径 (例如: results/gpt-4/test)
        """
        # 对模型名称进行文件系统安全处理
        safe_model = "".join([c if c.isalnum() or c in "-_." else "_" for c in model_name])
        safe_dataset = "".join([c if c.isalnum() or c in "-_." else "_" for c in (dataset or "unknown")]) or "unknown"

        base_name = os.path.basename(os.path.normpath(base_dir))
        if base_name == safe_model:
            path = base_dir
        else:
            # 将所有日志归档在以 "模型名" 命名的子目录下
            # run_task 中的 _save_conversation_log 会在这个目录下创建 {task_id}.json
            path = os.path.join(base_dir, safe_model)

        # 在模型目录下再按数据集分层
        path = os.path.join(path, safe_dataset)

        # 确保目录存在
        os.makedirs(path, exist_ok=True)
        return path

    def _extract_domain_d1(self, metadata: Optional[Dict[str, Any]]) -> str:
        """提取一级领域名（d1），缺省则返回 'unknown'。"""
        md = self._safe_dict(metadata)
        domain = md.get("domain") or {}
        if isinstance(domain, dict):
            val = domain.get("d1") or "unknown"
        else:
            val = "unknown"
        return str(val)

    def initialize_with_task_config(self, task_config: Dict[str, Any]) -> None:
        """
        接受任务级别的特定配置（如分辨率要求、特定环境参数等）。
        框架会在 run_task 之前调用此方法。
        """
        if not task_config:
            return

        # 减少日志：移除任务配置应用日志
        # logger.info(f"[{self.worker_id}] Applying task specific config: {task_config}")
        # 更新实例配置，以便后续 allocate/setup 阶段可以使用新参数
        self.config.update(task_config)

    def init(self):
        """
        Worker 进程启动后的可选初始化钩子。
        框架在实例化环境后会尝试调用此方法。
        """
        # 在 MCP 模式下，连接已经在 env_start 中建立，此处可留空或做额外检查
        pass

    def cleanup(self, worker_id: Optional[str] = None):
        """
        清理资源的统一入口。
        框架在 Worker 退出或收到停止信号时会调用此方法。
        """
        wid = worker_id or self.worker_id
        # 减少日志：移除清理开始日志
        # logger.info(f"[{wid}] Cleaning up environment resources...")
        try:
            # [新增] 关闭心跳线程池（避免资源泄漏）
            if hasattr(self, '_heartbeat_executor'):
                try:
                    self._heartbeat_executor.shutdown(wait=False)  # 不等待，快速关闭
                except Exception as e:
                    logger.debug(f"[{wid}] Error shutting down heartbeat executor: {e}")
            
            # 1. 释放远端资源
            self.release_resource(wid)
            # 2. 关闭 MCP 连接
            self.env_close()
        except Exception as e:
            logger.error(f"[{wid}] Cleanup failed: {e}")

    def _truncate_data(self, data: Any, max_len: int = 100) -> Any:
        """
        辅助函数：递归截断数据结构中的长字符串，仅用于日志展示。
        """
        if isinstance(data, dict):
            return {k: self._truncate_data(v, max_len) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._truncate_data(i, max_len) for i in data]
        elif isinstance(data, str):
            if len(data) > max_len:
                # 保留前 max_len 个字符，并提示总长度
                return f"{data[:max_len]}... [TRUNCATED, total_len={len(data)}]"
            return data
        else:
            return data
