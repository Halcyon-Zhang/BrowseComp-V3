# src/utils/mcp_sse_client.py
"""
MCP SSE 客户端封装。

设计目标：
1. **不再永久挂起**：在 `ClientSession` 上设置 `read_timeout_seconds`，
   当 SDK 内部 `post_writer` 因 `httpx.ConnectTimeout` 等异常静默死亡后，
   `call_tool` 会在 `read_timeout_seconds` 后抛出 `McpError`，而不是无限等。
2. **更宽容的连接超时**：默认 5s 的 connect timeout 在并发上来后非常容易触发
   server 端瞬时阻塞导致的 SYN backlog 溢出。我们把它放大到 30s。
3. **会话健康检测 + 自动重连**：每次调用记录最近一次的失败状态；下次调用时
   优先 `close + connect` 再发请求。同时显式提供 `is_alive()` / `reset()`。
4. **真正的 close**：`close_sync` 之前只把对象引用置 None，会泄漏 SSE 长连接、
   后台 anyio 任务和 httpx 连接池。改为安全地在持有事件循环时跑 `aclose()`。
"""

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

import httpx

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import CallToolResult, Tool

logger = logging.getLogger(__name__)


# 默认参数：经验值，可以用环境变量覆盖。这里写常量，让上层 env 显式传更直观。
DEFAULT_HTTP_CONNECT_TIMEOUT = 30.0     # SSE 连接 + POST /messages 的超时（默认 5s 太短）
DEFAULT_SSE_READ_TIMEOUT = 60 * 10.0    # SSE 流空闲超时（保持长连接）
DEFAULT_SESSION_READ_TIMEOUT = 180.0    # call_tool 等待 JSON-RPC 响应的兜底超时


def _make_httpx_factory(
    connect_timeout: float,
    sse_read_timeout: float,
) -> Callable[..., httpx.AsyncClient]:
    """
    包装 SDK 的 `create_mcp_http_client`，把全局 timeout 替换成更细粒度的：
    - connect / write / pool: connect_timeout
    - read: sse_read_timeout（保留长 SSE 流）

    SDK 内部调用形如 `factory(headers=..., auth=..., timeout=httpx.Timeout(...))`。
    我们覆盖 timeout，但保留其他 kwargs。
    """

    def factory(*args, **kwargs) -> httpx.AsyncClient:
        kwargs["timeout"] = httpx.Timeout(
            connect=connect_timeout,
            read=sse_read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        return create_mcp_http_client(*args, **kwargs)

    return factory


class MCPSSEClient:
    """
    兼容 SSE (Server-Sent Events) 协议的 MCP 客户端。
    用于连接通过 HTTP 暴露的 MCP 网关。
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8080/sse",
        *,
        http_connect_timeout: float = DEFAULT_HTTP_CONNECT_TIMEOUT,
        sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
        session_read_timeout: float = DEFAULT_SESSION_READ_TIMEOUT,
    ):
        """
        :param server_url: SSE 端点地址 (例如 http://localhost:8080/sse)
        :param http_connect_timeout: 建立 HTTP 连接 + POST /messages 的超时（秒）
        :param sse_read_timeout: SSE 流空闲超时（秒）
        :param session_read_timeout: 等待 JSON-RPC 响应的兜底超时（秒）
        """
        self.server_url = server_url
        self.http_connect_timeout = http_connect_timeout
        self.sse_read_timeout = sse_read_timeout
        self.session_read_timeout = session_read_timeout

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.read = None
        self.write = None

        # 会话健康标志：一旦 call_tool 失败，置 False；下次调用先 reset 再发请求。
        self._healthy: bool = False
        # 失败次数（仅做日志/限流，不影响主流程逻辑）
        self._failure_count: int = 0
        # 防止 reset 与正常调用并发抢占资源
        self._reset_lock: Optional[asyncio.Lock] = None

        # 抑制清理阶段的 anyio 任务上下文异常（不影响功能，仅美化日志）
        self._setup_exception_handler()

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------

    def _setup_exception_handler(self) -> None:
        """设置异步任务异常处理器，抑制 anyio TaskGroup 的上下文清理噪音。"""

        def exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
            exception = context.get("exception")
            if isinstance(exception, RuntimeError):
                err_msg = str(exception)
                if "cancel scope" in err_msg or "different task" in err_msg:
                    return
            if isinstance(exception, GeneratorExit):
                return

            message = context.get("message", "Unhandled exception")
            logger.warning(f"⚠️ Async task exception: {message}")
            if exception:
                logger.warning(f"   Exception: {exception!r}")

        try:
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(exception_handler)
        except RuntimeError:
            # 没有事件循环时（典型: 多进程 worker 启动早期），先忽略，
            # 真正用到时再由 connect() 安装一次。
            pass

    def _ensure_lock(self) -> asyncio.Lock:
        if self._reset_lock is None:
            self._reset_lock = asyncio.Lock()
        return self._reset_lock

    # ------------------------------------------------------------------
    # 连接 / 关闭
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """建立 SSE 连接并初始化会话。"""
        logger.info(f"📡 Connecting to SSE Endpoint: {self.server_url} ...")

        # 重置事件循环异常处理器（init 时可能没有 loop）
        self._setup_exception_handler()

        # 防止半开连接：旧的 exit_stack 必须先关闭
        if self.session is not None or self.read is not None:
            await self._aclose_quietly()

        try:
            factory = _make_httpx_factory(
                connect_timeout=self.http_connect_timeout,
                sse_read_timeout=self.sse_read_timeout,
            )

            streams = await self.exit_stack.enter_async_context(
                sse_client(
                    self.server_url,
                    timeout=self.http_connect_timeout,
                    sse_read_timeout=self.sse_read_timeout,
                    httpx_client_factory=factory,
                )
            )
            self.read, self.write = streams

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(
                    self.read,
                    self.write,
                    read_timeout_seconds=timedelta(seconds=self.session_read_timeout),
                )
            )

            await self.session.initialize()
            self._healthy = True
            self._failure_count = 0
            logger.info(
                "✅ MCP Session Initialized "
                f"(connect_to={self.http_connect_timeout}s, sse_read={self.sse_read_timeout}s, "
                f"session_read={self.session_read_timeout}s)"
            )

        except Exception as e:
            self._healthy = False
            logger.error(f"❌ MCP Connection Failed: {e}")
            # 失败时清理已部分进入的上下文
            await self._aclose_quietly()
            raise

    async def reset(self) -> None:
        """关闭旧会话并重新建立连接。被并发调用时只执行一次。"""
        lock = self._ensure_lock()
        async with lock:
            if self._healthy and self.session is not None:
                # 在等锁期间已经被另一个协程恢复，无需重连
                return
            # 区分"首次懒加载"（无任何失败，仅是 is_alive==False）和
            # "故障后重连"（有失败计数）。前者用 INFO，避免日志被误读为故障。
            if self._failure_count > 0:
                logger.warning(
                    f"♻️ MCPSSEClient resetting connection to {self.server_url} "
                    f"(failures={self._failure_count})"
                )
            else:
                logger.info(
                    f"📡 MCPSSEClient lazy-connecting to {self.server_url}"
                )
            await self._aclose_quietly()
            await self.connect()

    async def _aclose_quietly(self) -> None:
        """异步地关闭 exit_stack，并把所有引用清掉。任何异常仅做日志。"""
        old_stack = self.exit_stack
        # 立刻替换成新的，避免 aclose 期间被外部再次入栈
        self.exit_stack = AsyncExitStack()
        self.session = None
        self.read = None
        self.write = None
        self._healthy = False

        try:
            await old_stack.aclose()
        except RuntimeError as re:
            msg = str(re)
            if "cancel scope" in msg or "different task" in msg:
                logger.debug(f"AsyncExitStack cleanup race (harmless): {re}")
            else:
                logger.warning(f"AsyncExitStack aclose error: {re}")
        except Exception as e:
            logger.warning(f"AsyncExitStack aclose error: {e}")

    async def close(self) -> None:
        """优雅关闭连接（异步方式）。"""
        await self._aclose_quietly()
        logger.info("🔌 MCP Client Disconnected")

    def close_sync(self) -> None:
        """
        同步关闭（用于在非 async 上下文紧急清理，例如 atexit）。

        注意：和旧实现不同，本方法 **会** 真正关闭后台任务和 httpx 连接，
        前提是 caller 提供的事件循环仍可用。如果事件循环已经关闭/未启动，
        我们退化为只清掉引用（资源会随进程退出回收）。
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop is not None and not loop.is_closed() and not loop.is_running():
            try:
                loop.run_until_complete(self._aclose_quietly())
                logger.info("🔌 MCP Client closed (sync via loop)")
                return
            except Exception as e:
                logger.warning(f"close_sync via loop failed, falling back to drop refs: {e}")

        # 兜底：仅清引用。SSE 后台任务由 GC + 进程退出兜底。
        self.session = None
        self.read = None
        self.write = None
        self._healthy = False
        try:
            self.exit_stack = AsyncExitStack()
        except Exception:
            pass
        logger.info("🔌 MCP Client force-closed (drop refs)")

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def is_alive(self) -> bool:
        """快速健康判断。仅看本地状态，不发起网络探测。"""
        return self._healthy and self.session is not None

    # ------------------------------------------------------------------
    # 远程调用
    # ------------------------------------------------------------------

    async def list_tools(self) -> List[Tool]:
        """获取服务器暴露的所有工具列表。"""
        if not self.is_alive():
            await self.reset()
        assert self.session is not None
        try:
            result = await self.session.list_tools()
            return result.tools
        except Exception as e:
            self._mark_failure(e, "list_tools")
            raise

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        read_timeout_seconds: Optional[float] = None,
    ) -> CallToolResult:
        """
        调用工具并返回原始结果对象 (CallToolResult)。

        :param name: 工具名称
        :param arguments: 工具参数字典
        :param read_timeout_seconds: 单次调用的等待超时；不传则用会话默认值
        :return: CallToolResult 对象，包含完整的多模态响应内容
        """
        if arguments is None:
            arguments = {}

        # 调用前自检：连接挂掉则先重连
        if not self.is_alive():
            await self.reset()
        assert self.session is not None

        # 调试日志（截断长字符串避免污染）
        debug_args: Dict[str, Any] = {}
        try:
            for k, v in arguments.items():
                if isinstance(v, str) and len(v) > 200:
                    debug_args[k] = v[:200] + "...(truncated)"
                else:
                    debug_args[k] = v
        except Exception:
            debug_args = {"<args repr failed>": True}

        logger.info(f"[MCP-CLI] ➡️ REQ Tool: {name} args={debug_args}")
        t0 = time.time()

        # 准备 send_request 的可选超时（per-call 覆盖 session 级）
        call_kwargs: Dict[str, Any] = {}
        if read_timeout_seconds is not None:
            call_kwargs["read_timeout_seconds"] = timedelta(seconds=read_timeout_seconds)

        try:
            # ClientSession.call_tool 在不同 SDK 版本里 signature 略有差异，
            # 我们按官方 1.x 文档传 read_timeout_seconds；不支持的话退回不传。
            try:
                result = await self.session.call_tool(name, arguments, **call_kwargs)
            except TypeError:
                # SDK 不支持 read_timeout_seconds kwarg，退化
                result = await self.session.call_tool(name, arguments)
        except Exception as e:
            self._mark_failure(e, name)
            elapsed = time.time() - t0
            logger.error(f"[MCP-CLI] ❌ Tool {name} failed after {elapsed:.1f}s: {e!r}")
            raise

        elapsed = time.time() - t0
        # 响应摘要（截断）
        content_summary = "Empty"
        if getattr(result, "content", None):
            content_summary = str(result.content)[:500]
        logger.info(
            f"[MCP-CLI] ⬅️ RES Tool: {name} ({elapsed:.2f}s) data={content_summary}"
        )
        return result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _mark_failure(self, exc: BaseException, where: str) -> None:
        """记录失败并标记会话为不健康。"""
        self._failure_count += 1
        self._healthy = False
        logger.warning(
            f"[MCP-CLI] session marked unhealthy at {where}: "
            f"{type(exc).__name__}: {exc} (failure_count={self._failure_count})"
        )


# --- 简单的自测代码 ---
async def main():
    client = MCPSSEClient("http://localhost:8080/sse")
    try:
        await client.connect()

        tools = await client.list_tools()
        print(f"\n🔍 Found {len(tools)} tools:")
        for t in tools:
            desc = (t.description or "")[:50]
            print(f"   - {t.name}: {desc}...")

    except Exception as e:
        print(f"Test Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
