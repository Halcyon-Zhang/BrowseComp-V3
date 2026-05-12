import logging
from typing import Any, Dict, Optional

from .http_mcp_env import HttpMCPEnv

logger = logging.getLogger(__name__)


class HttpMCPNoToolEnv(HttpMCPEnv):
    """
    纯无工具环境：继承 HttpMCPEnv，但完全禁用工具/资源调用。
    适用于只需直接回答问题、不依赖 MCP 工具的场景。
    """

    def __init__(self, *args, **kwargs):
        # 强制清空网关配置，避免产生可用资源或工具
        kwargs["gateway_config_path"] = kwargs.get("gateway_config_path", "")
        super().__init__(*args, **kwargs)

        # 覆盖父类初始化的工具缓存
        self.tool_schemas = []
        self.tool_descriptions = ""
        self.local_tools = {}
        self._tools_initialized = True

    @property
    def mode(self) -> str:
        return "http_mcp_no_tool"

    def _initialize_tools(self):
        """
        禁用工具初始化，保证不会向 MCP Server 拉取工具列表。
        """
        self.tool_schemas = []
        self.tool_descriptions = ""
        self.local_tools = {}
        self._tools_initialized = True

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        """
        覆盖以返回空配置，避免产生资源/工具依赖。
        """
        return {"modules": []}

    def env_start(self):
        """
        无工具模式下无需连接 MCP，保持空实现。
        """
        logger.info(f"[{self.worker_id}] HttpMCPNoToolEnv started (no MCP connection).")

    def get_system_prompt(self, task_question: Optional[str] = None, **kwargs) -> str:
        """
        使用简化的直接回答提示词，不注入工具描述。
        """
        prompt = (
            "You are a concise assistant. Tools are unavailable, so answer directly "
            "with your own knowledge.\n"
            "Keep the final answer short and factual without extra reasoning text."
        )
        if task_question:
            prompt += f"\nCurrent question: {task_question}"
        return prompt
