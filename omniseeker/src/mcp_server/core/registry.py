import logging
from typing import List, Callable, Dict, Optional
import importlib
import pkgutil
import sys

logger = logging.getLogger("ToolRegistry")

class ToolRegistry:
    """
    工具注册中心。
    负责维护 '工具组名称' 到 '具体函数列表' 的映射。
    """
    
    # 动态存储结构： {"group_name": [func1, func2]}
    _REGISTRY: Dict[str, List[Callable]] = {}

    @classmethod
    def register_tool(cls, group_name: str, hidden: bool = False):
        """
        装饰器：用于将函数注册到指定的工具组
        
        :param group_name: 工具组名称
        :param hidden: 是否为隐藏工具（不暴露给 LLM，但可通过代码调用）
        """
        def decorator(func: Callable):
            # 如果标记为隐藏，在文档字符串前添加特殊标记
            if hidden:
                original_doc = func.__doc__ or ""
                # 使用特殊前缀标记，稍后在 Client 端解析时过滤
                func.__doc__ = f"[HIDDEN] {original_doc}"
                # 同时也给函数对象打上标记（供内部逻辑使用）
                func.hidden = True

            if group_name not in cls._REGISTRY:
                cls._REGISTRY[group_name] = []
            
            # 避免重复注册
            if func not in cls._REGISTRY[group_name]:
                cls._REGISTRY[group_name].append(func)
                logger.debug(f"Registered tool '{func.__name__}' to group '{group_name}'")
            return func
        return decorator

    @classmethod
    def get_tools_by_group(cls, group_name: str) -> List[Callable]:
        """根据组名获取工具函数列表"""
        tools = cls._REGISTRY.get(group_name)
        if not tools:
            logger.warning(f"Tool group '{group_name}' not found in registry.")
            return []
        return tools

    @classmethod
    def autodiscover(cls, package_path: str):
        """
        自动扫描指定包路径下的所有模块，触发装饰器注册。
        例如传入 'src.mcp_server'
        """
        try:
            package = importlib.import_module(package_path)
            if hasattr(package, "__path__"):
                for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                    try:
                        importlib.import_module(name)
                        logger.info(f"Scanned module: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to scan module {name}: {e}")
        except ImportError as e:
            logger.error(f"Failed to import package {package_path}: {e}")

    @classmethod
    def get_tools_by_config(cls, module_config: dict) -> List[Callable]:
        """
        根据配置项自动解析需要的工具。
        """
        tools = []
        
        # 1. 显式指定的工具组
        groups = module_config.get("tool_groups", [])
        for group in groups:
            tools.extend(cls.get_tools_by_group(group))
            
        # 2. (可选) 根据 resource_type 自动追加默认组的逻辑可以移除或保留
        # 建议：既然现在配置很明确，可以移除这里的隐式逻辑，全靠 gateway_config.json 控制
        
        # 3. 去重
        seen = set()
        unique_tools = []
        for tool in tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)
        
        return unique_tools