# -*- coding: utf-8 -*-
"""Environment factory for the shipped BrowseComp-V3 runtime modes."""

from typing import Any, Dict

_ENVIRONMENT_REGISTRY: Dict[str, Any] = {}


def register_environment(mode: str, env_class: Any) -> None:
    _ENVIRONMENT_REGISTRY[mode] = env_class


def unregister_environment(mode: str) -> None:
    _ENVIRONMENT_REGISTRY.pop(mode, None)


def list_registered_environments() -> list[str]:
    return sorted(_ENVIRONMENT_REGISTRY)


def is_registered(mode: str) -> bool:
    return mode in _ENVIRONMENT_REGISTRY


def get_environment_class(mode: str) -> Any:
    if mode not in _ENVIRONMENT_REGISTRY:
        available_modes = ", ".join(list_registered_environments())
        raise ValueError(f"Unknown environment mode: {mode!r}. Available modes: {available_modes}.")
    return _ENVIRONMENT_REGISTRY[mode]


def create_environment(mode: str, **kwargs: Any) -> Any:
    return get_environment_class(mode)(**kwargs)


class EnvironmentFactory:
    def register(self, mode: str, env_class: Any) -> None:
        register_environment(mode, env_class)

    def create(self, mode: str, **kwargs: Any) -> Any:
        return create_environment(mode, **kwargs)

    def list_modes(self) -> list[str]:
        return list_registered_environments()

    def is_available(self, mode: str) -> bool:
        return is_registered(mode)


def _auto_register_builtin_environments() -> None:
    from .http_mcp_env import HttpMCPEnv
    from .http_mcp_no_tool_env import HttpMCPNoToolEnv
    from .http_mcp_search_env import HttpMCPSearchEnv
    from .tool_free_env import ToolFreeEnv

    register_environment("http_mcp", HttpMCPEnv)
    register_environment("http_mcp_search", HttpMCPSearchEnv)
    register_environment("http_mcp_no_tool", HttpMCPNoToolEnv)
    register_environment("tool_free", ToolFreeEnv)


_auto_register_builtin_environments()
