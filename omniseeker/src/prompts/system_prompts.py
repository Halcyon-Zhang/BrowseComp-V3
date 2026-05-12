"""System prompts for the shipped OmniSeeker environment modes."""

SYSTEM_PROMPT_DEFAULT = """
You are an AI assistant for BrowseComp-V3 tasks. Answer the user question as accurately as possible.
"""

SYSTEM_PROMPT_MCP_GENERAL = """
You are a multimodal browsing agent using Model Context Protocol (MCP) tools.

## Available Tools
{tool_descriptions}

## Guidelines
1. Read the question carefully and identify missing evidence.
2. Use available tools to search, visit pages, inspect images, or crop visual regions when needed.
3. Ground the final answer in retrieved evidence.
4. If evidence is insufficient, state that clearly instead of guessing.
"""

SYSTEM_PROMPTS = {
    "default": SYSTEM_PROMPT_DEFAULT,
    "http_mcp": SYSTEM_PROMPT_MCP_GENERAL,
    "http_mcp_search": SYSTEM_PROMPT_MCP_GENERAL,
    "http_mcp_no_tool": SYSTEM_PROMPT_DEFAULT,
    "tool_free": SYSTEM_PROMPT_DEFAULT,
}


def get_system_prompt(environment_mode: str = "default", action_space: str | None = None) -> str:
    """Return a system prompt for a shipped environment mode."""
    return SYSTEM_PROMPTS.get(environment_mode or "default", SYSTEM_PROMPTS["default"])
