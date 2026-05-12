"""
Patch for FastMCP to fix Pydantic v2 compatibility issue.
Issue: create_model(name, result=type) should be create_model(name, result=(type, ...))
"""
import logging
from typing import Any
from pydantic import BaseModel, create_model

logger = logging.getLogger("FastMCP_Patch")

def patched_create_wrapped_model(func_name: str, annotation: Any) -> type[BaseModel]:
    """
    Patched version of _create_wrapped_model that works with Pydantic v2.
    
    Original code:
        return create_model(model_name, result=annotation)
    
    Fixed code:
        return create_model(model_name, result=(annotation, ...))
    """
    model_name = f"{func_name}Output"
    # 修复: 使用元组格式 (type, ...) 而不是直接传 type
    return create_model(model_name, result=(annotation, ...))

def apply_fastmcp_patch():
    """Apply the patch to FastMCP"""
    try:
        from mcp.server.fastmcp.utilities import func_metadata
        
        # 保存原始函数
        original_func = func_metadata._create_wrapped_model
        
        # 替换为补丁版本
        func_metadata._create_wrapped_model = patched_create_wrapped_model
        
        logger.info("✅ FastMCP patch applied successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply FastMCP patch: {e}")
        return False
