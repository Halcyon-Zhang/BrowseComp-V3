"""
MCP Server Adapter for Search V2 Services
Type: Standalone/Stateless Pattern
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Annotated
from pydantic import Field
from mcp_server.core.registry import ToolRegistry

_logger = logging.getLogger(__name__)

# 尝试导入 search_v2 服务
try:
    from utils.search_v2 import (
        TextSearchService,
        ImageSearchService,
        CloudStorageService,
        ImageProcessor,
    )

    SEARCH_V2_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Search V2 services not found: {e}")
    SEARCH_V2_AVAILABLE = False

# ==============================================================================
# Service Initialization (Singleton)
# ==============================================================================
# 这些服务会在模块加载时初始化，它们内部会读取 .env 中的配置
_text_service = None
_image_service = None
_cloud_service = None
_image_processor = None


def get_text_service():
    global _text_service
    if not _text_service and SEARCH_V2_AVAILABLE:
        _text_service = TextSearchService()
    return _text_service


def get_image_service():
    global _image_service
    if not _image_service and SEARCH_V2_AVAILABLE:
        _image_service = ImageSearchService()
    return _image_service


def get_cloud_service():
    global _cloud_service
    if not _cloud_service and SEARCH_V2_AVAILABLE:
        _cloud_service = CloudStorageService()
    return _cloud_service


def get_image_processor():
    global _image_processor
    if not _image_processor and SEARCH_V2_AVAILABLE:
        _image_processor = ImageProcessor()
    return _image_processor


# ==============================================================================
# Tool Definitions
# Group: search_tools
# ==============================================================================


@ToolRegistry.register_tool("search_tools")
async def TextSearch(
    query: Annotated[
        str, Field(description="Search keywords or question text.", min_length=1)
    ],
) -> str:
    """
    Web search with summaries (stateless).
    Return shape (JSON string): List of items with
    - title, url, snippet
    Format hints:
    - query: plain text (no URLs needed unless part of the question)
    """
    service = get_text_service()
    if not service:
        return "Error: TextSearchService not available (check dependencies/config)."

    try:
        k = getattr(service.config, "DEFAULT_SEARCH_RESULTS", 5)
        region = getattr(service.config, "DEFAULT_REGION", "us")
        results = await service.search_with_summaries(query=query, k=k, region=region)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback

        print(f"[ERROR] TextSearch failed: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"status": "error", "tool": "TextSearch", "message": str(e)},
            ensure_ascii=False,
        )


@ToolRegistry.register_tool("search_tools")
async def WebVisit(
    url: Annotated[
        str,
        Field(
            description="Webpage URL (often taken from another search result).",
            min_length=1,
        ),
    ],
) -> str:
    """
    Visit Access a specific webpage via its URL and return the cleaned content processed by Jina Reader (jinaapi), including text and image URLs.
    Response shape mirrors Jina Reader JSON: {"status", "ok": true, "source": "jina_reader", "data": [{"url", "content", "content_length"}]}.
    """
    service = get_text_service()
    if not service:
        return "Error: TextSearchService not available (check dependencies/config)."
    try:
        region = getattr(service.config, "DEFAULT_REGION", "us")
        page_data = await service.visit_url(url=url, region=region)
        ok = page_data.get("ok", True)
        st = page_data.get("status")
        if ok is False or st == "error":
            try:
                payload_preview = json.dumps(page_data, ensure_ascii=False)[:2500]
            except Exception:
                payload_preview = repr(page_data)[:2500]
            _logger.warning(
                "WebVisit tool result not ok url=%r ok=%s status=%r source=%r "
                "message=%r payload_preview=%s",
                url,
                ok,
                st,
                page_data.get("source"),
                (page_data.get("message") or "")[:500],
                payload_preview,
            )
        return json.dumps(page_data, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback

        _logger.exception("WebVisit raised url=%r", url)
        print(f"[ERROR] WebVisit failed: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"status": "error", "tool": "WebVisit", "message": str(e)},
            ensure_ascii=False,
        )


@ToolRegistry.register_tool("search_tools")
async def ImageSearch(
    query: Annotated[
        str,
        Field(
            description="Text query describing the target image (e.g., 'AIRFold logo').",
            min_length=1,
        ),
    ],
) -> str:
    """
    Search images by text and return URLs (thumbnail/full).
    Return shape (JSON string): List of items with
    - title, link
    - thumbnail/thumbnailUrl, image_url/imageUrl?, source?, domain?, position?
    Uses fixed k from config to reduce noise. Provide concise, visual-friendly keywords.
    """
    service = get_image_service()
    if not service:
        return "Error: ImageSearchService not available."

    try:
        k = getattr(service.config, "DEFAULT_IMAGE_RESULTS", 5)
        results = await service.search_by_query(query=query, k=k)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback

        print(f"[ERROR] ImageSearch failed: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"status": "error", "tool": "ImageSearch", "message": str(e)},
            ensure_ascii=False,
        )


@ToolRegistry.register_tool("search_tools")
async def ReverseImageSearch(
    image_token: Annotated[
        Optional[str],
        Field(
            description="Image token from conversation history, e.g., 'image_1' for <image_1>. Prefer this when available.",
            min_length=1,
        ),
    ] = None,
    image_url: Annotated[
        Optional[str],
        Field(
            description="Direct HTTP(S) image URL. Use when no image_token is available; base64 is not supported here to avoid oversized tool calls.",
            min_length=1,
        ),
    ] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Reverse image search using either a referenced image token from the chat or a direct HTTP(S) image URL.
    Return shape (JSON string): List of items with
    - title,
    - thumbnail
    - link
    Notes:
    - Provide either image_token (preferred) or image_url (HTTP/HTTPS). At least one is required.
    - image_token must match a token already present in messages (<image_x> or <obs_x>).
    - messages is normally auto-injected by the client; callers usually omit it.
    - Base64 inputs via image_url are not supported (to keep tool calls short); tokens may still resolve base64 internally and will be uploaded automatically.
    """
    # First try to get the image processor to extract image source from token
    processor = get_image_processor()
    if not processor:
        return "Error: ImageProcessor not available."

    service = get_image_service()
    if not service:
        return "Error: ImageSearchService not available."

    try:
        target_url = None
        normalized_token = None

        # Direct URL path (preferred when provided)
        if image_url:
            if image_url.startswith("data:image"):
                return "Error: Base64 image input is not supported; please provide an HTTP(S) URL or use image_token."
            if not image_url.startswith(("http://", "https://")):
                return "Error: image_url must start with http:// or https://."
            target_url = image_url

        # Token-based path (fallback / preferred for prior images)
        if not target_url:
            if not image_token:
                return "Error: Provide either image_token or image_url."
            # Normalize token to accept both "<image_1>" and "image_1"
            if isinstance(image_token, str):
                normalized_token = image_token.strip()
                if normalized_token.startswith("<") and normalized_token.endswith(">") and len(normalized_token) > 2:
                    normalized_token = normalized_token[1:-1].strip()
            if not normalized_token:
                return "Error: image_token is empty after normalization."

            # Extract image sources from messages
            image_sources = {}
            if messages:
                image_sources.update(processor.extract_sources_from_messages(messages))

            # Find the image URL corresponding to the token
            if normalized_token not in image_sources:
                return f"Error: Image token '{image_token}' not found in conversation history."

            image_detail = image_sources[normalized_token]
            target_url = image_detail.get("image_url", {}).get("url", "")
            if not target_url:
                return f"Error: Could not extract URL for image token '{image_token}'."

            # If token resolves to base64, upload then continue (kept for compatibility; tool args stay short)
            if isinstance(target_url, str) and target_url.startswith("data:image"):
                cloud_service = get_cloud_service()
                if not cloud_service:
                    return "Error: CloudStorageService not available for base64 image upload."

                try:
                    upload_result = await cloud_service.upload_data_url(
                        target_url,
                        filename_prefix="reverse_search",
                    )
                    target_url = upload_result.get("url", "")

                    if not target_url:
                        return "Error: Failed to upload base64 image to cloud storage."

                    print(
                        f"[INFO] Base64 image uploaded successfully, URL: {target_url[:100]}..."
                    )

                except Exception as upload_err:
                    import traceback

                    print(
                        f"[ERROR] Failed to upload base64 image: {upload_err}\n{traceback.format_exc()}"
                    )
                    return json.dumps(
                        {
                            "status": "error",
                            "tool": "ReverseImageSearch",
                            "message": f"Failed to upload base64 image: {str(upload_err)}",
                        },
                        ensure_ascii=False,
                    )

        # Perform reverse image search using the URL (now guaranteed to be HTTP/HTTPS)
        k = getattr(service.config, "DEFAULT_IMAGE_RESULTS", 5)
        results = await service.search_by_image(image_input=target_url, k=k)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback

        print(f"[ERROR] ReverseImageSearch failed: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"status": "error", "tool": "ReverseImageSearch", "message": str(e)},
            ensure_ascii=False,
        )


@ToolRegistry.register_tool("search_tools", hidden=True)
async def upload_file_to_cloud(
    file_path: Annotated[
        str,
        Field(description="Absolute path to the local file to upload.", min_length=1),
    ],
) -> str:
    """
    [Hidden] Upload a local file (e.g., screenshot) to cloud storage and return a public URL.
    """
    service = get_cloud_service()
    if not service:
        return "Error: CloudStorageService not available."

    try:
        from pathlib import Path

        path_obj = Path(file_path)
        if not path_obj.exists():
            return f"Error: File not found at {file_path}"

        result = await service.upload_single_image(path_obj)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback

        print(f"[ERROR] upload_file_to_cloud failed: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"status": "error", "tool": "upload_file_to_cloud", "message": str(e)},
            ensure_ascii=False,
        )


@ToolRegistry.register_tool("search_tools")
async def CropImage(
    crop_config: Annotated[
        Dict[str, List[int]],
        Field(
            description=(
                "Dictionary mapping image tokens to crop boxes. "
                "Key: image token (e.g., '<image_1>', '<obs_2>', or 'image_1' without brackets). "
                "Value: list of exactly 4 integers [left, top, right, bottom] in pixels. "
                "Single crop example: {'<image_1>': [100, 200, 500, 600]}. "
                "Multiple crops example: {'<image_1>': [100, 200, 500, 600], '<image_2>': [50, 50, 300, 300]}. "
                "Do NOT use nested objects with 'token' and 'coordinates' fields."
            )
        ),
    ],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Tool name: CropImage

    Purpose:
    - Crop one or more rectangular regions from images that appear in the conversation
      (tokens like <image_1> or <obs_2>), in order to zoom in on local details
      (e.g. text, numbers, a small object, or a specific chart region).

    When to use this tool:
    - The user asks about a specific *part* of an image (a corner, a line of text,
      a number, one object among many, etc.).
    - You need to zoom in to read small content or focus on a local region
      before doing further reasoning or image search.

    Input schema (VERY IMPORTANT):

    ✅ CORRECT FORMAT - Single Image Crop:
    {
      "crop_config": {
        "<image_1>": [100, 200, 500, 600]
      }
    }

    ✅ CORRECT FORMAT - Multiple Images Crop:
    {
      "crop_config": {
        "<image_1>": [100, 200, 500, 600],
        "<obs_2>": [50, 50, 300, 300],
        "<image_3>": [0, 0, 800, 600]
      }
    }

    Format Rules:
    - crop_config: A dictionary where:
      - KEY: Image token string (e.g., "<image_1>", "<obs_2>", or "image_1" without angle brackets)
      - VALUE: List of exactly 4 integers [left, top, right, bottom] in pixels
        * left: X-coordinate of top-left corner
        * top: Y-coordinate of top-left corner
        * right: X-coordinate of bottom-right corner
        * bottom: Y-coordinate of bottom-right corner
    - Do NOT add extra fields like "token", "coordinates", "annotated", "prettify", etc.
    - Do NOT use nested objects or dictionaries as values
    - All coordinates must be positive integers
    - Ensure right > left and bottom > top

    ❌ WRONG FORMAT - Nested Objects (DO NOT USE):
    {
      "crop_config": {
        "thumb_1": {
          "token": "<image_1>",
          "coordinates": [100, 200, 500, 600]
        }
      }
    }

    ❌ WRONG FORMAT - Extra Fields (DO NOT USE):
    {
      "crop_config": {
        "<image_1>": [100, 200, 500, 600]
      },
      "prettify": true,
      "annotated": false
    }

    Common Mistakes to Avoid:
    1. Using nested objects with "token" and "coordinates" fields
    2. Adding extra top-level parameters like "prettify", "annotated", "thumb_N"
    3. Using float numbers instead of integers for coordinates
    4. Providing less or more than 4 coordinates per crop box
    5. Including the "messages" parameter (it's auto-injected by the client)

    Usage Examples:

    Scenario 1: Crop a specific region from one image
      Question: "What text is written in the top-left corner of <image_1>?"
      
      Tool Call:
      {
        "crop_config": {
          "<image_1>": [0, 0, 400, 200]
        }
      }

    Scenario 2: Crop different regions from multiple images
      Question: "Compare the charts in <image_1> and <image_2>"
      
      Tool Call:
      {
        "crop_config": {
          "<image_1>": [100, 150, 600, 450],
          "<image_2>": [80, 100, 550, 400]
        }
      }

    Scenario 3: Crop the same region from multiple images
      Question: "Extract the title area from all three images"
      
      Tool Call:
      {
        "crop_config": {
          "<image_1>": [0, 0, 800, 100],
          "<image_2>": [0, 0, 800, 100],
          "<image_3>": [0, 0, 800, 100]
        }
      }

    Constraints:
    - Each crop box MUST be exactly four integers: [left, top, right, bottom]
    - Tokens must match existing <image_x> or <obs_x> entries in the conversation
    - Do NOT provide the "messages" parameter when calling this tool (auto-injected by client)

    Return value (JSON string):
    - The tool returns a JSON string with the following shape:

        {
          "status": "success" | "partial_success" | "error",
          "message": "<detailed status or error info>",
          "results": {
            "token": "<local path or url or error string>",
            ...
          },
          "images": [
            "<base64 or url>",
            ...
          ],
          "success_tokens": ["<token1>", "<token2>", ...],
          "failed_tokens": ["<token3>", ...]
        }

    - "status" values:
      - "success": All crops succeeded
      - "partial_success": Some crops succeeded, some failed (check message for details)
      - "error": All crops failed
      
    - "message" content:
      - For failures: Lists each failed token with specific error reason
      - For partial success: Lists failures + successful token list
      - For success: Confirms total cropped images and mentions <obs_N> tags
      
    - "success_tokens" / "failed_tokens": Arrays for programmatic handling (optional fields)

    - "results": Maps each input token to:
      - Success: Local file path or cloud URL
      - Failure: String starting with "Error:" containing detailed error message
      
    - "images": Array of successfully cropped images (base64 or URL), extracted by the 
      environment layer and injected as <obs_N> tags in the next message

    Notes:
    - Prefer calling CropImage whenever only a local region of an image is relevant
      to the question, instead of reasoning on the full image.
    """
    service = get_image_processor()
    if not service:
        return "Error: ImageProcessor not available."

    try:
        if not isinstance(crop_config, dict) or not crop_config:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        "crop_config must be a non-empty dictionary. "
                        "Example for single crop: {'<image_1>': [100, 200, 500, 600]}. "
                        "Example for multiple crops: {'<image_1>': [100, 200, 500, 600], '<image_2>': [50, 50, 300, 300]}."
                    ),
                },
                ensure_ascii=False,
            )

        # 规范化 token：去掉引号、尖括号、空白
        normalized_config: Dict[str, List[int]] = {}
        for raw_key, val in crop_config.items():
            if not isinstance(raw_key, str):
                return json.dumps(
                    {"status": "error", "message": "crop_config keys must be strings"},
                    ensure_ascii=False,
                )
            key = raw_key.strip()
            # 先去掉外层引号（如果有）
            if key.startswith('"') and key.endswith('"') and len(key) > 2:
                key = key[1:-1].strip()
            # 再去掉尖括号（如果有）
            if key.startswith("<") and key.endswith(">") and len(key) > 2:
                key = key[1:-1].strip()
            if not key:
                return json.dumps(
                    {"status": "error", "message": "crop_config contains empty token"},
                    ensure_ascii=False,
                )
            if key in normalized_config:
                return json.dumps(
                    {"status": "error", "message": f"duplicate token: {key}"},
                    ensure_ascii=False,
                )
            normalized_config[key] = val

        # 转换 crop_config 的 List 为 Tuple，适配 PIL
        formatted_config: Dict[str, Tuple[int, int, int, int]] = {}
        for k, v in normalized_config.items():
            if not isinstance(v, (list, tuple)) or len(v) != 4:
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            f"Invalid crop box format for token '{k}'. "
                            f"Expected a list of 4 integers [left, top, right, bottom], but received: {type(v).__name__}. "
                            f"Correct example: {{'<image_1>': [100, 200, 500, 600]}}. "
                            f"Do NOT use nested objects like {{'token': '...', 'coordinates': [...]}}."
                        ),
                    },
                    ensure_ascii=False,
                )
            try:
                box = tuple(int(x) for x in v)
            except Exception:
                return json.dumps(
                    {
                        "status": "error",
                        "message": (
                            f"Crop box coordinates for token '{k}' must be integers. "
                            f"Received: {v}. "
                            f"Correct format: {{'<image_1>': [100, 200, 500, 600]}} (all integers)."
                        ),
                    },
                    ensure_ascii=False,
                )
            formatted_config[k] = box

        # 注意：messages 参数依赖于 Gateway/Client 端的注入机制
        # 如果未注入，工具会尝试仅处理 content (如果被错误传入) 或报错
        results: Dict[str, str] = await service.batch_crop_images(
            crop_config=formatted_config, messages=messages
        )
        images: List[str] = []
        storage_mode = getattr(service.config, "STORAGE_MODE", "cloud")

        for _, val in results.items():
            if not isinstance(val, str):
                continue
            # cloud: URL 直接返回
            if storage_mode == "cloud" and val.startswith(("http://", "https://")):
                images.append(val)
            # local: 读取文件为 base64
            elif storage_mode == "local":
                path = Path(val)
                if path.exists():
                    try:
                        with open(path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                            images.append(b64)
                    except Exception as _:
                        pass

        # 状态初始化
        status = "success"
        message = ""
        failed_items = []
        success_items = []

        # 分类结果：遍历 results 字典，区分成功和失败
        for token, val in results.items():
            if isinstance(val, str) and val.startswith("Error:"):
                # 提取具体错误（去掉 "Error: " 前缀，避免冗余）
                error_detail = val[7:] if len(val) > 7 else val
                # 限制单条错误长度，防止超长消息
                if len(error_detail) > 200:
                    error_detail = error_detail[:197] + "..."
                failed_items.append(f"  - <{token}>: {error_detail}")
            else:
                success_items.append(token)

        # 构建详细消息
        if failed_items:
            total = len(results)
            failed_count = len(failed_items)
            success_count = len(success_items)
            
            if success_count > 0:
                # 部分成功
                status = "partial_success"
                message = f"{failed_count}/{total} crop(s) failed:\n" + "\n".join(failed_items)
                # 追加成功项提示
                success_tokens_str = ", ".join(f"<{t}>" for t in success_items)
                message += f"\n\nSuccessful crops ({success_count}): {success_tokens_str}"
                message += "\nSuccessful images will appear as <obs_N> tags in the next message."
            else:
                # 全部失败
                status = "error"
                message = f"All {total} crop(s) failed:\n" + "\n".join(failed_items)
        else:
            # 全部成功
            if success_items:
                success_count = len(success_items)
                message = f"Successfully cropped {success_count} image(s). Results will appear as <obs_N> tags in the next message."

        # 构建 payload（保持 results 和 images 字段不变）
        payload = {
            "status": status,
            "message": message,
            "results": results,
            "images": images,
            # 新增：便于程序化处理
            "success_tokens": success_items,
            "failed_tokens": [token for token in results.keys() if token not in success_items],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback

        print(f"[ERROR] CropImage failed: {e}\n{traceback.format_exc()}")
        return json.dumps(
            {"status": "error", "tool": "CropImage", "message": str(e)},
            ensure_ascii=False,
        )
