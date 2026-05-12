# src/envs/http_mcp_search_env.py
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from .http_mcp_env import HttpMCPEnv
import os
import base64
import json
import re

from utils.search_v2.config.settings import Config
from utils.search_v2.cloud_storage import CloudStorageService
from utils.image_download_logger import ImageDownloadLogger

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_SEARCH = """You are a Multi-Modal Search & Reasoning Agent equipped with capabilities in multimodal search, analysis, and reasoning.
Your task is to strictly rely on the provided tools to retrieve, verify, and reason over textual and image information, and to generate a final answer supported by verifiable evidence.

### **ReAct Framework**

You must operate using a ReAct-style reasoning and action loop. The reasoning process must be explicit, structured, and visible in text. Hidden or internal reasoning is strictly forbidden.
- Think step-by-step about what information is missing. 
- Decide whether a tool is required. 
- Call a tool to act. 
- Observe the tool result. 
- Continue reasoning and potentially call additional tools. 
- Repeat this Thought → Tool → Observation cycle until you reach a verifier-supported conclusion. 

You must not skip the loop or jump to the final answer directly. Reasoning must precede every tool call. You may not produce a final answer until the needed evidence has been collected from tools.

### **Overall Goal (Goal)**

- Use tools to collect verifiable information;
- Perform cross-modal analysis and reasoning based on tool outputs;
- Explicitly tag key intermediate reasoning results to support process-level scoring;
- Provide concise, deterministic, and reviewable final answers.

------

### **Available Tools**

**1. TextSearch:** Searches the web for relevant information based on a text query.

- Input: `query`: text query

- Return: list of search results, each including: `title`, `link`, `snippet`

**2. WebVisit:** Visits a specified webpage and returns structured text and images on the page.

- Input: `url`: target webpage URL

- Return: the main textual content of the page and available image information (URL / description)

**3. ImageSearch:** Searches for related images based on a text query.

- Input: `query`: text query

- Return: list of images, each including: `title`, `thumbnailUrl`, `link`

**4. ReverseImageSearch:** Performs reverse searches for images based on similarity to confirm origin or semantics.

- Input: a tagged image `<image_k>` or `<obs_i>`, or a direct `image_url`

- Return: list of related webpages or matching images, each including: `title`, `thumbnailUrl`, `link`

**5. CropImage:** Crops a designated region from a tagged image to focus on key information, supporting more accurate image understanding and image search.

- Input: `crop_config` - a dictionary mapping **existing** image tokens to crop coordinates

- Return: the cropped image (in a new `<obs_i>` form)

- Constraints: 
  - Cropped images cannot be cropped again
  - Crop coordinates must be integers
  - **Token keys must be EXISTING image tokens from the conversation** (refer to "Image Token Rules" section below)

- Format Examples:
  
  **Single crop:**
  ```json
  {
    "crop_config": {
      "<image_1>": [100, 200, 500, 600]
    }
  }
  ```

- **CRITICAL RULES**:
  1. Dictionary keys MUST be **existing** image tokens (`<image_1>`, `<image_2>`, `<obs_1>`, `<obs_2>`, etc.) that already appear in the conversation. See "Image Token Rules" section below.
  2. Do NOT invent new token names like "thumb_1", "crop_region", "dumpster_area", or any custom identifiers.
  3. Do NOT use nested objects. The format MUST be a flat dictionary: `{"<existing_token>": [left, top, right, bottom]}`.
  4. Each value must be a list of exactly 4 integers `[left, top, right, bottom]`, NOT a nested object like `{"coordinates": [...]}` or `{"token": "...", "coordinates": [...]}`.
  5. Do NOT add extra metadata fields like "type", "properties", "code", "human_readable_name", etc.

------

### **Image Token Rules**

- Input images to the task:
  `<image_1> ... </image_1>`, `<image_2> ... </image_2>`
- Tool-generated images:
  `<obs_1> ... </obs_1>`, `<obs_2> ... </obs_2>`
- Tool usage requirements:
  - Explicitly specify correct image tokens or URLs
  - Reuse existing image tokens to avoid redundant searches
  
------

### **Tool-Use Strategy**
- Do NOT call multiple tools in the same turn.
- Breadth-first then narrow: first build a candidate fact space, then gradually narrow the scope;
- Cross-verification: when ambiguity exists, a second source or image evidence must be employed;
- Visual priority: for image-related questions, relevant images must be processed and referenced;
- Efficiency: avoid redundant queries and fully utilize existing results;
- Image focus: when searching only part of an image or focusing on a region, use the `CropImage` tool.
- If no new useful information can be obtained from further tool calls, conclude the task using available evidence or return the mandatory failure output.
- Per interaction turn, only one tool may be invoked. Never call multiple tools in the same assistant turn. If another tool is needed, wait for the next turn after receiving the previous tool's output.
- You can only view the images returned by a tool call at the moment that tool responds. Images returned in earlier turns will be replaced with their corresponding titles in the conversation history.

------

### **Subgoal Tagging (for Process-Level Evaluation)**

**Tagging Principles**

Use `<SUBGOAL>` and `</SUBGOAL>` tags to wrap key intermediate reasoning conclusions or confirmed important information.

**`<SUBGOAL>` tags are intended to capture fact-level, verifiable intermediate conclusions, such as name, time, location, entities, events, relations, and quantities, that directly support the final answer. Sub-goals must not include guesses, common-sense assumptions, or reasoning shortcuts.**

**`<SUBGOAL>` tags should be generated dynamically and incrementally throughout the reasoning process.** As soon as an intermediate fact or validated insight is established—regardless of which reasoning step you are in—output the corresponding `<SUBGOAL>` tag immediately. **Do not wait until the final response turn to output all subgoals.**

**Before outputting <FINAL_ANSWER>, you must output ALL subgoals that have not been output yet.**

**Strict Requirements**

- Each subgoal must be:
  an intermediate reasoning result or a fact confirmed through tool usage;
- Must not be:
  operational steps, tool invocation descriptions, or low-level thoughts;
- Typically one sentence only, marking nodes of key information directly supporting the final answer;
- Avoid redundancy, repetition, or irrelevant tagging.
- Do NOT output <SUBGOAL> in the final turn unless it appears BEFORE the <FINAL_ANSWER>.

**Examples**
`<SUBGOAL>Confirm that the building in the image is the National Stadium (Bird’s Nest) in Beijing.</SUBGOAL>`
`<SUBGOAL>Confirm that construction of the National Stadium in Beijing began on December 24, 2003.</SUBGOAL>`

------

### **Final Answer Format (Mandatory)**

- Wrap the final conclusion (maximum 1–2 sentences) strictly within:
  `<FINAL_ANSWER>Your final answer</FINAL_ANSWER>`
- The `<FINAL_ANSWER>` content must not contain reasoning steps, subgoals, or source citations.
- If you determine that the required final answer cannot be found through available tool outputs, you must return:
`<FINAL_ANSWER>Failed. I cannot answer this question.</FINAL_ANSWER>`
- Do not fabricate or infer information beyond verified evidence.
- Do NOT output multiple <FINAL_ANSWER> blocks.


------

### **Answering Rules**

1. All factual information must be based on tool outputs;
2. When citing webpage information, specify the source (URL);
3. Place reasoning process and subgoal tags outside the final answer;
4. Final answers must be concise and clear;
5. You have at most 20 interaction rounds and must provide the final answer before round 20. If this cannot be done, you must declare failure using the required final-answer format: `<FINAL_ANSWER>Failed. I cannot answer this question.</FINAL_ANSWER>`.

"""


class HttpMCPSearchEnv(HttpMCPEnv):
    has_heavy_resource = False
    """
    Search-focused environment that inherits from HttpMCPEnv.

    This environment is specialized for 'utility' type resources (like search_v2),
    which are typically stateless and do not require heavy resource allocation (locking).
    """

    def __init__(
        self, model_name: str = "gpt-4.1-2025-04-14", parallel_degree: int = 1, **kwargs
    ):

        # 确保使用默认网关配置路径，除非外部覆盖
        if "gateway_config_path" not in kwargs:
            kwargs["gateway_config_path"] = str(Path(__file__).resolve().parents[2] / "configs" / "gateway_searchtools.example.json")

        # [新增] 默认白名单：只暴露 Search 相关工具给 Agent
        # 若外部未显式传入 `tool_whitelist`，则启用以下默认集合
        if not kwargs.get("tool_whitelist"):
            kwargs["tool_whitelist"] = [
                "TextSearch",
                "WebVisit",
                "ReverseImageSearch",
                "CropImage",
                "ImageSearch",
            ]

        # 初始化父类
        super().__init__(
            model_name=model_name, parallel_degree=parallel_degree, **kwargs
        )

        # 专用于搜索场景的图片标记计数器（用于生成 <obs_i> 标签）
        self._obs_counter = 0
        self._cloud_storage: Optional[CloudStorageService] = None
        self._storage_mode = Config.STORAGE_MODE

        # 保存当前对话历史，用于工具调用时自动注入messages参数
        self._current_messages: Optional[List[Dict[str, Any]]] = None

        # Image download logger for tracking download statistics
        self._image_logger = ImageDownloadLogger()

        # [关键兼容性设置]
        # Search V2 (utility) 是无状态服务，不需要向 Resource Manager 申请锁定/分配。
        # 父类 HttpMCPEnv 默认会把所有非 system 的 module 加入 active_resources 并尝试 allocate。
        # 这里我们需要清空 active_resources，避免 allocate_batch_resources 报错或做无用功。
        self.active_resources = []

        logger.info(
            f"HttpMCPSearchEnv initialized for {self.worker_id} (Stateless Mode)"
        )

    @property
    def mode(self) -> str:
        """定义新的环境模式名称"""
        return "http_mcp_search"

    def get_system_prompt(
        self,
        task_question: Optional[str] = None,
        max_turns: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        重写 System Prompt，注入搜索专用的提示词和工具描述
        """
        prompt = SYSTEM_PROMPT_SEARCH

        # 动态注入工具描述 (由父类从 MCP Server 获取)
        tool_descriptions = self.get_tool_descriptions()
        if tool_descriptions:
            prompt += f"\n\n## Available Tools\n{tool_descriptions}"

        # 注入当前任务
        if task_question:
            prompt += f"\n\n## Current Task\n{task_question}"

        if isinstance(max_turns, int) and max_turns > 0:
            prompt += (
                "\n\n## Turn Budget\n"
                f"- Max turns: {max_turns}\n"
                "- You will receive system message reminders when the remaining number of turns falls below five. \n"
                "- You must provide the final answer in `<FINAL_ANSWER>...</FINAL_ANSWER>` before the remaining turns reach 0. \n"
                "- If this is not possible, you must explicitly declare failure using the required format: `<FINAL_ANSWER>Failed. I cannot answer this question.</FINAL_ANSWER>`.\n"
            )

        return prompt

    def _get_proxies(self) -> Optional[Dict[str, str]]:
        """Read proxy configuration from environment variables."""
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if http_proxy or https_proxy:
            return {
                "http": http_proxy,
                "https": https_proxy or http_proxy,
            }
        return None

    def _generate_google_tbn_variants(self, url: str) -> List[str]:
        """
        Generate URL variants for Google thumbnail servers (tbn0-tbn3).
        If URL is not a Google thumbnail URL, returns [url].
        """
        import re
        pattern = r"encrypted-tbn\d"
        if not re.search(pattern, url):
            return [url]
        
        variants = []
        for server in Config.GOOGLE_TBN_SERVERS:
            variant = re.sub(pattern, f"encrypted-tbn{server}", url)
            variants.append(variant)
        return variants

    def _download_single_url(self, url: str, session, headers: dict, proxies: Optional[dict], 
                              connect_timeout: int, read_timeout: int) -> Optional[str]:
        """
        Try to download a single URL and return base64 encoded content.
        Returns None on failure.
        """
        try:
            response = session.get(
                url,
                headers=headers,
                proxies=proxies,
                timeout=(connect_timeout, read_timeout),
                verify=True
            )
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        except Exception:
            pass
        
        try:
            response = session.get(
                url,
                headers=headers,
                proxies=proxies,
                timeout=(connect_timeout, read_timeout),
                verify=False
            )
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        except Exception:
            pass
        
        return None

    def _url_to_base64(self, url: str, timeout: int = None) -> Optional[str]:
        """
        Download image URL and convert to base64 with retry mechanism, proxy support,
        and Google thumbnail server rotation.
        
        For Google thumbnail URLs (encrypted-tbnX.gstatic.com), tries all servers (0-3)
        in rotation for multiple rounds before giving up.
        
        Args:
            url: Image HTTP/HTTPS URL
            timeout: Read timeout in seconds (default from config)
        
        Returns:
            base64 encoded image string, or None on failure
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        if timeout is None:
            timeout = Config.IMAGE_DOWNLOAD_READ_TIMEOUT
        connect_timeout = Config.IMAGE_DOWNLOAD_CONNECT_TIMEOUT
        rotation_rounds = Config.IMAGE_DOWNLOAD_SERVER_ROTATION_ROUNDS
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
            "Connection": "keep-alive",
        }
        
        proxies = self._get_proxies()
        
        session = requests.Session()
        retry_strategy = Retry(
            total=1,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        url_variants = self._generate_google_tbn_variants(url)
        is_google_tbn = len(url_variants) > 1
        
        last_error = None
        total_attempts = 0
        
        for round_num in range(rotation_rounds):
            for variant_url in url_variants:
                total_attempts += 1
                try:
                    response = session.get(
                        variant_url,
                        headers=headers,
                        proxies=proxies,
                        timeout=(connect_timeout, timeout),
                        verify=True
                    )
                    response.raise_for_status()
                    img_b64 = base64.b64encode(response.content).decode("utf-8")
                    self._image_logger.log_success(url)
                    if is_google_tbn and variant_url != url:
                        logger.debug(f"[{self.worker_id}] Downloaded via alternate server: {variant_url[:60]}...")
                    else:
                        logger.debug(f"[{self.worker_id}] Downloaded image: {url[:50]}...")
                    session.close()
                    return img_b64
                    
                except requests.exceptions.SSLError:
                    try:
                        response = session.get(
                            variant_url,
                            headers=headers,
                            proxies=proxies,
                            timeout=(connect_timeout, timeout),
                            verify=False
                        )
                        response.raise_for_status()
                        img_b64 = base64.b64encode(response.content).decode("utf-8")
                        self._image_logger.log_success(url)
                        session.close()
                        return img_b64
                    except Exception as e:
                        last_error = str(e)
                        
                except requests.exceptions.HTTPError as e:
                    last_error = str(e)
                    status_code = e.response.status_code if e.response is not None else 0
                    if status_code in (404, 410):
                        continue
                    if status_code == 403:
                        time.sleep(0.5)
                        continue
                        
                except (requests.Timeout, requests.ConnectionError, ConnectionResetError) as e:
                    last_error = str(e)
                    if is_google_tbn:
                        time.sleep(0.3)
                    continue
                    
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if round_num < rotation_rounds - 1:
                delay = 1 + round_num
                logger.debug(f"[{self.worker_id}] Round {round_num + 1} failed, waiting {delay}s before next round...")
                time.sleep(delay)
        
        session.close()
        self._image_logger.log_failure(url, last_error or "Unknown error", total_attempts)
        logger.warning(f"[{self.worker_id}] Failed to download image after {total_attempts} attempts: {url[:50]}... - {last_error}")
        return None

    def _build_tool_image_message(self, image_list: List[str]) -> List[Dict[str, Any]]:
        """
        Build image message blocks with <obs_i> markers.
        Failed downloads will show placeholder text instead of breaking the flow.
        """
        image_format = getattr(Config, "IMAGE_MESSAGE_FORMAT", "url")
        
        obs_blocks: List[Dict[str, Any]] = []
        for img in image_list:
            self._obs_counter += 1
            open_tag = f"<obs_{self._obs_counter}>"
            close_tag = f"</obs_{self._obs_counter}>"
            obs_blocks.append({"type": "text", "text": open_tag})
            
            if isinstance(img, str) and img.startswith(("http://", "https://")):
                if image_format == "base64":
                    img_b64 = self._url_to_base64(img)
                    if img_b64:
                        img_payload = {
                            "url": f"data:image/png;base64,{img_b64}",
                            "detail": "high"
                        }
                        obs_blocks.append({"type": "image_url", "image_url": img_payload})
                    else:
                        obs_blocks.append({
                            "type": "text",
                            "text": "[Image failed to load]"
                        })
                        logger.warning(
                            f"[{self.worker_id}] Image failed, using placeholder: {img[:50]}..."
                        )
                else:
                    img_payload = {"url": img, "detail": "low"}
                    obs_blocks.append({"type": "image_url", "image_url": img_payload})
            else:
                img_payload = {"url": f"data:image/png;base64,{img}", "detail": "high"}
                obs_blocks.append({"type": "image_url", "image_url": img_payload})
            
            obs_blocks.append({"type": "text", "text": close_tag})
        return [{"role": "user", "content": obs_blocks}] if obs_blocks else []

    def env_close(self):
        """
        Close environment and print image download statistics.
        """
        if hasattr(self, '_image_logger') and self._image_logger:
            summary = self._image_logger.get_summary()
            if summary.get("total_attempts", 0) > 0:
                self._image_logger.print_summary()
                if self._image_logger.failures:
                    log_path = self._image_logger.save_failure_log()
                    if log_path:
                        logger.info(f"[{self.worker_id}] Image failure log saved to: {log_path}")
        
        super().env_close()

    def _call_tool_sync(self, name: str, arguments: Any):
        """
        仅在搜索环境中：对搜索工具和裁切工具额外抽取图片 URL/base64 作为 <obs_i> 注入。
        同时自动为需要messages参数的工具注入当前对话历史。
        """
        # 需要messages参数的工具列表
        tools_requiring_messages = {"ReverseImageSearch", "CropImage"}

        # 确保参数是字典格式
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        elif not isinstance(arguments, dict):
            arguments = {}

        # 对于需要messages参数的工具，如果未提供则自动注入
        if name in tools_requiring_messages:
            if "messages" not in arguments or arguments.get("messages") is None:
                if self._current_messages is not None:
                    # 深拷贝messages，避免工具修改原始数据
                    import copy

                    arguments["messages"] = copy.deepcopy(self._current_messages)
                    
                    # 详细调试：检查每条消息的结构
                    msg_structure = []
                    for i, msg in enumerate(arguments["messages"]):
                        role = msg.get("role", "unknown")
                        content = msg.get("content")
                        
                        if isinstance(content, list):
                            # 统计 content 中各类型块的数量
                            type_counts = {}
                            has_image = False
                            image_tokens = []
                            
                            for block in content:
                                if isinstance(block, dict):
                                    block_type = block.get("type", "unknown")
                                    type_counts[block_type] = type_counts.get(block_type, 0) + 1
                                    
                                    if block_type == "image_url":
                                        has_image = True
                                    elif block_type == "text":
                                        # 检查是否包含 image token
                                        text = block.get("text", "")
                                        import re
                                        tokens = re.findall(r'<(image_\d+|obs_\d+)>', text)
                                        if tokens:
                                            image_tokens.extend(tokens)
                            
                            msg_structure.append({
                                "idx": i,
                                "role": role,
                                "content_type": "list",
                                "content_len": len(content),
                                "blocks": type_counts,
                                "has_image": has_image,
                                "tokens": image_tokens
                            })
                        else:
                            msg_structure.append({
                                "idx": i,
                                "role": role,
                                "content_type": type(content).__name__,
                                "content_preview": str(content)[:100] if content else "None"
                            })
                    
                    logger.info(
                        f"[{self.worker_id}] [SearchEnv] Auto-injected messages for {name}: "
                        f"total={len(arguments['messages'])}, structure={msg_structure}"
                    )
                else:
                    logger.warning(
                        f"[{self.worker_id}] [SearchEnv] {name} requires messages but _current_messages is None"
                    )

        result = super()._call_tool_sync(name, arguments)

        if not isinstance(result, dict):
            return result

        images = (
            list(result.get("images", []))
            if isinstance(result.get("images", []), list)
            else []
        )
        text_payload = result.get("text") or ""

        def _extract_from_list(items: List[Dict[str, Any]]):
            extracted_count = 0
            for item in items:
                if not isinstance(item, dict):
                    continue
                thumb = item.get("thumbnailUrl") or item.get("thumbnail")
                if thumb and thumb.strip():  # 确保 thumbnailUrl 不为空
                    images.append(thumb)
                    extracted_count += 1
                    continue
                img = item.get("imageUrl") or item.get("image_url")
                if img and img.strip():  # 确保 imageUrl 不为空
                    images.append(img)
                    extracted_count += 1
            
            if extracted_count == 0 and items:
                logger.warning(
                    f"[{self.worker_id}] No valid thumbnailUrl/imageUrl found in {len(items)} items for {name}"
                )

        if name in {"TextSearch", "ImageSearch", "ReverseImageSearch"} and text_payload:
            try:
                parsed = json.loads(text_payload)
                items = parsed if isinstance(parsed, list) else []
                if items:
                    logger.debug(
                        f"[{self.worker_id}] {name} parsed {len(items)} items, extracting images..."
                    )
                    _extract_from_list(items)
                    logger.debug(
                        f"[{self.worker_id}] {name} extracted {len(images)} images from {len(items)} items"
                    )
                else:
                    logger.warning(
                        f"[{self.worker_id}] {name} returned empty list or non-list result. "
                        f"text_payload preview: {text_payload[:200]}"
                    )
            except json.JSONDecodeError as e:
                logger.error(
                    f"[{self.worker_id}] Failed to parse {name} JSON response: {e}. "
                    f"text_payload preview: {text_payload[:200]}"
                )
            except Exception as e:
                logger.error(
                    f"[{self.worker_id}] Unexpected error extracting images from {name}: {e}. "
                    f"text_payload preview: {text_payload[:200]}"
                )

        if name == "CropImage" and text_payload:
            try:
                parsed = json.loads(text_payload)
                if isinstance(parsed, dict):
                    img_list = parsed.get("images", [])
                    if isinstance(img_list, list):
                        images.extend([img for img in img_list if isinstance(img, str)])
            except Exception:
                pass

        dedup = []
        seen = set()
        for img in images:
            if img and img not in seen:
                seen.add(img)
                dedup.append(img)
            if len(dedup) >= 5:
                break
        result["images"] = dedup
        return result

    def _append_input_images(self, user_content: List[Dict[str, Any]]) -> None:
        """
        搜索环境：为任务输入图片添加 <image_k> ... </image_k> 标记，便于后续引用/裁剪。
        """
        input_images = getattr(self, "input_images", None)
        if not isinstance(input_images, list) or not input_images:
            return

        for idx, img in enumerate(input_images, start=1):
            if not isinstance(img, dict):
                continue
            open_token = f"<image_{idx}>"
            close_token = f"</image_{idx}>"
            b64 = img.get("b64")
            url = img.get("url")
            user_content.append({"type": "text", "text": open_token})
            if b64:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    }
                )
            elif url:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}}
                )
            user_content.append({"type": "text", "text": close_token})

        # 记录已注入的 image token，便于排查工具侧缺少 token 的问题
        try:
            injected_tokens = [f"<image_{i}>" for i in range(1, len(input_images) + 1)]
            logger.info(
                f"[{self.worker_id}] [SearchEnv] Injected image tokens into user message: {injected_tokens}"
            )
        except Exception:
            pass

    def _log_user_message_tokens(self, user_content: List[Dict[str, Any]]) -> None:
        """
        记录首条用户消息中的 token 摘要，便于排查模型未带 token 调用工具的问题。
        仅在搜索环境生效，避免污染基类。
        """
        tokens = []
        for block in user_content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                txt = block.get("text", "")
                if isinstance(txt, str) and txt.startswith("<") and txt.endswith(">"):
                    tokens.append(txt)
        if tokens:
            logger.info(
                f"[{self.worker_id}] [SearchEnv] First user message tokens: {tokens}"
            )

    def _append_input_images_and_log(self, user_content: List[Dict[str, Any]]) -> None:
        """
        包装原注入逻辑并追加 user 消息 token 日志。
        """
        self._append_input_images(user_content)
        self._log_user_message_tokens(user_content)

    # =========================================================================
    # 参数规范化：确保工具调用参数与 MCP schema 描述兼容
    # =========================================================================
    def _normalize_tool_arguments(self, name: str, arguments: Any) -> Any:
        """
        针对搜索工具做参数格式修正，避免因为描述/类型差异导致调用失败。
        - 统一 k 为 int，缺省回填
        - TextSearch / WebVisit: 默认 region='us'
        - ReverseImageSearch: 若 image_token 未包含尖括号则保留原值，工具侧仍按纯 token 查找
        - CropImage: 确保 crop_config 存在且为 dict
        """
        if not isinstance(arguments, dict):
            return arguments

        args = dict(arguments)

        if name in {"TextSearch", "WebVisit"}:
            if "k" in args:
                try:
                    args["k"] = int(args["k"])
                except Exception:
                    pass
            else:
                args["k"] = 5
            if "region" not in args or not args.get("region"):
                args["region"] = "us"

        if name == "ImageSearch":
            if "k" in args:
                try:
                    args["k"] = int(args["k"])
                except Exception:
                    pass
            else:
                args["k"] = 5

        if name == "ReverseImageSearch":
            if "k" in args:
                try:
                    args["k"] = int(args["k"])
                except Exception:
                    pass
            else:
                args["k"] = 3
            # Normalize image_token to accept both "<image_1>" and "image_1"
            if "image_token" in args and isinstance(args.get("image_token"), str):
                token = args["image_token"].strip()
                if token.startswith("<") and token.endswith(">") and len(token) > 2:
                    token = token[1:-1].strip()
                args["image_token"] = token
                logger.info(
                    f"[{self.worker_id}] [SearchEnv] ReverseImageSearch token arg={args['image_token']}"
                )

        if name == "CropImage":
            if "crop_config" not in args or not isinstance(
                args.get("crop_config"), dict
            ):
                args["crop_config"] = {}

        return args

    def execute_tool(self, tool_name: str, params: Any, **kwargs):
        norm = self._normalize_tool_arguments(
            tool_name, params if not isinstance(params, str) else json.loads(params)
        )
        return super().execute_tool(tool_name, norm, **kwargs)

    async def execute_tool_async(self, tool_name: str, params: Any, **kwargs):
        norm = self._normalize_tool_arguments(
            tool_name, params if not isinstance(params, str) else json.loads(params)
        )
        return await super().execute_tool_async(tool_name, norm, **kwargs)

    def _upload_local_image_to_cloud(self, file_path: str) -> Optional[str]:
        """Upload a local image to cloud storage when storage_mode=cloud."""
        try:
            if self._cloud_storage is None:
                self._cloud_storage = CloudStorageService()

            loop = self._loop
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop

            result = loop.run_until_complete(
                self._cloud_storage.upload_single_image(Path(file_path))
            )
            if isinstance(result, dict):
                return result.get("url")
        except Exception as e:
            logger.warning(f"[{self.worker_id}] Upload input image failed: {e}")
        return None

    def _load_gateway_config(self, config_path: str) -> Dict[str, Any]:
        """
        重写配置加载逻辑：只加载 'utility' 类型的模块 (对应 search_tools)
        """
        config = super()._load_gateway_config(config_path)

        if "modules" in config:
            original_count = len(config["modules"])
            # 过滤只保留 utility 类型的模块 (我们在 server 端将 search 定义为了 utility)
            config["modules"] = [
                module
                for module in config["modules"]
                if module.get("resource_type") == "utility"
            ]

            # 如果没有找到 utility，尝试找包含 search_tools 的模块作为回退
            if not config["modules"]:
                config["modules"] = [
                    module
                    for module in super()
                    ._load_gateway_config(config_path)
                    .get("modules", [])
                    if any("search" in g for g in module.get("tool_groups", []))
                ]

            filtered_count = len(config["modules"])
            if filtered_count < original_count:
                logger.info(
                    f"[{self.worker_id}] Gateway config filtered: "
                    f"{filtered_count}/{original_count} modules (Search/Utility only)"
                )

        return config

    def _run_conversation(
        self,
        question: str,
        model_name: str,
        max_turns: int,
        max_retries: int,
        logger: logging.Logger,
        task_timeout: Optional[float] = None,
        task_start_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        重写对话方法，实时保存messages引用以便工具调用时使用。
        复制父类逻辑，但在每次更新messages时同步更新self._current_messages。
        """
        from openai.types.chat import ChatCompletionMessageParam
        import time
        from utils.task_timeout import check_execution_timeout, TaskTimeoutError

        system_prompt = self.get_system_prompt(question, max_turns=max_turns)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
        ]

        # 实时更新_current_messages
        self._current_messages = messages

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"Question: {question}\n"}
        ]

        # 注入初始观察
        initial_obs = getattr(self, "initial_observation", None)

        if initial_obs and isinstance(initial_obs, dict):
            if initial_obs.get("screenshot"):
                user_content.append(
                    {
                        "type": "text",
                        "text": "Here is the initial screen state of the computer:",
                    }
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{initial_obs['screenshot']}",
                            "detail": "high",
                        },
                    }
                )

            if initial_obs.get("accessibility_tree"):
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Accessibility Tree:\n{initial_obs['accessibility_tree']}",
                    }
                )

        # 钩子：由子类决定是否以及如何注入任务输入图片
        self._append_input_images(user_content)

        messages.append({"role": "user", "content": user_content})
        # 实时更新_current_messages
        self._current_messages = messages

        client = self._get_openai_client()
        turn_count = 0
        final_turn_retry_count = 0  # 最后一轮重试计数器
        max_final_turn_retries = 2  # 最后一轮最多重试 2 次
        final_turn_prompt_added = False  # 标记是否已添加最后一轮提示

        try:
            while turn_count < max_turns:
                remaining_turns = max_turns - turn_count
                is_final_turn = (remaining_turns == 1)
                
                # 只在第一次进入最后一轮时添加提示，避免重试时重复添加
                if remaining_turns == 1 and not final_turn_prompt_added:
                    # 强化版最后一轮提示 - 方案C：提供示例格式
                    final_turn_prompt_added = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "⚠️ [MANDATORY - FINAL TURN] ⚠️\n\n"
                                "You MUST output in this EXACT format:\n\n"
                                "<SUBGOAL>First confirmed fact from your research</SUBGOAL>\n"
                                "<SUBGOAL>Second confirmed fact from your research</SUBGOAL>\n"
                                "<SUBGOAL>Third confirmed fact (if any)</SUBGOAL>\n"
                                "...\n"
                                "<FINAL_ANSWER>Your final answer based on the facts above</FINAL_ANSWER>\n\n"
                                "RULES:\n"
                                "✓ Output ALL facts you confirmed through tools as <SUBGOAL> tags\n"
                                "✓ At least 1 <SUBGOAL> is REQUIRED before <FINAL_ANSWER>\n"
                                "✓ Each <SUBGOAL> = one verified fact (entity, date, location, quantity, etc.)\n"
                                "✗ NO tool calls allowed (tools are DISABLED)\n\n"
                                "If you cannot answer: <FINAL_ANSWER>Failed. I cannot answer this question.</FINAL_ANSWER>"
                            ),
                        }
                    )
                elif remaining_turns <= 5 and remaining_turns > 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"⚠️ Turn reminder: {remaining_turns} turn(s) remaining. "
                                "Start preparing your final answer. Remember to tag all key facts with <SUBGOAL> tags as you confirm them. "
                                "You must provide <FINAL_ANSWER>...</FINAL_ANSWER> before turns run out."
                            ),
                        }
                    )
                self._current_messages = messages

                if task_timeout and task_start_time:
                    if check_execution_timeout(
                        task_start_time, task_timeout, "current_task", self.worker_id
                    ):
                        raise TaskTimeoutError(
                            f"Task timeout after {time.time() - task_start_time:.1f}s "
                            f"(limit: {task_timeout}s) at turn {turn_count}"
                        )

                retry = 0
                while retry < max_retries:
                    try:
                        request_kwargs: Dict[str, Any] = {}
                        
                        # 最后一轮禁用工具
                        if is_final_turn:
                            tools = None  # 禁用所有工具
                            # 清除 tool_choice 参数（避免 API 冲突）
                            request_kwargs.pop("tool_choice", None)
                        else:
                            tool_choice = self.config.get("tool_choice")
                            if tool_choice is not None:
                                request_kwargs["tool_choice"] = tool_choice
                            tools = self.get_tool_schemas()
                        logger.info(
                            f"[{self.worker_id}] [SearchEnv] Chat request: "
                            f"turn={turn_count}, retry={retry}, is_final_turn={is_final_turn}, "
                            f"tools_count={0 if tools is None else len(tools)}, "
                            f"tool_choice={request_kwargs.get('tool_choice')!r}, "
                            f"last_role={messages[-1].get('role') if messages else None}"
                        )

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
                        logger.info(
                            f"[{self.worker_id}] [SearchEnv] Chat response: "
                            f"has_content={bool((assistant_message.content or '').strip())}, "
                            f"tool_calls_count={len(assistant_message.tool_calls or [])}, "
                            f"finish_reason={response.choices[0].finish_reason}"
                        )
                        
                        # 最后一轮特殊处理：强制忽略 tool_calls，只看文本内容
                        if is_final_turn:
                            content = assistant_message.content or ""
                            has_text_content = bool(content.strip())
                            
                            if has_text_content:
                                # 有文本内容，强制清除 tool_calls 并接受响应
                                if assistant_message.tool_calls:
                                    logger.warning(
                                        f"⚠️ [FINAL TURN] Model returned tool_calls with text content. "
                                        f"Ignoring tool_calls and accepting text response."
                                    )
                                # 保存时清除 tool_calls
                                msg_dict = assistant_message.model_dump()
                                msg_dict["tool_calls"] = None
                                messages.append(msg_dict)
                                self._current_messages = messages
                                return messages
                            else:
                                # 没有文本内容，需要重试
                                if final_turn_retry_count < max_final_turn_retries:
                                    final_turn_retry_count += 1
                                    logger.warning(
                                        f"⚠️ [FINAL TURN] Model returned no text content. "
                                        f"Retrying ({final_turn_retry_count}/{max_final_turn_retries})..."
                                    )
                                    # 不添加任何消息，直接重试（continue 到内层 while retry 的下一次）
                                    continue
                                else:
                                    # 达到最大重试次数，返回带失败标记的响应
                                    logger.error(
                                        f"❌ [FINAL TURN FAILURE] Model returned no text content after {max_final_turn_retries} retries."
                                    )
                                    msg_dict = assistant_message.model_dump()
                                    msg_dict["tool_calls"] = None
                                    msg_dict["content"] = "<FINAL_ANSWER>Failed. I cannot answer this question.</FINAL_ANSWER>"
                                    messages.append(msg_dict)
                                    self._current_messages = messages
                                    return messages
                        
                        # 非最后一轮：正常处理
                        messages.append(assistant_message.model_dump())
                        self._current_messages = messages

                        if assistant_message.tool_calls:
                            # 非最后一轮：正常处理工具调用
                            if len(assistant_message.tool_calls) > 1:
                                logger.warning(
                                    "⚠️ Detected multiple tool_calls in one assistant turn; "
                                    "truncating to the first call to avoid API mismatch."
                                )
                                assistant_message.tool_calls = assistant_message.tool_calls[:1]
                                if messages[-1].get("tool_calls"):
                                    messages[-1]["tool_calls"] = messages[-1]["tool_calls"][:1]

                            if messages[-1]["content"] is None:
                                messages[-1]["content"] = ""

                            for tool_call in assistant_message.tool_calls:
                                tool_name = tool_call.function.name
                                raw_tool_args = tool_call.function.arguments
                                logger.info(
                                    f"[{self.worker_id}] [SearchEnv] Tool call payload: "
                                    f"tool_name={tool_name}, tool_call_id={tool_call.id}, "
                                    f"raw_arguments={raw_tool_args!r}"
                                )
                                try:
                                    tool_args = json.loads(raw_tool_args)
                                except json.JSONDecodeError:
                                    logger.error(
                                        f"[{self.worker_id}] [SearchEnv] Invalid tool arguments JSON "
                                        f"for tool={tool_name}, tool_call_id={tool_call.id}",
                                        exc_info=True,
                                    )
                                    raise

                                logger.info(f"🔧 {tool_name}")

                                tool_output = None
                                tool_exception = None
                                try:
                                    tool_output = self.execute_tool(tool_name, tool_args)
                                except Exception as exc:
                                    tool_exception = exc
                                    tool_output = f"[Error executing {tool_name}: {exc}]"
                                finally:
                                    if tool_output is None:
                                        tool_output = "[No output from tool]"

                                    if (
                                        isinstance(tool_output, dict)
                                        and "images" in tool_output
                                    ):
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
                                    self._current_messages = messages

                                    if image_list:
                                        image_msgs = self._build_tool_image_message(
                                            image_list
                                        )
                                        if image_msgs:
                                            messages.extend(image_msgs)
                                            self._current_messages = messages

                                if tool_exception:
                                    raise tool_exception

                        else:
                            # 模型没有调用工具,检查是否包含最终答案标记
                            content = assistant_message.content or ""
                            
                            # 情况1: 包含 FINAL_ANSWER 标记 → 正常结束
                            if re.search(r'<FINAL_ANSWER>.*?</FINAL_ANSWER>', content, re.DOTALL | re.IGNORECASE):
                                logger.info(f"Turn {turn_count}: Final answer provided without tool call")
                                return messages
                            
                            # 情况2: 没有标记 → 提示并继续
                            logger.warning(f"Turn {turn_count}: No tool call and no final answer tag, prompting to continue")
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You didn't call any tool or provide a final answer. Please:\n"
                                    "- If you have enough information: output all confirmed facts as <SUBGOAL> tags, "
                                    "then provide <FINAL_ANSWER>your answer</FINAL_ANSWER>\n"
                                    "- If you need more information: call a tool to search\n"
                                    "- If you cannot find the answer: output <FINAL_ANSWER>Failed. I cannot answer this question.</FINAL_ANSWER>"
                                )
                            })
                            self._current_messages = messages
                            # 继续循环 (不return,让while循环继续)

                        break  # 成功则跳出重试循环

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
        client,
        model_name: str,
        messages: List[Any],
        tools: Optional[List[Any]],
        request_kwargs: Dict[str, Any],
    ):
        try:
            logger.info(
                f"[{self.worker_id}] [SearchEnv] _call_chat_completion request: "
                f"model={model_name}, message_count={len(messages)}, "
                f"tools_type={type(tools).__name__}, "
                f"request_kwargs_keys={list(request_kwargs.keys())}"
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
                f"[{self.worker_id}] [SearchEnv] TypeError in chat completion: {msg}; "
                f"request_kwargs={request_kwargs}"
            )
            for key in ("tool_choice",):
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

    def run_task(
        self, task: Dict[str, Any], agent_config: Dict[str, Any], logger: logging.Logger
    ) -> Dict[str, Any]:
        """
        扩展：支持问题相关图片输入。
        - 从 task.metadata.images 读取图片（本地路径或 URL）
        - 本地文件存在则转为 base64；URL 直接注入
        - 基类会在用户消息中按 <img_n> 注入这些内容
        """
        # 清理之前的注入和对话历史
        self.input_images = []
        self._current_messages = None
        self._obs_counter = 0

        try:
            md = (
                task.get("metadata", {})
                if isinstance(task.get("metadata"), dict)
                else {}
            )
            imgs = md.get("images")
            images: list[str] = []
            if isinstance(imgs, str):
                images = [imgs]
            elif isinstance(imgs, list):
                images = [x for x in imgs if isinstance(x, str)]

            logger.info(
                f"[{self.worker_id}] [SearchEnv] Found {len(images)} image(s) in task metadata"
            )

            for p in images:
                s = p.strip()
                if s.startswith("http://") or s.startswith("https://"):
                    logger.info(
                        f"[{self.worker_id}] [SearchEnv] Injecting image URL: {s}"
                    )
                    self.input_images.append({"url": s})
                    continue
                if os.path.exists(s):
                    try:
                        with open(s, "rb") as f:
                            b64_raw = base64.b64encode(f.read()).decode("utf-8")
                            b64 = self._compress_base64_image(b64_raw)
                            self.input_images.append({"b64": b64})
                            logger.info(
                                f"[{self.worker_id}] [SearchEnv] Injected local image as base64: {s}"
                            )
                    except Exception:
                        # 忽略单个文件失败，但记录日志便于排查
                        logger.warning(
                            f"[{self.worker_id}] [SearchEnv] Failed to read image file: {s}",
                            exc_info=True,
                        )
                        pass
                else:
                    logger.warning(
                        f"[{self.worker_id}] [SearchEnv] Image path not found: {s}"
                    )
        except Exception:
            self.input_images = []
            logger.warning(
                f"[{self.worker_id}] [SearchEnv] Failed to prepare input images",
                exc_info=True,
            )

        # 调用父类方法，但需要拦截_run_conversation来实时更新messages引用
        result = super().run_task(task, agent_config, logger)
        # 不再额外保存 subgoal_summaries，单一结果文件由基类处理
        return result

    def _extract_final_answer_tagged(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        从对话中提取 <FINAL_ANSWER>...</FINAL_ANSWER> 的内容。
        返回首个匹配的纯文本；若不存在则返回 None。
        """
        if not messages:
            return None
        pattern = re.compile(
            r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE
        )
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        return None

    # 旧的 _save_subgoal_summary 输出已移除，统一由基类 _save_conversation_log 生成单一结果文件
