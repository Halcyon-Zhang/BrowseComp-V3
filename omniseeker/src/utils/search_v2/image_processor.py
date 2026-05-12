import asyncio
import io
import os
import base64
import requests
import re
import time
import uuid
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path

from .config.settings import Config
from .cloud_storage import CloudStorageService

class ImageProcessor:
    """
    处理 OpenAI 消息中的图片，支持基于 Token 的定位和裁切。
    
    【内存优化版】
    - 解析阶段仅提取元数据，不加载图片实体。
    - 裁切阶段按需加载，用完即毁，最小化内存占用。
    """
    
    # 添加裁切产物的特征标记
    CROP_MARKER = "mcp_derived_crop_"
    
    def __init__(self):
        self.config = Config()
        self.cloud_storage = None
        self.config.create_directories()

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
        pattern = r"encrypted-tbn\d"
        if not re.search(pattern, url):
            return [url]
        
        variants = []
        for server in self.config.GOOGLE_TBN_SERVERS:
            variant = re.sub(pattern, f"encrypted-tbn{server}", url)
            variants.append(variant)
        return variants

    def _load_image_from_source(self, image_detail: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Load image from source with retry mechanism and proxy support.
        Only called when image processing is actually needed.
        """
        url_str = image_detail.get("image_url", {}).get("url", "")
        
        if not url_str:
            return None

        try:
            if url_str.startswith("data:image"):
                if "," in url_str:
                    header, encoded = url_str.split(",", 1)
                else:
                    encoded = url_str
                return Image.open(io.BytesIO(base64.b64decode(encoded)))
            
            elif url_str.startswith(("http://", "https://")):
                return self._download_image_with_retry(url_str)
            
            else:
                print(f"[WARN] Unknown image format prefix: {url_str[:30]}...")
                return None

        except Exception as e:
            print(f"[ERROR] Failed to load image data: {e}")
            return None

    def _download_image_with_retry(self, url: str) -> Optional[Image.Image]:
        """
        Download image from URL with retry mechanism, proxy support,
        and Google thumbnail server rotation.
        """
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        connect_timeout = self.config.IMAGE_DOWNLOAD_CONNECT_TIMEOUT
        read_timeout = self.config.IMAGE_DOWNLOAD_READ_TIMEOUT
        rotation_rounds = self.config.IMAGE_DOWNLOAD_SERVER_ROTATION_ROUNDS
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
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
                    resp = session.get(
                        variant_url,
                        headers=headers,
                        proxies=proxies,
                        timeout=(connect_timeout, read_timeout),
                        verify=True
                    )
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content))
                    if is_google_tbn and variant_url != url:
                        print(f"[INFO] Downloaded via alternate server: {variant_url[:60]}...")
                    session.close()
                    return img
                    
                except requests.exceptions.SSLError:
                    try:
                        resp = session.get(
                            variant_url,
                            headers=headers,
                            proxies=proxies,
                            timeout=(connect_timeout, read_timeout),
                            verify=False
                        )
                        resp.raise_for_status()
                        img = Image.open(io.BytesIO(resp.content))
                        session.close()
                        return img
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
                print(f"[INFO] Round {round_num + 1} failed, waiting {delay}s before next round...")
                time.sleep(delay)
        
        session.close()
        print(f"[ERROR] Failed to download image after {total_attempts} attempts: {url[:50]}... - {last_error}")
        return None

    def _map_tokens_to_sources(self, content: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        [轻量级解析] 扫描 content，建立 Token 到 图片元数据 的映射。
        注意：此处不下载/解码图片，只保存引用。
        """
        source_map = {}
        last_token = None
        
        token_pattern = re.compile(r"<([a-zA-Z0-9_]+)>\s*$")

        for item in content:
            item_type = item.get("type")
            
            if item_type == "text":
                text = item.get("text", "")
                match = token_pattern.search(text)
                if match:
                    last_token = match.group(1)
                else:
                    if text.strip():
                        last_token = None
            
            elif item_type == "image_url":
                if last_token:
                    # 关键修改：只存储 item (元数据)，不调用 _load_image_from_source
                    source_map[last_token] = item
                    # 消费 Token
                    last_token = None

        return source_map

    def extract_sources_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        遍历对话历史，提取所有 Token 对应的图片元数据。
        """
        all_sources = {}
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                sources = self._map_tokens_to_sources(content)
                all_sources.update(sources)
        return all_sources

    def _crop_one_to_local(
        self,
        token: str,
        source_item: Dict[str, Any],
        box: Tuple[int, int, int, int],
        local_crop_dir: Path,
    ) -> Tuple[Path, str]:
        """
        同步执行单张图片的"加载 → 校验 → 裁切 → 落盘"。

        必须在线程池中调用（被 batch_crop_images 通过 asyncio.to_thread 调度），
        以避免阻塞 server 端的 asyncio 事件循环。

        :return: (本地落盘路径, 图片格式字符串)
        :raises ValueError / Exception: 各类失败
        """
        img = None
        try:
            img = self._load_image_from_source(source_item)
            if not img:
                raise ValueError(
                    f"Failed to load image <{token}>. "
                    f"The image URL may be invalid or the image has been removed."
                )

            if len(box) != 4:
                raise ValueError(
                    f"Invalid crop box format for <{token}>: {box}. "
                    f"Expected exactly 4 integers: [left, top, right, bottom]."
                )

            left, top, right, bottom = box
            width, height = img.size
            original_box = (left, top, right, bottom)

            clamped = (
                max(0, min(left, width)),
                max(0, min(top, height)),
                max(0, min(right, width)),
                max(0, min(bottom, height)),
            )
            if clamped != (left, top, right, bottom):
                print(
                    f"[WARN] Crop box for <{token}> clamped from "
                    f"{(left, top, right, bottom)} to {clamped} (size={img.size})"
                )
            left, top, right, bottom = clamped

            if left >= right or top >= bottom:
                raise ValueError(
                    f"Invalid crop box {original_box} for image <{token}>. "
                    f"Image size is {width}x{height} pixels. "
                    f"Valid coordinate ranges: "
                    f"left∈[0,{width}], top∈[0,{height}], right∈[0,{width}], bottom∈[0,{height}]. "
                    f"Note: right must be > left, and bottom must be > top."
                )

            cropped_img = img.crop((left, top, right, bottom))

            fmt = img.format if img.format else 'PNG'
            filename = f"{self.CROP_MARKER}{token}_{int(time.time())}_{uuid.uuid4().hex[:6]}.{fmt.lower()}"
            file_path = local_crop_dir / filename

            cropped_img.save(file_path, format=fmt)
            return file_path, fmt
        finally:
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass

    async def batch_crop_images(self,
                              crop_config: Dict[str, Tuple[int, int, int, int]],
                              messages: Optional[List[Dict[str, Any]]] = None,
                              content: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
        """
        执行批量裁切任务（内存优化版）。
        存储模式由配置文件的 STORAGE_MODE 决定，不再通过参数传递。
        """
        # Debug: 输出接收到的参数信息
        print(f"[DEBUG] batch_crop_images called with:")
        print(f"  - crop_config: {crop_config}")
        print(f"  - messages: {len(messages) if messages else 'None'} messages")
        print(f"  - content: {len(content) if content else 'None'} items")
        
        # Normalize crop_config tokens to accept both "<image_1>", "image_1", and "\"<image_1>\"".
        normalized_config: Dict[str, Tuple[int, int, int, int]] = {}
        for raw_token, box in crop_config.items():
            if not isinstance(raw_token, str):
                continue
            token = raw_token.strip()
            # 先去掉外层引号（如果有）
            if token.startswith('"') and token.endswith('"') and len(token) > 2:
                token = token[1:-1].strip()
            # 再去掉尖括号（如果有）
            if token.startswith("<") and token.endswith(">") and len(token) > 2:
                token = token[1:-1].strip()
            if not token:
                continue
            if token in normalized_config:
                # Keep first occurrence to avoid silent overrides.
                continue
            normalized_config[token] = box

        if not normalized_config:
            return {k: "Error: No valid tokens in crop_config" for k in crop_config.keys()}

        # 1. 建立索引 (内存占用极小)
        # 仅存储 Token 字符串和 URL 字符串
        image_sources = {}
        if content:
            sources_from_content = self._map_tokens_to_sources(content)
            image_sources.update(sources_from_content)
            print(f"[DEBUG] Extracted {len(sources_from_content)} tokens from content: {list(sources_from_content.keys())}")
        if messages:
            sources_from_messages = self.extract_sources_from_messages(messages)
            image_sources.update(sources_from_messages)
            print(f"[DEBUG] Extracted {len(sources_from_messages)} tokens from messages: {list(sources_from_messages.keys())}")
            
        print(f"[DEBUG] Total available tokens: {list(image_sources.keys())}")
            
        if not image_sources:
            print("[WARN] No images found in the provided context.")
            print(f"[DEBUG] messages count: {len(messages) if messages else 0}")
            print(f"[DEBUG] content count: {len(content) if content else 0}")
            return {k: "Error: No images found for this token" for k in crop_config.keys()}
        
        results = {}
        # 使用专用的裁切目录
        local_crop_dir = self.config.CROPS_DIR
        local_crop_dir.mkdir(parents=True, exist_ok=True)

        # 读取全局存储模式配置
        current_mode = self.config.STORAGE_MODE
        
        # 2. 串行处理每个裁切任务 (保证同一时间内存中只有一张图片)
        # 高并发下 server 是单一 asyncio 事件循环，PIL/requests 全部同步会
        # 阻塞 accept 队列导致客户端 ConnectTimeout。把同步重活全部丢线程池。
        for token, box in normalized_config.items():
            if token not in image_sources:
                print(f"[WARN] Token <{token}> not found in extracted images.")
                print(f"[DEBUG] Available tokens: {list(image_sources.keys())}")
                print(f"[DEBUG] Requested token: {repr(token)}")
                results[token] = f"Error: Image <{token}> not found"
                continue

            # === 关键：按需加载 ===
            source_item = image_sources[token]

            # 拦截逻辑：禁止二次裁切（通过文件名标记与 URL 子串同时判断）
            url_str = source_item.get("image_url", {}).get("url", "")
            if self.CROP_MARKER in url_str:
                msg = f"Error: recursive cropping is not allowed. Image <{token}> is already a cropped fragment."
                print(f"[INFO] Blocked recursive crop for <{token}>")
                results[token] = msg
                continue

            try:
                # 把"下载 + 解码 + 裁切 + 落盘"整体丢到线程池，避免阻塞事件循环。
                file_path, fmt = await asyncio.to_thread(
                    self._crop_one_to_local,
                    token,
                    source_item,
                    box,
                    local_crop_dir,
                )

                if current_mode == "local":
                    results[token] = str(file_path.absolute())

                elif current_mode == "cloud":
                    try:
                        if self.cloud_storage is None:
                            self.cloud_storage = CloudStorageService()
                        upload_res = await self.cloud_storage.upload_single_image(file_path)
                        results[token] = upload_res['url']

                        # 缓存清理控制
                        if not self.config.KEEP_LOCAL_CACHE:
                            try:
                                await asyncio.to_thread(file_path.unlink)
                            except Exception:
                                pass
                        else:
                            print(f"[DEBUG] Kept local crop cache: {file_path.name}")

                    except Exception as upload_err:
                        results[token] = f"Error upload: {upload_err}"

                else:
                    results[token] = f"Error: Unknown STORAGE_MODE: {current_mode}"

            except Exception as e:
                print(f"[ERROR] Failed to crop <{token}>: {e}")
                results[token] = f"Error: {str(e)}"
                
        return results
