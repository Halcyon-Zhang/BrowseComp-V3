import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Dict, Union, Any

from ..cache import (
    TOOL_SERPER_IMAGES,
    TOOL_SERPER_LENS,
    build_url_query,
    get_cache_service,
)
from .config.settings import Config
from .cloud_storage import CloudStorageService

logger = logging.getLogger(__name__)


class ImageSearchService:
    """Service for handling image search functionality"""

    def __init__(self):
        self.config = Config()
        self.cloud_storage = CloudStorageService()

    async def search_by_image(
        self, image_input: Union[str, Path], k: int = None
    ) -> List[Dict[str, str]]:
        """
        Search for similar images using reverse image search

        Args:
            image_input: Either a URL string or a Path to a local image file
            k: Number of results to return (default from config)

        Returns:
            List of dictionaries containing title, thumbnail, and link
        """
        if k is None:
            k = self.config.DEFAULT_IMAGE_RESULTS

        # Validate API key
        if not self.config.SERPER_API_KEY:
            raise ValueError("SERPER_API_KEY is required for image search")

        # Determine if input is URL or local file and get search URL
        image_url = await self._prepare_image_url(image_input)

        # Perform the actual search
        async with aiohttp.ClientSession(trust_env=True) as session:
            return await self._search_by_url(session, image_url, k)

    async def _prepare_image_url(self, image_input: Union[str, Path]) -> str:
        """
        Prepare image URL for search - handle URL, Base64, or local files
        """
        # =================================================================
        # 1. Handle String Input (URL or Base64)
        # =================================================================
        if isinstance(image_input, str):
            # --- Case A: Base64 String ---
            if image_input.startswith("data:image"):
                try:
                    upload_result = await self.cloud_storage.upload_data_url(
                        image_input,
                        filename_prefix="b64_search",
                    )
                    return upload_result["url"]
                except Exception as e:
                    raise ValueError(f"Base64 processing failed: {e}")

            # --- Case B: HTTP/HTTPS URL ---
            elif image_input.startswith(("http://", "https://")):
                return image_input

            # --- Case C: Local File Path (String format) ---
            else:
                image_input = Path(image_input)

        # =================================================================
        # 2. Handle Path Input (Local File)
        # =================================================================
        if isinstance(image_input, Path):
            if not image_input.exists():
                raise FileNotFoundError(f"File not found: {image_input}")

            if not self.cloud_storage.is_supported_image(image_input):
                raise ValueError(f"Unsupported image format: {image_input.suffix}")

            # 本地文件也是自动上传，转为 URL 供 Serper.dev 使用
            print(f"[INFO] Auto-uploading local file: {image_input.name}")
            upload_result = await self.cloud_storage.upload_single_image(image_input)
            return upload_result["url"]

        raise ValueError(f"Invalid image input type: {type(image_input)}")

    async def _search_by_url(
        self, session: aiohttp.ClientSession, image_url: str, k: int
    ) -> List[Dict[str, str]]:
        """
        Perform reverse image search using Serper.dev Lens endpoint (Google Lens)
        """
        cache = get_cache_service()
        options: Dict[str, Any] = {
            "endpoint": self.config.SERPER_LENS_URL,
            "fallback_endpoint": self.config.SERPER_IMAGES_URL,
            "search_mode": "image",
            "__trace": {
                "raw_image_url": image_url,
            },
        }

        async def _fetch_fresh(_: str, opts: Dict[str, Any]) -> List[Dict[str, str]]:
            raw_image_url = str(opts.get("__trace", {}).get("raw_image_url", image_url))
            payload_variants: List[Dict[str, Any]] = [
                {"imageUrl": raw_image_url},
                {"url": raw_image_url},
                {},
            ]

            results: List[Dict[str, str]] = []
            headers = {
                "X-API-KEY": self.config.SERPER_API_KEY,
                "Content-Type": "application/json",
            }

            for payload in payload_variants:
                try:
                    async with session.post(
                        self.config.SERPER_LENS_URL,
                        json=payload,
                        headers=headers,
                        timeout=self.config.REQUEST_TIMEOUT,
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            print(
                                f"[WARN] Serper.dev lens HTTP {resp.status}: {error_text}"
                            )
                            continue

                        data: Dict[str, Any] = await resp.json()
                        items = self._extract_lens_items(data)
                        logger.info(
                            "[SearchV2] ReverseImageSearch (Lens) raw_items=%s payload_keys=%s",
                            len(items),
                            list(payload.keys()),
                        )

                        for item in items:
                            entry = self._extract_search_result(item)
                            if entry and entry["link"] and entry["thumbnail"]:
                                results.append(entry)

                            if len(results) >= k:
                                break

                        if results:
                            print(
                                f"[INFO] Found {len(results)} results using Lens endpoint with payload keys {list(payload.keys())}"
                            )
                            break

                except Exception as e:
                    print(f"[WARN] Serper.dev lens request failed: {e}")
                    continue

            if not results:
                print("[INFO] Lens endpoint failed, falling back to images endpoint")
                fallback_payloads = [
                    {"q": raw_image_url, "type": "reverse_image"},
                    {"q": raw_image_url},
                ]

                for payload in fallback_payloads:
                    try:
                        async with session.post(
                            self.config.SERPER_IMAGES_URL,
                            json=payload,
                            headers=headers,
                            timeout=self.config.REQUEST_TIMEOUT,
                        ) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                print(
                                    f"[WARN] Serper.dev images HTTP {resp.status}: {error_text}"
                                )
                                continue

                            data: Dict[str, Any] = await resp.json()
                            items = self._extract_image_items(data)
                            logger.info(
                                "[SearchV2] ReverseImageSearch (fallback) raw_items=%s payload_keys=%s",
                                len(items),
                                list(payload.keys()),
                            )

                            for item in items:
                                entry = self._extract_search_result(item)
                                if entry and entry["link"] and entry["thumbnail"]:
                                    results.append(entry)

                                if len(results) >= k:
                                    break

                            if results:
                                print(
                                    f"[INFO] Found {len(results)} results using fallback endpoint"
                                )
                                break

                    except Exception as e:
                        print(f"[WARN] Serper.dev images request failed: {e}")
                        continue

            deduped = self._deduplicate_results(results)[:k]
            logger.info(
                "[SearchV2] ReverseImageSearch filtered_results=%s k=%s",
                len(deduped),
                k,
            )
            return self._project_fields(deduped)

        return await cache.aget_or_fetch(
            tool_name=TOOL_SERPER_LENS,
            query=build_url_query(image_url),
            options=options,
            real_fetch_func=_fetch_fresh,
        )

    def _extract_image_items(self, data: Dict[str, Any]) -> List[dict]:
        """Return image item list from various possible response shapes"""
        candidates = (
            data.get("images")
            or data.get("image_results")
            or data.get("inline_images")
            or data.get("inlineImages")
            or data.get("visual_matches")
            or []
        )
        return candidates if isinstance(candidates, list) else []
    
    def _extract_lens_items(self, data: Dict[str, Any]) -> List[dict]:
        """
        Extract image items from Lens API response.
        
        Lens API returns results in different formats:
        1. type="lens" with organic results containing imageUrl and thumbnailUrl (proper reverse image search)
        2. visual_matches: List of visually similar images
        3. images: Direct image results
        4. organic with image fields: Text search fallback with image information
        """
        # Check if this is a Lens reverse image search response
        search_params = data.get("searchParameters", {})
        is_lens_search = search_params.get("type") == "lens"
        
        # Priority 1: Lens reverse image search results (type="lens" with organic results)
        # This is the proper format for reverse image search
        if is_lens_search:
            organic = data.get("organic", [])
            if organic and isinstance(organic, list):
                # Filter for results that have image information
                lens_results = []
                for item in organic:
                    # Lens results should have imageUrl and/or thumbnailUrl
                    if "imageUrl" in item or "thumbnailUrl" in item:
                        lens_results.append(item)
                
                if lens_results:
                    logger.info(f"[SearchV2] Found {len(lens_results)} Lens reverse image search results")
                    return lens_results
        
        # Priority 2: Visual matches (alternative Lens format)
        visual_matches = (
            data.get("visual_matches")
            or data.get("visualMatches")
            or []
        )
        
        if visual_matches and isinstance(visual_matches, list):
            logger.info(f"[SearchV2] Found {len(visual_matches)} visual matches from Lens API")
            return visual_matches
        
        # Priority 3: Direct image results
        images = (
            data.get("images")
            or data.get("image_results")
            or data.get("imageResults")
            or []
        )
        
        if images and isinstance(images, list):
            logger.info(f"[SearchV2] Found {len(images)} image results from Lens API")
            return images
        
        # Priority 4: If Lens API returns text search results (fallback behavior)
        # This happens when Lens API doesn't support the image URL or falls back to text search
        organic = data.get("organic", [])
        knowledge_graph = data.get("knowledgeGraph", {})
        image_related = []
        
        # Extract from knowledgeGraph if it has imageUrl
        if knowledge_graph and isinstance(knowledge_graph, dict):
            if "imageUrl" in knowledge_graph:
                image_related.append({
                    "title": knowledge_graph.get("title", ""),
                    "link": knowledge_graph.get("website", ""),
                    "imageUrl": knowledge_graph.get("imageUrl", ""),
                    "thumbnailUrl": knowledge_graph.get("imageUrl", ""),
                    "snippet": knowledge_graph.get("description", ""),
                    "source": knowledge_graph.get("descriptionSource", ""),
                })
                logger.info(f"[SearchV2] Extracted image from knowledgeGraph: {knowledge_graph.get('title', 'N/A')}")
        
        # Extract from organic results that have image-related fields
        if organic and isinstance(organic, list):
            logger.info(f"[SearchV2] Lens API returned text search results, checking {len(organic)} organic results for images")
            for item in organic:
                # Check if result has image-related fields
                if any(key in item for key in ["imageUrl", "image_url", "thumbnail", "thumbnailUrl"]):
                    image_related.append(item)
        
        if image_related:
            logger.info(f"[SearchV2] Extracted {len(image_related)} image-related results from text search fallback")
            return image_related
        
        # Priority 5: Generic results field
        results = data.get("results") or data.get("matches") or []
        if results and isinstance(results, list):
            return results
        
        logger.warning(f"[SearchV2] No image results found in Lens API response. Available keys: {list(data.keys())}")
        return []

    async def search_by_query(self, query: str, k: int = None) -> List[Dict[str, str]]:
        """
        Search images by text query using Serper.dev Google Images endpoint.

        Args:
            query: Text query for image search
            k: Number of results to return

        Returns:
            List of dictionaries containing title, thumbnail, and link
        """
        if k is None:
            k = self.config.DEFAULT_IMAGE_RESULTS

        if not self.config.SERPER_API_KEY:
            raise ValueError("SERPER_API_KEY is required for image search by query")

        async with aiohttp.ClientSession(trust_env=True) as session:
            cache = get_cache_service()
            options: Dict[str, Any] = {
                "endpoint": self.config.SERPER_IMAGES_URL,
                "search_mode": "text",
                "__trace": {
                    "raw_query": query,
                },
            }

            async def _fetch_fresh(_: str, opts: Dict[str, Any]) -> List[Dict[str, str]]:
                payload = {"q": str(opts.get("__trace", {}).get("raw_query", query))}
                headers = {
                    "X-API-KEY": self.config.SERPER_API_KEY,
                    "Content-Type": "application/json",
                }
                results: List[Dict[str, str]] = []
                try:
                    async with session.post(
                        self.config.SERPER_IMAGES_URL,
                        json=payload,
                        headers=headers,
                        timeout=self.config.REQUEST_TIMEOUT,
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            print(
                                f"[WARN] Serper.dev images (text) HTTP {resp.status}: {error_text}"
                            )
                            return []
                        data = await resp.json()
                except Exception as e:
                    print(f"[WARN] Serper.dev images (text) request failed: {e}")
                    return []

                items = self._extract_image_items(data)
                logger.info(
                    "[SearchV2] ImageSearch(raw_query) raw_items=%s",
                    len(items),
                )
                for item in items:
                    entry = self._extract_search_result(item)
                    if entry and entry["link"] and entry["thumbnail"]:
                        results.append(entry)
                    if len(results) >= k:
                        break

                deduped = self._deduplicate_results(results)[:k]
                logger.info(
                    "[SearchV2] ImageSearch(raw_query) filtered_results=%s k=%s",
                    len(deduped),
                    k,
                )
                return self._project_fields(deduped)

            return await cache.aget_or_fetch(
                tool_name=TOOL_SERPER_IMAGES,
                query=query,
                options=options,
                real_fetch_func=_fetch_fresh,
            )

    def _extract_search_result(self, item: dict) -> Dict[str, str]:
        """
        Extract and normalize search result from API response.
        Prioritizes imageUrl (original image) over thumbnailUrl - use original image for better quality.
        """
        title = item.get("title") or item.get("source") or "No Title"
        
        # Priority 1: Use imageUrl (original image) - preferred for better quality
        image_url = (
            item.get("original")
            or item.get("image")
            or item.get("image_url")
            or item.get("imageUrl")
            or ""
        )
        
        # Priority 2: If no imageUrl, use thumbnailUrl as fallback
        thumbnail = (
            item.get("thumbnail")
            or item.get("thumbnail_url")
            or item.get("thumbnailUrl")
            or ""
        )
        
        # Use imageUrl if available, otherwise fallback to thumbnailUrl
        # This ensures we use the original image (better quality) when available
        # final_image_url = image_url if image_url else thumbnail
        final_image_url = thumbnail if thumbnail else image_url
        
        if image_url:
            logger.debug(f"[SearchV2] Using original imageUrl for: {title[:50]}")
        elif thumbnail:
            logger.debug(f"[SearchV2] Using thumbnailUrl (no imageUrl available) for: {title[:50]}")
        
        source = item.get("source") or item.get("domain") or ""
        link = item.get("link") or item.get("source") or item.get("image") or ""

        result = {
            "title": title,
            "thumbnail": final_image_url,  # backward compat - use original image
            "thumbnailUrl": final_image_url,  # Use original image instead of thumbnail
            "link": link,
            "url": link,  # alias for downstream WebVisit
        }
        
        # Store original imageUrl and thumbnailUrl separately for reference
        if image_url:
            result["imageUrl"] = image_url
        if thumbnail and thumbnail != final_image_url:
            result["_thumbnailUrl"] = thumbnail  # Store original thumbnail for reference
        
        if source:
            result["source"] = source
        if item.get("domain"):
            result["domain"] = item.get("domain")
        if item.get("position"):
            result["position"] = item.get("position")
        return result

    def _deduplicate_results(
        self, results: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Remove duplicate results based on link"""
        seen_links = set()
        deduplicated = []

        for result in results:
            link = result.get("link", "")
            if link and link not in seen_links:
                seen_links.add(link)
                deduplicated.append(result)

        return deduplicated

    def _is_valid_image_url(self, url: str) -> bool:
        """
        Validate if a URL is likely to be an image URL.
        
        Checks:
        1. URL is not empty
        2. URL is not truncated (doesn't end with incomplete characters)
        3. URL appears to be an image (has image extension or image-related path patterns)
        4. URL is not obviously a webpage (doesn't have .html, .htm, etc.)
        """
        if not url or not url.strip():
            return False
        
        url = url.strip()
        
        # Check if URL is truncated (ends with quote marks - clear sign of log truncation)
        # Single or double quotes at the end are almost always from log truncation
        if url.endswith("'") or url.endswith('"'):
            logger.warning(f"[SearchV2] Truncated URL detected (ends with quote): {url[:100]}")
            return False
        
        # Check if URL ends with ellipsis (another truncation indicator)
        if url.endswith("..."):
            logger.warning(f"[SearchV2] Truncated URL detected (ends with ellipsis): {url[:100]}")
            return False
        
        # Check if URL appears incomplete (ends with very short word that's not a file extension)
        # URLs should typically end with file extension, query params, or slash
        # If it ends with a 3-char word that's not a common pattern, it might be truncated
        # But be conservative - only reject if it's clearly incomplete
        url_without_query = url.split('?')[0].split('#')[0]
        if url_without_query and not url_without_query.endswith('/'):
            last_part = url_without_query.split('/')[-1]
            # Check if it ends with a very short word (2-3 chars) that's not a file extension
            if last_part and 2 <= len(last_part) <= 3:
                # Common incomplete words that suggest truncation
                incomplete_words = ['Th', 'Im', 'Ph', 'Pi', 'Me']  # Exclude 'Med' as it might be valid
                if last_part in incomplete_words:
                    logger.warning(f"[SearchV2] Possibly truncated URL (ends with short word '{last_part}'): {url[:100]}")
                    return False
        
        # Check for obvious non-image patterns
        non_image_patterns = [
            ".html", ".htm", ".php", ".aspx", ".jsp",
            "/page", "/article", "/post", "/view",
        ]
        url_lower = url.lower()
        for pattern in non_image_patterns:
            if pattern in url_lower:
                # But allow if it's part of an image path (e.g., /image/page.jpg)
                if not any(ext in url_lower for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"]):
                    logger.debug(f"[SearchV2] Non-image URL pattern detected: {url[:100]}")
                    return False
        
        # Check for image file extensions
        image_extensions = [
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico",
            ".JPG", ".JPEG", ".PNG", ".GIF", ".WEBP", ".SVG", ".BMP", ".ICO"
        ]
        if any(url.endswith(ext) for ext in image_extensions):
            return True
        
        # Check for image-related path patterns (even without extension)
        image_path_patterns = [
            "/image/", "/img/", "/images/", "/photo/", "/photos/",
            "/picture/", "/pictures/", "/thumbnail", "/thumb",
            "/static/image", "/media/", "/assets/images",
            "lw1200",  # Springer Nature image size parameter
            "thumbnail", "thumb", "preview"
        ]
        if any(pattern in url_lower for pattern in image_path_patterns):
            # Additional check: make sure it's not a webpage URL with "image" in the path
            # e.g., /image-gallery.html should be rejected
            if not any(non_img in url_lower for non_img in [".html", ".htm", "/page", "/article"]):
                return True
        
        # If URL contains query parameters that suggest it's an image
        # e.g., ?q=tbn: (Google encrypted thumbnails), ?w= (width parameter)
        if "?" in url:
            query_params = ["q=tbn", "w=", "width=", "h=", "height=", "format=", "image="]
            if any(param in url_lower for param in query_params):
                return True
        
        # If we can't determine, be conservative and accept it
        # (better to include a potentially invalid URL than to exclude a valid one)
        logger.debug(f"[SearchV2] URL validation uncertain, accepting: {url[:100]}")
        return True

    def _project_fields(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Keep only the fields needed by ImageSearch responses.
        Returns thumbnailUrl which now contains the original image URL (imageUrl) when available.
        Validates that thumbnailUrl is a valid image URL.
        """
        projected: List[Dict[str, str]] = []
        for result in results:
            # Use thumbnailUrl (which now contains original imageUrl if available)
            thumb = result.get("thumbnailUrl") or result.get("thumbnail")
            link = result.get("link")
            if not link or not thumb:
                continue
            
            # Validate that thumbnailUrl is a valid image URL
            if not self._is_valid_image_url(thumb):
                logger.warning(
                    f"[SearchV2] Skipping invalid image URL for '{result.get('title', 'Unknown')[:50]}': {thumb[:100]}"
                )
                continue
            
            projected.append(
                {
                    "title": result.get("title", ""),
                    "thumbnailUrl": thumb,  # Contains original imageUrl when available, otherwise thumbnailUrl
                    "link": link,
                }
            )
        return projected

    async def batch_search_images(
        self, image_inputs: List[Union[str, Path]], k: int = None
    ) -> List[List[Dict[str, str]]]:
        """Search multiple images concurrently"""
        if k is None:
            k = self.config.DEFAULT_IMAGE_RESULTS

        tasks = [self.search_by_image(image_input, k) for image_input in image_inputs]
        return await asyncio.gather(*tasks, return_exceptions=True)
