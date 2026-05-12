import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Optional, Any, Mapping, Union
from urllib.parse import quote
from openai import OpenAI

from ..cache import (
    TOOL_JINA_READER,
    TOOL_SERPER_SEARCH,
    build_url_query,
    get_cache_service,
    normalize_url,
)
from .config.settings import Config

logger = logging.getLogger(__name__)


def _jina_key_hint(key: str) -> str:
    """Log whether a key is present without leaking the full secret."""
    if not key:
        return "(missing)"
    if len(key) <= 8:
        return "(set, hidden)"
    return f"...{key[-4:]}"


def _preview_text(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit} chars)"


class TextSearchService:
    """Service for handling text search (raw sources without LLM summarization)"""

    def __init__(self):
        self.config = Config()
        self._openai_client: Optional[OpenAI] = None

    @property
    def openai_client(self) -> OpenAI:
        """Lazy initialization of OpenAI client"""
        if self._openai_client is None:
            if not self.config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is required for text search")

            self._openai_client = OpenAI(
                api_key=self.config.OPENAI_API_KEY, base_url=self.config.OPENAI_BASE_URL
            )
        return self._openai_client

    async def search_with_summaries(
        self,
        query: str,
        k: int = None,
        region: Optional[str] = None,
        lang: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Perform text search and return raw sources (title + url) without Jina fetch or LLM summarization.
        """
        if k is None:
            k = self.config.DEFAULT_SEARCH_RESULTS
        if region is None:
            region = self.config.DEFAULT_REGION
        if llm_model is None:
            llm_model = self.config.DEFAULT_LLM_MODEL

        # Only require Serper.dev for raw search
        if not self.config.SERPER_API_KEY:
            raise ValueError("SERPER_API_KEY is required for text search")

        async with aiohttp.ClientSession(trust_env=True) as session:
            # Step 1: Get search results from Serper.dev
            search_results = await self._get_search_results(
                session, query, k, region, lang
            )

            # Directly return search results (title + url) without fetching page contents or LLM summarization
            return search_results

    async def visit_url(
        self, url: str, region: Optional[str] = None, lang: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Visit a specific URL via Jina Reader and return cleaned Markdown content + metadata.
        """
        if not url:
            raise ValueError("URL is required for web visiting.")
        if not self.config.JINA_API_KEY:
            raise ValueError("JINA_API_KEY is required for web visiting.")
        if region is None:
            region = self.config.DEFAULT_REGION

        logger.info(
            "jina_reader.visit_url start url=%r region=%r jina_base=%r key=%s",
            url,
            region,
            self.config.JINA_BASE,
            _jina_key_hint(self.config.JINA_API_KEY),
        )
        async with aiohttp.ClientSession(trust_env=True) as session:
            result = await self._fetch_page_markdown(session, url)
        logger.info(
            "jina_reader.visit_url end url=%r ok=%s status=%r source=%r message=%r data_len=%s",
            url,
            result.get("ok"),
            result.get("status"),
            result.get("source"),
            (result.get("message") or "")[:300],
            len(result.get("data") or []),
        )
        return result

    def _validate_api_keys(self):
        """Validate that required API keys are available"""
        required_keys = {
            "SERPER_API_KEY": self.config.SERPER_API_KEY,
            "JINA_API_KEY": self.config.JINA_API_KEY,
            "OPENAI_API_KEY": self.config.OPENAI_API_KEY,
        }

        missing = [name for name, value in required_keys.items() if not value]
        if missing:
            raise ValueError(f"Missing required API keys: {', '.join(missing)}")

    async def _get_search_results(
        self,
        session: aiohttp.ClientSession,
        query: str,
        k: int,
        region: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Get search results from Serper.dev"""
        cache = get_cache_service()
        options: Dict[str, Any] = {
            "endpoint": self.config.SERPER_SEARCH_URL,
            "region": region,
            "lang": lang,
            "__trace": {
                "raw_query": query,
            },
        }

        async def _fetch_fresh(_: str, opts: Dict[str, Any]) -> List[Dict[str, str]]:
            raw_query = str(opts.get("__trace", {}).get("raw_query", query))
            payload: Dict[str, Any] = {
                "q": raw_query,
                "num": min(k * 2, 10),  # Get more results in case some fail to fetch
            }

            if opts.get("region"):
                payload["gl"] = opts["region"]
            if opts.get("lang"):
                payload["hl"] = opts["lang"]

            headers = {
                "X-API-KEY": self.config.SERPER_API_KEY,
                "Content-Type": "application/json",
            }

            try:
                async with session.post(
                    self.config.SERPER_SEARCH_URL,
                    json=payload,
                    headers=headers,
                    timeout=self.config.REQUEST_TIMEOUT,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"[ERROR] Serper.dev HTTP {resp.status}: {error_text}")
                        return []

                    data = await resp.json()
            except Exception as e:
                print(f"[ERROR] Serper.dev request failed: {e}")
                return []

            organic_results = data.get("organic") or data.get("organic_results") or []
            logger.info(
                "[SearchV2] TextSearch raw_results=%s k=%s region=%s",
                len(organic_results),
                k,
                region,
            )
            candidates = []

            for idx, item in enumerate(organic_results, start=1):
                title = item.get("title", "")
                url = item.get("link") or item.get("url")
                snippet = item.get("snippet") or item.get("description") or ""

                if url:
                    result = {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                    candidates.append(result)

                if len(candidates) >= k:
                    break

            logger.info(
                "[SearchV2] TextSearch filtered_results=%s k=%s region=%s",
                len(candidates),
                k,
                region,
            )
            print(f"[INFO] Found {len(candidates)} search results")
            return candidates

        return await cache.aget_or_fetch(
            tool_name=TOOL_SERPER_SEARCH,
            query=query,
            options=options,
            real_fetch_func=_fetch_fresh,
        )

    async def _fetch_page_contents(
        self, session: aiohttp.ClientSession, search_results: List[Dict[str, str]]
    ) -> List[str]:
        """Fetch page content using Jina Reader"""

        async def fetch_single_page(result: Dict[str, str]) -> str:
            safe_url = quote(result["url"], safe=":/?&=#%+-~._")
            jina_url = f"{self.config.JINA_BASE}{safe_url}"
            headers = {"Authorization": f"Bearer {self.config.JINA_API_KEY}"}
            raw_u = result["url"]
            logger.info(
                "jina_reader.batch_fetch GET jina_url=%r raw_url=%r key=%s",
                jina_url,
                raw_u,
                _jina_key_hint(self.config.JINA_API_KEY),
            )

            try:
                async with session.get(
                    jina_url, headers=headers, timeout=self.config.REQUEST_TIMEOUT
                ) as resp:
                    usage = resp.headers.get("x-usage-tokens")
                    logger.info(
                        "jina_reader.batch_fetch response raw_url=%r http_status=%s "
                        "content_type=%r x_usage_tokens=%r",
                        raw_u,
                        resp.status,
                        resp.content_type,
                        usage,
                    )
                    if resp.status == 200:
                        content = await resp.text()
                        logger.info(
                            "jina_reader.batch_fetch ok raw_url=%r text_len=%s",
                            raw_u,
                            len(content),
                        )
                        return content
                    error_text = await resp.text()
                    logger.warning(
                        "jina_reader.batch_fetch http_error raw_url=%r status=%s body=%r",
                        raw_u,
                        resp.status,
                        _preview_text(error_text, 600),
                    )

            except Exception:
                logger.exception(
                    "jina_reader.batch_fetch exception raw_url=%r jina_url=%r",
                    raw_u,
                    jina_url,
                )

            return ""

        # Fetch all pages concurrently
        return await asyncio.gather(
            *[fetch_single_page(result) for result in search_results]
        )

    async def _fetch_page_markdown(
        self, session: aiohttp.ClientSession, url: str
    ) -> Dict[str, Any]:
        """Fetch a single page's cleaned markdown via Jina Reader API."""
        cache = get_cache_service()
        normalized_url = normalize_url(url)
        cache_key = build_url_query(normalized_url)
        logger.info(
            "jina_reader._fetch_page_markdown cache_key=%r normalized_url=%r raw_url=%r",
            cache_key,
            normalized_url,
            url,
        )
        options: Dict[str, Any] = {
            "endpoint": self.config.JINA_BASE,
            "accept": "application/json",
            "__trace": {
                "raw_url": url,
                "normalized_url": normalized_url,
            },
        }

        async def _fetch_fresh(_: str, opts: Dict[str, Any]) -> Dict[str, Any]:
            raw_url = str(opts.get("__trace", {}).get("raw_url", url))
            safe_url = quote(raw_url, safe=":/?&=#%+-~._")
            jina_url = f"{self.config.JINA_BASE}{safe_url}"
            headers = {
                "Authorization": f"Bearer {self.config.JINA_API_KEY}",
                "Accept": "application/json",
            }

            logger.info(
                "jina_reader.http_request method=GET jina_url=%r raw_url=%r "
                "timeout=%s accept=application/json key=%s",
                jina_url,
                raw_url,
                self.config.REQUEST_TIMEOUT,
                _jina_key_hint(self.config.JINA_API_KEY),
            )

            try:
                async with session.get(
                    jina_url, headers=headers, timeout=self.config.REQUEST_TIMEOUT
                ) as resp:
                    status_code = resp.status
                    usage = resp.headers.get("x-usage-tokens")
                    logger.info(
                        "jina_reader.http_response raw_url=%r http_status=%s "
                        "content_type=%r x_usage_tokens=%r",
                        raw_url,
                        status_code,
                        resp.content_type,
                        usage,
                    )

                    parsed_body: Optional[Union[Dict[str, Any], List[Any]]] = None
                    raw_body: str = ""
                    json_err: Optional[Exception] = None
                    try:
                        parsed_body = await resp.json(content_type=None)
                        raw_body = (
                            json.dumps(parsed_body, ensure_ascii=False)
                            if isinstance(parsed_body, (dict, list))
                            else str(parsed_body)
                        )
                    except Exception as e:
                        json_err = e
                        raw_body = await resp.text()
                        logger.info(
                            "jina_reader.body_not_json raw_url=%r err=%r "
                            "treating_as_text len=%s",
                            raw_url,
                            e,
                            len(raw_body),
                        )

                    if status_code != 200:
                        jmsg = None
                        if isinstance(parsed_body, dict):
                            jmsg = parsed_body.get("message") or parsed_body.get(
                                "readableMessage"
                            )
                        logger.warning(
                            "jina_reader.http_error raw_url=%r http_status=%s "
                            "json_parse_error=%r jina_message=%r body_preview=%r",
                            raw_url,
                            status_code,
                            repr(json_err) if json_err else None,
                            jmsg,
                            _preview_text(raw_body, 900),
                        )
                        return {
                            "status": status_code,
                            "ok": False,
                            "source": "jina_reader",
                            "url": raw_url,
                            "data": [],
                            "message": f"Jina Reader HTTP {status_code}",
                            "raw": raw_body[:500],
                        }

                    parsed = self._parse_jina_reader_payload(
                        parsed_body if parsed_body is not None else raw_body,
                        fallback_url=raw_url,
                    )
                    if parsed:
                        parsed.setdefault("code", parsed.get("status", status_code))
                        parsed["ok"] = True
                        parsed.setdefault("source", "jina_reader")
                        n_items = len(parsed.get("data") or [])
                        logger.info(
                            "jina_reader.parse_ok raw_url=%r data_items=%s "
                            "top_status=%r code=%r",
                            raw_url,
                            n_items,
                            parsed.get("status"),
                            parsed.get("code"),
                        )
                        return parsed

                    logger.warning(
                        "jina_reader.parse_fallback_plaintext raw_url=%r "
                        "had_json_object=%s body_len=%s preview=%r",
                        raw_url,
                        parsed_body is not None,
                        len(raw_body),
                        _preview_text(raw_body, 400),
                    )
                    content = raw_body.strip()
                    return {
                        "status": status_code,
                        "ok": True,
                        "source": "jina_reader",
                        "data": [
                            {
                                "url": raw_url,
                                "content": content,
                                "content_length": len(content),
                            }
                        ],
                    }

            except Exception as e:
                logger.exception(
                    "jina_reader.request_failed raw_url=%r jina_url=%r err_type=%s",
                    raw_url,
                    jina_url,
                    type(e).__name__,
                )
                return {
                    "status": "error",
                    "ok": False,
                    "url": raw_url,
                    "message": str(e),
                    "data": [],
                    "source": "jina_reader",
                    "error_type": type(e).__name__,
                }

        return await cache.aget_or_fetch(
            tool_name=TOOL_JINA_READER,
            query=build_url_query(normalized_url),
            options=options,
            real_fetch_func=_fetch_fresh,
        )

    @staticmethod
    def _parse_jina_reader_payload(
        body: Union[str, Mapping[str, Any], List[Any]], fallback_url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse Jina Reader JSON (s.jina.ai/r.jina.ai) into a normalized dict list.
        Returns:
            {
              "status": <http/status from jina>,
              "source": "jina_reader",
              "data": [
                {"title"?, "url", "description"?, "content", "content_length"?},
                ...
              ],
              "code"?, "request_id"?, "usage"?
            }
        """
        if isinstance(body, (dict, list)):
            data: Any = body
        else:
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                logger.warning(
                    "jina_reader.parse_json_decode_error fallback_url=%r err=%s preview=%r",
                    fallback_url,
                    e,
                    _preview_text(body if isinstance(body, str) else str(body), 400),
                )
                return None

        payload: Any = data.get("data", data) if isinstance(data, Mapping) else data
        entries: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            entries = [payload]
        elif isinstance(payload, list):
            entries = [p for p in payload if isinstance(p, dict)]
        else:
            logger.warning(
                "jina_reader.parse_payload_bad_shape fallback_url=%r "
                "payload_type=%s data_top_keys=%s",
                fallback_url,
                type(payload).__name__,
                list(data.keys()) if isinstance(data, Mapping) else None,
            )
            return None

        normalized: List[Dict[str, Any]] = []
        for item in entries:
            content = item.get("content") or item.get("markdown") or item.get("text")
            record = {
                "title": item.get("title") or item.get("page_title"),
                "url": item.get("url") or fallback_url,
                "description": item.get("description") or item.get("excerpt"),
                "content": content,
            }
            if isinstance(content, str):
                record["content_length"] = len(content)

            # Drop empty fields
            cleaned = {k: v for k, v in record.items() if v}
            if cleaned:
                normalized.append(cleaned)

        if not normalized:
            logger.warning(
                "jina_reader.parse_no_extractable_content fallback_url=%r "
                "entry_count=%s entry_key_samples=%s",
                fallback_url,
                len(entries),
                [sorted(e.keys()) for e in entries[:3]],
            )
            return None

        status_code = data.get("status") or data.get("code") or 200
        result: Dict[str, Any] = {
            "status": status_code,
            "code": status_code,
            "ok": True,
            "source": "jina_reader",
            "data": normalized,
        }
        if "code" in data:
            result["code"] = data["code"]
        if "request_id" in data:
            result["request_id"] = data["request_id"]
        if "usage" in data:
            result["usage"] = data["usage"]
        return result

    async def _generate_integrated_summary(
        self,
        query: str,
        search_results: List[Dict[str, str]],
        page_contents: List[str],
        model: str,
    ) -> str:
        """Generate integrated AI summary from all search results"""
        # Combine all page contents with source information
        combined_content = ""
        for i, (result, content) in enumerate(zip(search_results, page_contents), 1):
            if content.strip():  # Only include non-empty content
                combined_content += (
                    f"\n=== 来源 {i}: {result['title']} ({result['url']}) ===\n"
                )
                combined_content += content[
                    : self.config.MAX_SUMMARY_CHARS // len(search_results)
                ]
                combined_content += "\n"

        if not combined_content.strip():
            return "无法获取有效内容进行摘要。"

        return await self._llm_summarize_integrated_async(
            query=query, combined_content=combined_content, model=model
        )

    def _create_summarized_passages(
        self, search_results: List[Dict[str, str]], integrated_summary: str
    ) -> List[Dict[str, str]]:
        """Create summarized passages linked to their respective sources"""
        return [
            {
                "title": "Integrated Search Summary",
                "url": f"Based on {len(search_results)} sources",
                "summary": integrated_summary,
                "sources": [
                    {"title": r["title"], "url": r["url"]} for r in search_results
                ],
            }
        ]

    async def _llm_summarize_integrated_async(
        self,
        query: str,
        combined_content: str,
        model: str = None,
        temperature: float = None,
    ) -> str:
        """Generate integrated AI summary from combined content using OpenAI-compatible API"""
        if model is None:
            model = self.config.DEFAULT_LLM_MODEL
        if temperature is None:
            temperature = self.config.DEFAULT_TEMPERATURE

        system_prompt = (
            "你是一个严谨的学术研究助手。请根据用户查询，综合分析所有提供的网页内容，生成一个全面且结构化的中文摘要报告：\n"
            "要求：\n"
            "1) 紧扣用户查询的核心意图，提取最相关的信息\n"
            "2) 综合所有来源的观点，形成完整的知识图景\n"
            "3) 保持客观中立的学术态度\n"
            "4) 结构清晰，分点论述，每个要点简洁明了\n"
            "5) 如有具体数据、时间、研究结论等关键信息，请明确标注\n"
            "6) 如发现不同来源间存在分歧或互补信息，请指出\n"
            "7) 控制篇幅在10-15句话内，突出重点"
        )

        user_content = (
            f"【用户查询】{query}\n\n" f"【综合内容来源】\n{combined_content}"
        )

        def _sync_call():
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )

                # === DEBUG LOG START ===
                # 打印响应类型和内容预览，帮助排查 API 返回 HTML 的问题
                print(f"\n[DEBUG] LLM Response Type: {type(response)}")
                if isinstance(response, str):
                    print(
                        f"[DEBUG] Raw String Response (First 1000 chars):\n{response[:1000]}"
                    )
                    print("-" * 60)
                # === DEBUG LOG END ===

                # 兼容性处理
                if isinstance(response, str):
                    if response.strip().startswith("{"):
                        try:
                            data = json.loads(response)
                            if isinstance(data, dict) and "choices" in data:
                                return data["choices"][0]["message"]["content"].strip()
                        except:
                            pass
                    return response.strip()

                elif isinstance(response, dict):
                    if "choices" in response:
                        msg = response["choices"][0].get("message", {})
                        if isinstance(msg, dict):
                            return msg.get("content", "").strip()
                        return msg.content.strip()

                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"[ERROR] Integrated LLM summarization failed: {e}")
                import traceback

                traceback.print_exc()
                return f"（综合摘要生成失败：{e}）"

        # Run the synchronous OpenAI call in a thread pool
        return await asyncio.to_thread(_sync_call)

    async def _llm_summarize_async(
        self,
        query: str,
        page_title: str,
        page_url: str,
        page_text: str,
        model: str = None,
        temperature: float = None,
    ) -> str:
        """Generate AI summary using OpenAI-compatible API"""
        if model is None:
            model = self.config.DEFAULT_LLM_MODEL
        if temperature is None:
            temperature = self.config.DEFAULT_TEMPERATURE

        system_prompt = (
            '你是一个严谨的学术摘要助手。请根据"用户查询"对"网页正文"进行高度相关的中文摘要：\n'
            "要求：1) 紧扣查询意图；2) 客观中立；3) 结构清晰≤6句；"
            "4) 如有数据/时间/结论请明确给出；5) 若正文相关性弱，请简要说明。"
        )

        user_content = (
            f"【用户查询】{query}\n"
            f"【网页标题】{page_title}\n"
            f"【网页链接】{page_url}\n"
            f"【网页正文（截断）】\n{page_text[:self.config.MAX_SUMMARY_CHARS]}"
        )

        def _sync_call():
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )

                # === DEBUG LOG START ===
                # 打印响应类型
                # print(f"[DEBUG] Single LLM Response Type: {type(response)}")
                if isinstance(response, str):
                    print(
                        f"[DEBUG] Raw String Response (First 500 chars):\n{response[:500]}"
                    )
                # === DEBUG LOG END ===

                if isinstance(response, str):
                    if response.strip().startswith("{"):
                        try:
                            data = json.loads(response)
                            if isinstance(data, dict) and "choices" in data:
                                return data["choices"][0]["message"]["content"].strip()
                        except:
                            pass
                    return response.strip()

                elif isinstance(response, dict):
                    if "choices" in response:
                        msg = response["choices"][0].get("message", {})
                        if isinstance(msg, dict):
                            return msg.get("content", "").strip()
                        return msg.content.strip()

                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"[ERROR] LLM summarization failed: {e}")
                return f"（摘要生成失败：{e}）"

        # Run the synchronous OpenAI call in a thread pool
        return await asyncio.to_thread(_sync_call)

    async def batch_search(
        self,
        queries: List[str],
        k: int = None,
        region: Optional[str] = None,
        lang: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> List[List[Dict[str, str]]]:
        """
        Perform multiple text searches concurrently
        """
        tasks = [
            self.search_with_summaries(query, k, region, lang, llm_model)
            for query in queries
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
