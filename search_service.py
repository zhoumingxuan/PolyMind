import time
from typing import Dict, List, Tuple

import requests

from config import ConfigHelper

config = ConfigHelper()

RECENCY_HINT = {
    "week": "最近7天",
    "month": "最近30天",
    "semiyear": "最近180天",
    "year": "最近365天",
}


class SearchProviderError(RuntimeError):
    """网络搜索提供方异常。"""


class SearchProviderBase:
    """所有搜索提供方的公共实现。"""

    NAME = "base"

    def __init__(self, cfg: ConfigHelper):
        self.cfg = cfg
        self._cache: Dict[str, Tuple[str, List[Dict]]] = {}
        self.cooldown = float(cfg.get("search_cooldown", 1.0) or 1.0)
        self.retry_delay = int(cfg.get("search_retry_delay", 30) or 30)

    def _cache_key(self, question: str, time_filter: str) -> str:
        return f"{question.strip()}|||{time_filter or 'none'}"

    def _from_cache(self, question: str, time_filter: str):
        key = self._cache_key(question, time_filter)
        return self._cache.get(key)

    def _store_cache(self, question: str, time_filter: str, data):
        key = self._cache_key(question, time_filter)
        self._cache[key] = data

    def search(self, question: str, time_filter: str = "none"):
        """统一的外部调用入口，带缓存与简单退避。"""
        cached = self._from_cache(question, time_filter)
        if cached:
            return cached

        try:
            result = self._search(question, time_filter)
        except SearchProviderError:
            raise
        except Exception:  # pylint: disable=broad-except
            time.sleep(self.retry_delay)
            result = self._search(question, time_filter)

        self._store_cache(question, time_filter, result)
        if self.cooldown:
            time.sleep(self.cooldown)
        return result

    def _search(self, question: str, time_filter: str):
        raise NotImplementedError

    @staticmethod
    def _apply_recency_hint(question: str, time_filter: str) -> str:
        if not time_filter or time_filter == "none":
            return question
        hint = RECENCY_HINT.get(time_filter, "")
        if not hint:
            return question
        return f"{question}（限定{hint}范围）"

    @staticmethod
    def _format_references(items: List[Dict]) -> Tuple[str, List[Dict]]:
        """将检索条目转换为文本与引用结构。"""
        snippets: List[str] = []
        references: List[Dict] = []

        for idx, item in enumerate(items, start=1):
            title = item.get("title") or item.get("source") or f"结果{idx}"
            summary = item.get("snippet") or item.get("summary") or item.get("content", "")
            publish_time = item.get("publish_time") or item.get("publish_at") or item.get("time")
            url = item.get("url") or item.get("link") or item.get("source_url")
            source = item.get("source") or item.get("site") or item.get("provider")

            detail = f"{idx}. {title}"
            if publish_time:
                detail += f"（{publish_time}）"
            if summary:
                detail += f"：{summary}"
            if url:
                detail += f" 来源：{url}"

            snippets.append(detail)
            references.append(
                {
                    "title": title,
                    "url": url,
                    "source": source,
                    "publish_time": publish_time,
                }
            )

        return "\n".join(snippets), references


class DashScopeWebSearchProvider(SearchProviderBase):
    """通义千问 Deep Search 应用接口。"""

    NAME = "dashscope"

    def __init__(self, cfg: ConfigHelper):
        super().__init__(cfg)
        self.api_key = cfg.get("dashscope_search_api_key") or cfg.get("qwen_key")
        self.app_id = cfg.get("dashscope_search_app_id")
        self.app_version = cfg.get("dashscope_search_app_version", "beta")
        self.endpoint = cfg.get(
            "dashscope_search_endpoint",
            "https://dashscope.aliyuncs.com/api/v2/apps/deep-search-agent/chat/completions",
        )
        self.timeout = int(cfg.get("search_timeout", 120) or 120)
        if not self.api_key:
            raise SearchProviderError("缺少 DashScope 搜索 API Key。")
        if not self.app_id:
            raise SearchProviderError("缺少 DashScope 搜索应用 ID。")

    def _search(self, question: str, time_filter: str):
        query = self._apply_recency_hint(question, time_filter)
        payload = {
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            },
            "parameters": {
                "agent_options": {
                    "agent_id": self.app_id,
                    "agent_version": self.app_version or "beta",
                }
            },
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.endpoint, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        output = data.get("output") or {}
        choices = output.get("choices") or []

        contents: List[str] = []
        for choice in choices:
            message = choice.get("message") or {}
            contents.extend(self._coerce_content_list(message.get("content")))
            contents.extend(self._coerce_content_list(message.get("reasoning_content")))

        normalized_items = self._normalize_app_references(
            output.get("references") or data.get("references") or []
        )
        formatted_text, references = self._format_references(normalized_items)

        base_text = "\n".join([c for c in contents if c]).strip()
        if formatted_text:
            final_text = f"{base_text}\n\n{formatted_text}" if base_text else formatted_text
        else:
            final_text = base_text or "搜索结果为空，可考虑调整关键词。"

        return final_text, references

    @staticmethod
    def _coerce_content_list(content) -> List[str]:
        if not content:
            return []
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            normalized: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    normalized.append(item.get("text") or item.get("content") or "")
                else:
                    normalized.append(str(item))
            return [c for c in normalized if c]
        return [str(content)]

    @staticmethod
    def _normalize_app_references(refs: List[Dict]) -> List[Dict]:
        if not isinstance(refs, list):
            return []

        normalized: List[Dict] = []
        for idx, ref in enumerate(refs, 1):
            normalized.append(
                {
                    "title": ref.get("title") or ref.get("name") or ref.get("source") or f"参考资料{idx}",
                    "snippet": ref.get("snippet") or ref.get("summary") or ref.get("content"),
                    "publish_time": ref.get("publish_time") or ref.get("time"),
                    "url": ref.get("url") or ref.get("link"),
                    "source": ref.get("source") or ref.get("site"),
                }
            )
        return normalized


class BaiduAISearchProvider(SearchProviderBase):
    """百度千帆智能搜索提供方，作为备用方案。"""

    NAME = "baidu"

    def __init__(self, cfg: ConfigHelper):
        super().__init__(cfg)
        self.api_key = cfg.get("baidu_key")
        self.top_k = int(cfg.get("search_top_k", 6) or 6)
        self.url = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
        if not self.api_key:
            raise SearchProviderError("缺少百度 AI 搜索密钥。")

    def _search(self, question: str, time_filter: str):
        query = self._apply_recency_hint(question, time_filter)
        body = {
            "messages": [{"role": "user", "content": query}],
            "search_source": "baidu_search_v2",
            "search_mode": "required",
            "temperature": 0.4,
            "instruction": (
                "【任务】围绕查询问题逐条归纳检索结果，不得输出建议或计划。\n"
                "【要求】\n"
                "1. 每条信息都要给出准确来源（网站名称+文章标题+URL）。\n"
                "2. 明确写出时间、地点、人物等关键要素，保持与原文一致。\n"
                "3. 如果没有检索到结果，必须返回“搜索结果为空”。\n"
                "4. 禁止补充主观推断或模型自带知识。"
            ),
            "response_format": "text",
            "enable_reasoning": False,
            "enable_corner_markers": False,
            "resource_type_filter": [
                {"type": "image", "top_k": 3},
                {"type": "web", "top_k": max(self.top_k, 6)},
            ],
            "model": "DeepSeek-R1",
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.url, headers=headers, json=body, timeout=1200)
        response.raise_for_status()
        payload = response.json()

        while not payload.get("choices"):
            time.sleep(self.retry_delay)
            response = requests.post(self.url, headers=headers, json=body, timeout=1200)
            response.raise_for_status()
            payload = response.json()

        messages = []
        references = payload.get("references", [])
        for item in payload.get("choices", []):
            content = item.get("message", {}).get("content")
            if content:
                messages.append(content)

        return "\n".join(messages), references


def _build_provider() -> SearchProviderBase:
    provider_name = (config.get("search_provider", "dashscope") or "dashscope").lower()

    if provider_name == DashScopeWebSearchProvider.NAME:
        return DashScopeWebSearchProvider(config)
    if provider_name == BaiduAISearchProvider.NAME:
        return BaiduAISearchProvider(config)

    raise SearchProviderError(f"未知的搜索提供方：{provider_name}")


_PROVIDER = None


def web_search(search_message: str, search_recency_filter: str = "none"):
    """统一对外暴露的搜索函数。"""
    global _PROVIDER  # pylint: disable=global-statement
    if _PROVIDER is None:
        _PROVIDER = _build_provider()
    return _PROVIDER.search(search_message, search_recency_filter or "none")
