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
        self.retry_delay = int(cfg.get("search_retry_delay", 60) or 60)

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


class BaiduAISearchProvider(SearchProviderBase):
    """百度千帆智能搜索适配实现。"""

    NAME = "baidu"

    def __init__(self, cfg: ConfigHelper):
        super().__init__(cfg)
        self.api_key = cfg.get("baidu_key")
        self.top_k = int(cfg.get("search_top_k", 6) or 6)
        self.url = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
        self.timeout = int(cfg.get("search_timeout", 1200) or 1200)
        if not self.api_key:
            raise SearchProviderError("缺少 Baidu AI Search 鉴权。")

    def _search(self, question: str, time_filter: str):
        time.sleep(3)
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

        response = requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        while not payload.get("choices"):
            time.sleep(self.retry_delay)
            response = requests.post(self.url, headers=headers, json=body, timeout=self.timeout)
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
    provider_name = (config.get("search_provider", "baidu") or "baidu").lower()
    if provider_name not in ("baidu", ""):
        raise SearchProviderError("当前仅支持 Baidu AI Search。")
    return BaiduAISearchProvider(config)


_PROVIDER = None


def web_search(search_message: str, search_recency_filter: str = "none"):
    """统一对外暴露的搜索调用。"""
    global _PROVIDER  # pylint: disable=global-statement
    if _PROVIDER is None:
        _PROVIDER = _build_provider()
    return _PROVIDER.search(search_message, search_recency_filter or "none")
