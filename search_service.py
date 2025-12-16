# -*- coding: utf-8 -*-
import json
import re
import time
from html import unescape
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote
from search_anylyze import (
    call_qwen_long,
    _filter_accessible_doc_urls,
    _qwen_doc_turbo_analyze,
)

import requests

from config import ConfigHelper

try:
    from bs4 import BeautifulSoup,Comment
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - optional dependency
    sync_playwright = None


config = ConfigHelper()

# 可配置的过滤关键字
_EXCLUDE_KEYWORDS = ("footer", "header", "nav", "menu", "helper","dialog","modal","csdn","repo")
_DOC_LINK_EXTS = (".pdf", ".doc", ".docx")

RECENCY_HINT = {
    "week": "最近7天",
    "month": "最近30天",
    "semiyear": "最近180天",
    "year": "最近365天",
}

_PLAYWRIGHT = None
_BROWSER = None


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



class BaiduAISearchProvider(SearchProviderBase):
    """百度搜索 /v2/ai_search/web_search 纯检索提供方。"""

    NAME = "baidu"

    def __init__(self, cfg: ConfigHelper):
        super().__init__(cfg)
        self.api_key = cfg.get("baidu_key")
        if not self.api_key:
            raise SearchProviderError("缺少百度 AI 搜索密钥。")

        # 搜索返回结果条数（仅web），默认给个 10，限制在 [1, 50]
        self.top_k = int(cfg.get("search_top_k", 10) or 10)

        # 版本：standard / lite，配置错了就不带，让服务端用默认值
        edition = (cfg.get("baidu_search_edition") or "standard").lower()
        self.edition = edition if edition in ("standard", "lite") else None

        self.url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
        self.timeout = int(cfg.get("search_timeout", 30) or 30)

    def _search(self, question: str, time_filter: str):
        # 这里不再调用 _apply_recency_hint，不往 query 里加中文提示，
        # 而是用官方的 search_recency_filter 参数控制时效。
        body: Dict = {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [
                {
                    "type": "web",
                    "top_k": 50,
                }
            ],
        }

        # 可选：standard / lite
        if self.edition:
            body["edition"] = self.edition

        # 将你现有的 time_filter 映射到官方 search_recency_filter
        if time_filter in ("week", "month", "semiyear", "year"):
            body["search_recency_filter"] = time_filter

        headers = {
            # 文档里有两种写法，这里两个头都带上，兼容性更好
            "Authorization": f"Bearer {self.api_key}",
            "X-Appbuilder-Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        # web_search 的主体结果都在 references 里
        raw_refs = payload.get("references") or []

        normalized_items: List[Dict] = []
        for ref in raw_refs:
            # 只保留我们上层真正需要的字段，其它放在原对象里也行，
            # 但为了和 DashScope provider 的结构统一，这里做一次规整。
            normalized_items.append(
                {
                    "title": ref.get("title"),
                    "snippet": ref.get("snippet") or ref.get("content"),
                    "publish_time": ref.get("date"),
                    "url": ref.get("url"),
                    "source": ref.get("website") or ref.get("web_anchor"),
                    "rerank_score": ref.get("rerank_score"),
                    "authority_score": ref.get("authority_score"),
                }
            )


        return normalized_items



def _build_provider() -> SearchProviderBase:
    provider_name = (config.get("search_provider", "baidu") or "baidu").lower()

    if provider_name == BaiduAISearchProvider.NAME:
        return BaiduAISearchProvider(config)

    raise SearchProviderError(f"未知的搜索提供方：{provider_name}")


def _is_document_url(url: str) -> bool:
    """根据后缀判断是否是 PDF/DOC/DOCX 文档链接，作为 content-type 不可用时的备用信息。"""
    if not url:
        return False
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return False
    return any(path.endswith(ext) for ext in _DOC_LINK_EXTS)


def _extract_doc_ext_from_disposition(content_disposition: str) -> str:
    """从 Content-Disposition 中提取文件扩展名（小写，含点），否则返回空字符串。"""
    disposition = (content_disposition or "").strip()
    if not disposition:
        return ""

    filename = ""
    for part in disposition.split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip().lower()
        if name not in ("filename", "filename*"):
            continue
        filename = value.strip().strip("\"'")
        if name == "filename*" and "''" in filename:
            filename = filename.split("''", 1)[1]
        break

    if not filename:
        return ""

    try:
        filename = unquote(filename)
    except Exception:
        pass

    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:].lower()


def _is_doc_content_type(content_type: str) -> bool:
    """通过 Content-Type 判断是否为文档类内容。"""
    ct = (content_type or "").lower()
    if not ct:
        return False
    if "pdf" in ct:
        return True
    return any(
        kw in ct
        for kw in (
            "msword",
            "vnd.openxmlformats-officedocument.wordprocessingml.document",
            "vnd.ms-word",
            "vnd.ms-office",
        )
    )


def _probe_content_type(url: str, timeout: int = 15) -> Tuple[bool, str, str]:
    """
    HEAD 优先尝试，若失败/405/4xx 再用 GET(stream=True) 取头，
    返回 (ok, content_type, content_disposition)，只用于决策，不读取内容。
    """
    if not url:
        return False, "", ""

    headers = {"User-Agent": "PolyMindBot/1.0"}

    def _req(method: str):
        return requests.request(
            method,
            url,
            timeout=timeout,
            allow_redirects=True,
            headers=headers,
            stream=True,
        )

    resp = None
    try:
        resp = _req("HEAD")
        code = int(resp.status_code or 0)
        content_type = resp.headers.get("content-type", "")
        content_disposition = resp.headers.get("content-disposition", "")
        resp.close()

        if code == 405 or code >= 400 or code == 0:
            resp = _req("GET")
            code = int(resp.status_code or 0)
            if not content_type:
                content_type = resp.headers.get("content-type", "")
            if not content_disposition:
                content_disposition = resp.headers.get("content-disposition", "")
            resp.close()

        ok = 200 <= code < 400
        return ok, content_type, content_disposition
    except Exception:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        return False, "", ""


def _should_use_doc_parser(url: str, content_type: str, content_disposition: str = "") -> bool:
    """
    判断是否走文档解析：
    1) Content-Type 已明确为文档，直接走文档解析；
    2) Content-Type 为 application/octet-stream 时，仅查看 Content-Disposition 的文件名后缀；
       若能识别出 PDF/DOC/DOCX，则走文档解析；
    3) Content-Type 为空时，再退回 URL 后缀兜底。
    """
    ct = (content_type or "").lower()
    if _is_doc_content_type(ct):
        return True

    if "application/octet-stream" in ct:
        disp_ext = _extract_doc_ext_from_disposition(content_disposition)
        if disp_ext in _DOC_LINK_EXTS:
            return True
        # octet-stream 未识别出文件后缀则不再视为文档
        return False

    if not ct:
        return _is_document_url(url)
    return False


def _summarize_document(search_question: str, url: str) -> str:
    """
    对文档类链接使用文档解析逻辑生成描述文本。
    失败时返回空字符串，不阻断整体搜索流程。
    """
    api_key = (config.get("qwen_key") or "").strip()
    if not api_key or not url:
        return ""

    accessible = _filter_accessible_doc_urls([url])
    if not accessible:
        return ""

    try:
        return _qwen_doc_turbo_analyze(
            api_key=api_key,
            search_question=search_question,
            urls=accessible,
            requirement="",
        )
    except Exception:
        return ""


_PROVIDER = None


def web_search(search_message: str, search_recency_filter: str = "none"):
    """统一对外暴露的搜索函数。"""
    global _PROVIDER  # pylint: disable=global-statement
    if _PROVIDER is None:
        _PROVIDER = _build_provider()

    ref_items = _PROVIDER.search(search_message, search_recency_filter or "none") or []

    def sort_key(item: Dict):
        return (
            float(item.get("authority_score") or 0),
            float(item.get("rerank_score") or 0),
        )

    sorted_refs = sorted(ref_items, key=sort_key, reverse=True)
    top_refs = sorted_refs[:10]

    fetch_timeout = int(config.get("search_fetch_timeout", 20) or 20)
    results: List[Dict] = []
    seen_urls = set()
    doc_summary_cache: Dict[str, str] = {}

    url_list = []
    for ref in top_refs:
        url = ref.get("url") or ""
        url_list.append(url)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        time.sleep(3)
        _ok_ct, content_type, content_disposition = _probe_content_type(
            url, timeout=fetch_timeout
        )
        is_doc = _should_use_doc_parser(url, content_type, content_disposition)

        web_content = ""
        if is_doc:
            if url not in doc_summary_cache:
                doc_summary_cache[url] = _summarize_document(search_message, url)
            web_content = doc_summary_cache.get(url, "")
            if not web_content:
                web_content = _fetch_url_content(url, timeout=fetch_timeout)
        else:
            web_content = _fetch_url_content(url, timeout=fetch_timeout)
        results.append(
            {
                "title": ref.get("title"),
                "snippet": ref.get("snippet"),
                "publish_time": ref.get("publish_time"),
                "url": url,
                "source": ref.get("source"),
                "authority_score": ref.get("authority_score"),
                "rerank_score": ref.get("rerank_score"),
                "web_content": web_content,
            }
        )



    # 取 title/publish_time/source/web_content/url 字段组成 JSON 字符串，供大模型整合并保留来源和时间
    result_content = json.dumps(
        [
            {
                "title": item.get("title"),
                "publish_time": item.get("publish_time"),
                "source": item.get("source"),
                "snippet": item.get("snippet"),
                "web_content": item.get("web_content"),
                "url": item.get("url"),
            }
            for item in results
        ],
        ensure_ascii=False,
    )
    
    result = ""
    try:
        result = call_qwen_long(search_message, result_content)
    except Exception as e:
        if len(result_content)>1024*32:
            print("Error;",url_list, "\nInfo:",e)
        else:
           print("Error:",result_content,"\nInfo:",e)

    return result

def _get_playwright_page(timeout: int):
    """获取一个 Playwright Page，多次调用共用浏览器实例。

    若未安装 playwright 或启动失败，则返回 None。
    """
    global _PLAYWRIGHT, _BROWSER

    if sync_playwright is None:
        # 用户没装 playwright，直接放弃走回退逻辑
        return None

    # 首次或异常后重建浏览器实例
    if _PLAYWRIGHT is None or _BROWSER is None:
        try:
            _PLAYWRIGHT = sync_playwright().start()
            # 这里用 Chromium，你也可以改成 .firefox / .webkit
            _BROWSER = _PLAYWRIGHT.chromium.launch(headless=True)
        except Exception:
            _PLAYWRIGHT = None
            _BROWSER = None
            return None

    try:
        page = _BROWSER.new_page()
        # 统一设置默认超时（Playwright 单位是 ms）
        page.set_default_timeout(timeout * 1000)
        return page
    except Exception:
        return None


def _prune_overlay_and_sensitive_blocks(soup: "BeautifulSoup") -> None:
    """仅做两件事：
    1) 删除明显的覆盖层/弹窗/横幅等（dialog/modal/popup/banner/overlay）；
    2) 按敏感词命中删除块级节点（不做任何“正文保护/主块识别”）。
    """
    if soup is None:
        return

    # ---- 1) 删除覆盖层/弹窗/横幅（低误伤）----
    OVERLAY_HINTS = (
        "modal", "dialog", "popup", "overlay", "banner", "toast", "layer",
        "mask", "backdrop", "float", "floating", "fixedbar",
    )

    def _attr_text(tag, name: str) -> str:
        v = tag.get(name)
        if not v:
            return ""
        if isinstance(v, (list, tuple)):
            return " ".join(map(str, v)).lower()
        return str(v).lower()

    def _is_overlay(tag) -> bool:
        role = _attr_text(tag, "role")
        aria_modal = _attr_text(tag, "aria-modal")
        if role in ("dialog", "alertdialog") or aria_modal == "true":
            return True

        cid = (_attr_text(tag, "id") + " " + _attr_text(tag, "class")).strip()
        return bool(cid) and any(h in cid for h in OVERLAY_HINTS)

    for bad in list(soup.find_all(_is_overlay)):
        bad.decompose()

    # ---- 2) 按敏感词块级剔除（只做敏感词，不做正文保护）----
    raw_kw = (config.get("html_sensitive_keywords") or "").strip()
    if raw_kw:
        sensitive_kws = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]
    else:
        sensitive_kws = [
            # 中文（保守）
            "博彩", "赌场", "投注", "娱乐城", "色情", "裸聊", "约炮", "招嫖", "代孕"
        ]

    if not sensitive_kws:
        return

    BLOCK_TAGS = ("div", "section", "article", "aside", "li", "p", "td")

    def _contains_sensitive(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        hits = 0
        for kw in sensitive_kws:
            hits += t.count(kw)
            if hits >= 2:
                return True
        # 只命中 1 次时，为了尽量减少误删：仅在短块上触发（广告/引流块更常见）
        return hits >= 1 and len(t) <= 600

    for blk in list(soup.find_all(BLOCK_TAGS)):
        txt = blk.get_text(" ", strip=True)
        if txt and _contains_sensitive(txt):
            blk.decompose()

_DOC_LINK_EXTS = (".pdf", ".doc", ".docx")

def _clean_body_html(html: str) -> str:
    """在 _extract_body_html 之后做进一步清洗：

    1) 删除 class / id 中包含 footer/header/nav/menu/helper 的元素；
    2) 删除 <footer>、<ins> 等元素（整个元素）；
    3) 若 <a> 标签不含“文档链接”（href 不含指定后缀），一律剔除整个 <a>；
    4) 所有包含 data / data-* 属性的元素：清空其 data 属性的值（保留属性名）；
    5) 删除所有 <img> 的 src 属性（保留 <img> 标签）；
    6) 删除 HTML 注释；
    7) 兜底：删除 body 中的 script/style 等。
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return html

    # 0) 兜底删除 body 内的明显非正文标签（按你要求扩展）
    for s in soup.find_all(
        ["script", "style", "footer", "ins", "time", "ul", "li", "form", "input", "button", "link", "head", "meta","object","svg"]
    ):
        s.decompose()

    # 0.1) 删除 HTML 注释 <!-- ... -->
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    # 3) a 标签：非文档链接一律剔除（同时兼容 attrs=None、无 href）
    def _is_doc_href(href: str) -> bool:
        h = (href or "").strip().lower()
        if not h:
            return False
        if h.startswith(("#", "javascript:", "mailto:", "tel:")):
            return False
        return any(ext in h for ext in _DOC_LINK_EXTS)

    for a in list(soup.find_all("a")):
        attrs = getattr(a, "attrs", None)
        if not isinstance(attrs, dict):
            a.decompose()
            continue

        href_val = attrs.get("href", "")
        href = href_val.strip().lower() if isinstance(href_val, str) else ""

        if (not href) or (not _is_doc_href(href)):
            a.decompose()

    # 1) 删除命中 class/id 关键字的元素（防御：attrs 可能为 None）
    def _should_remove(tag) -> bool:
        attrs = getattr(tag, "attrs", None)
        if not isinstance(attrs, dict):
            return False

        for attr_name in ("class", "id"):
            val = attrs.get(attr_name)
            if not val:
                continue

            if isinstance(val, (list, tuple)):
                text = " ".join(map(str, val))
            else:
                text = str(val)

            text = text.lower()
            if any(kw in text for kw in _EXCLUDE_KEYWORDS):
                return True

        return False

    for bad in list(soup.find_all(_should_remove)):
        bad.decompose()

    # 覆盖层/敏感块剔除逻辑保留
    _prune_overlay_and_sensitive_blocks(soup)

    # 4) 清空 data / data-* 属性值（防御：attrs 可能为 None）
    for tag in soup.find_all(True):
        attrs = getattr(tag, "attrs", None)
        if not isinstance(attrs, dict):
            continue
        for attr in list(attrs.keys()):
            al = str(attr).lower()
            if al == "data" or al.startswith("data-"):
                attrs[attr] = ""

    # 5) 删除 <img> 的 src（防御：attrs 可能为 None）
    for img in soup.find_all("img"):
        attrs = getattr(img, "attrs", None)
        if isinstance(attrs, dict):
            attrs.pop("src", None)

    if soup.body is not None:
        return str(soup.body)
    return str(soup)

def _fetch_url_content(url: str, timeout: int = 15) -> str:
    """下载网页并返回 <body> 的 HTML。

    流程保持不变：
    - 先 Playwright：page.content() -> _extract_body_html()（抽取 body + 删 script/style）
                      -> _clean_body_html()
    - 再 fallback 为 requests：字节级解码 -> _extract_body_html() -> _clean_body_html()
    """
    if not url:
        return ""

    # 1. 优先 Playwright
    page = _get_playwright_page(timeout)
    if page is not None:
        try:
            page.goto(url, wait_until="networkidle")
            html = page.content()
            page.close()
            html = (html or "").strip()
            if html:
                # 这里仍然调用你原来的逻辑：抽取 body + 删除 body 内 script/style
                body_html = _extract_body_html(html)
                # 在此基础上再加 class/id 过滤 和 img 去 src
                body_html = _clean_body_html(body_html)
                return body_html.strip()
        except Exception:
            try:
                page.close()
            except Exception:
                pass
            # 继续走 requests 兜底

    # 2. 回退方案：requests + 显式编码处理
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "PolyMindBot/1.0"},
        )
        response.raise_for_status()
    except Exception:
        return ""

    raw = response.content or b""

    # 2.1 响应头 charset
    encoding: Optional[str] = None
    content_type = response.headers.get("content-type", "").lower()
    m = re.search(r"charset=([\w\-]+)", content_type)
    if m:
        encoding = m.group(1).strip("'\" ").lower()

    # 2.2 若没写 charset，从 meta 里探测（只看前几 KB）
    if not encoding and raw:
        head = raw[:8192].decode("ascii", errors="ignore")
        m_meta = re.search(
            r"(?is)<meta[^>]+charset=['\"]?\s*([a-zA-Z0-9_\-]+)\s*['\"]?",
            head,
        )
        if m_meta:
            encoding = m_meta.group(1).strip().lower()

    # 2.3 再退一步，用 apparent_encoding
    if not encoding:
        try:
            encoding = (response.apparent_encoding or "").lower()
        except Exception:
            encoding = ""

    # 2.4 统一中文编码
    if encoding in ("gb2312", "gbk", "gb-2312", "gb_2312"):
        encoding = "gb18030"
    if not encoding:
        encoding = "utf-8"

    # 2.5 解码
    try:
        text = raw.decode(encoding, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")

    # 只对 HTML 做 body 抽取 + 清洗
    if "html" in content_type:
        # 先用原有函数抽 body + 去 script/style
        text = _extract_body_html(text)
        # 再做 class/id 过滤 和 img 去 src
        text = _clean_body_html(text)

    return text.strip()



def _extract_body_html(html: str) -> str:
    """获取<body>部分并移除脚本/样式，优先使用 BeautifulSoup。"""
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        body = soup.body
        if body:
            return str(body)
        # 无 body 时退回整个文档（已去脚本/样式）
        return str(soup)

    # 简单正则回退：截 body，去 script/style，保留剩余 HTML
    body_match = re.search(r"(?is)<body[^>]*>(.*?)</body>", html)
    body_html = body_match.group(1) if body_match else html
    body_html = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", body_html)
    body_html = re.sub(r"\s+", " ", body_html)
    return unescape(body_html).strip()
