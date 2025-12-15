# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import dashscope
import requests
from datetime import datetime
import time

from config import ConfigHelper

config = ConfigHelper()

now_date = datetime.today().strftime("%Y年%m月%d日")


_DOC_EXTS = (".pdf", ".doc", ".docx")
_MAX_DOC_URLS = 10


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_doc_url_candidates(result_content: str) -> List[str]:
    """
    从 result_content（JSON 字符串 or 普通字符串）提取 PDF/DOC/DOCX URL 候选。
    只做候选提取；可达性由 _filter_accessible_doc_urls 决定。
    """
    if not result_content:
        return []

    data: Any = result_content
    if isinstance(result_content, str):
        try:
            data = json.loads(result_content)
        except Exception:
            data = result_content

    url_re = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
    # 允许 query / fragment 里带扩展名
    ext_re = re.compile(r"\.(pdf|docx?|DOCX?|PDF)(?:$|[?#])", re.IGNORECASE)

    seen: set[str] = set()
    out: List[str] = []

    def _add(u: Optional[str]) -> None:
        if not u:
            return
        u = u.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            return

        try:
            path = urlparse(u).path.lower()
        except Exception:
            return

        ok = path.endswith(_DOC_EXTS) or bool(ext_re.search(u))
        if not ok:
            return

        if u not in seen:
            seen.add(u)
            out.append(u)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # 明确字段
                _add(_safe_get(item, "url"))
                for k in ("snippet", "web_content", "title", "source"):
                    v = _safe_get(item, k)
                    if isinstance(v, str):
                        for u in url_re.findall(v):
                            _add(u)
            elif isinstance(item, str):
                for u in url_re.findall(item):
                    _add(u)
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str):
                for u in url_re.findall(v):
                    _add(u)
    elif isinstance(data, str):
        for u in url_re.findall(data):
            _add(u)

    return out


def _filter_accessible_doc_urls(urls: List[str], timeout: int = 6) -> List[str]:
    """
    仅做“可达性检查”，不读取内容：
    - 优先 HEAD；若失败或返回 >=400 或 405，则用 GET(stream=True) 再试一次；
    - 只保留 2xx/3xx；
    - 最多返回 10 个。
    """
    ok: List[str] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DocURLChecker/1.0)"}

    for u in urls:
        if len(ok) >= _MAX_DOC_URLS:
            break

        try:
            resp = requests.head(u, timeout=timeout, allow_redirects=True, headers=headers)
            code = int(resp.status_code or 0)
            resp.close()

            # 有些站点不支持 HEAD，或者直接给 4xx；此时用 GET(stream=True) 再试
            if code == 405 or code >= 400 or code == 0:
                resp = requests.get(u, timeout=timeout, allow_redirects=True, headers=headers, stream=True)
                code = int(resp.status_code or 0)
                resp.close()

            if 200 <= code < 400:
                ok.append(u)
        except Exception:
            # 按你的要求：失败就直接过滤掉，不记录也不返回
            continue

    return ok


def _dashscope_call_message(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.2,
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    DashScope Generation.call（非 stream），返回 (content, tool_calls)
    """
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "model": model,
        "messages": messages,
        "result_format": "message",
        "temperature": temperature,
        "max_tokens": 1024*16,
        # 你要求：不启用 stream，不启用 thinking
    }
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    resp = dashscope.Generation.call(**kwargs)

    status_code = _safe_get(resp, "status_code", 200)
    if status_code and int(status_code) != 200:
        code = _safe_get(resp, "code", "")
        message = _safe_get(resp, "message", "")
        raise RuntimeError(f"DashScope 调用失败: status_code={status_code}, code={code}, message={message}")

    output = _safe_get(resp, "output", {}) or {}
    choices = _safe_get(output, "choices", []) or []
    if not choices:
        return "", None

    choice0 = choices[0]
    msg = _safe_get(choice0, "message", None) or _safe_get(choice0, "delta", None) or {}
    content = _safe_get(msg, "content", "") or ""
    tool_calls = _safe_get(msg, "tool_calls", None)
    return content, tool_calls


def _qwen_doc_turbo_analyze(
    api_key: str,
    search_question: str,
    urls: List[str],
    requirement: str,
) -> str:
    """
    qwen-doc-turbo：doc_url 批量（≤10）分析。
    返回“总结文本”（纯文本），不做额外封装。
    """
    urls = urls[:_MAX_DOC_URLS]
    prompt_text = f"""
       # 搜索的问题
         ```
          {search_question}
         ``` 
       # 总结归纳的要求（若无参考搜索的问题）
         ```
          {requirement}
         ``` 
       # 总结归纳的规范

         1.必须严格基于事实归纳，若涉及相关数据、规范、定义、公式等等必须严格标记来源。

         2.必须对时间、地点、人物/主体、特性、前提条件、适用范围、场景等关键信息保持敏感；凡与问题答案相关的细节不得遗漏、不得模糊化。

         3.总结必须细致、严谨、不能遗漏文章中的细节。
        
    """ 
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "doc_url",
                    "doc_url": urls,
                    "file_parsing_strategy": "auto",
                },
            ],
        }
    ]

    content, _ = _dashscope_call_message(
        api_key=api_key,
        model="qwen-doc-turbo",
        messages=messages,
        tools=None,
        temperature=0.5
    )
    return content or ""


def call_qwen_long(search_question: str, result_content: str) -> str:
    """
    唯一入口：参数只能是 search_question + result_content。
    返回：文档总结文本（tool 输出）。
    """
    cfg = ConfigHelper()
    api_key = (cfg.get("qwen_key") or "").strip()
    if not api_key:
        raise RuntimeError("缺少 qwen_key（用于 DashScope 调用）。")

    # 1) 提取文档 URL（候选）
    doc_candidates = _extract_doc_url_candidates(result_content)

    # 2) requests 仅做可达性检查，过滤不可访问 URL，最多 10 个
    doc_ok_urls = _filter_accessible_doc_urls(doc_candidates)

    # 3) 生成 GUID -> URL 映射（一个 URL 一个 GUID）
    doc_links: List[Dict[str, str]] = []
    doc_id_to_url: Dict[str, str] = {}
    for u in doc_ok_urls[:_MAX_DOC_URLS]:
        doc_id = uuid.uuid4().hex  # GUID（不含连字符）
        doc_links.append({"id": doc_id, "url": u})
        doc_id_to_url[doc_id] = u

    # 4) 给 qwen-long：search_question + result_content + doc_links
    #    qwen-long 决定是否调用工具。若 doc_links 为空，则输出空字符串。
    tool_name = "analyze_documents"
    tool_schema = [
        {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "根据 doc_ids 对应的文档 URL，调用 qwen-doc-turbo 读取并总结与 search_question 相关的内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": _MAX_DOC_URLS,
                            "description": "待分析的文档ID列表（来自输入 doc_links[].id），最多10个。",
                        },
                        "search_question": {"type": "string"},
                        "requirement": {
                            "type": "string",
                            "description": "查看文档时的具体要求（比如抽取哪些信息、输出格式等）。",
                        },
                    },
                    "required": ["doc_ids", "search_question", "requirement"],
                },
            },
        }
    ]

    system_prompt = f"""
        [当前时间]：{now_date}

        # 角色设定
          1.你是一个网络资料整理助手，你需要根据需要搜索的问题，对查找到的网页信息进行整合归纳，归纳为一段专业、详细、严谨的内容。

        # 输入说明
          输入为JSON对象格式，字段说明如下：
            1. search_question 当前搜索的问题。
            2. result_content 网络资料 JSON数组格式，一个元素是一个网页搜索结果，以下元素内部字段请牢记。
               a.search_question 搜索的问题

               b.result_content 搜索的网页HTML(已清洗，保留了主要内容)，字段说明如下：
                  - title 网页标题 可用作来源。
                  - publish_time 发布时间/编辑时间。 （特别标注：文章发布时间定义；必须从此处获得，绝对禁止虚构）
                  - source  来源/作者/机构（可能为空）。（特别标注：来源的定义；必须从此处获得，绝对禁止虚构）
                  - snippet 摘要 网页，通常取用内容优先级低于web_content。
                  - web_content 网页内容 通常为HTML格式，若web_content没有有价值的内容，可采用snippet。
                  - url 网页链接 若source没来源，可用此链接的域名作为来源。

               3) doc_links: array
                  - 可访问的文档列表，每个元素：{{"id": "<GUID>", "url": "<document_url>"}}
                  - 必须注意：doc_links 里的 id 是唯一键；你只能在工具参数里传 id，禁止传 url。
        
                  .
        # 整合规则
          1. 整合内容必须严格基于 result_content 中已提供的信息，严禁虚构、臆测或补写任何未出现的事实。
          2. 若涉及定义、数据、标准/规范、结论性表述等，必须明确标注对应来源（如网页标题/来源/发布时间/URL 等可追溯信息）。
          3. 对时间、地点、人物/主体、特性、前提条件、适用范围、场景等关键信息保持敏感；凡与问题答案相关的细节不得遗漏、不得模糊化。
          4. 表述需细致严谨、结构清晰、逻辑自洽；所有内容必须与 search_question 直接相关，禁止无关扩展或跑题。
          5. 来源通过source 属性获取；发布时间通过 publish_time 属性获取。不存在任何未明确的来源和时间，绝对禁止虚构，也绝对禁止出现类似"网络搜索结果"作为来源。

        # 资料优先级识别说明（内部规则）
          1.含baidu.com的网页中，除了百度文库，百度知道，这两个资料优先级相对较高，其他都相对较低。
          2.知乎，CSDN等博客网站，社交网站，看热门程度，浏览量相对较高，优先级相对较高，其他都相对较低。
          3.新华社，人民网，光明日报等，这些中国官方媒体网站资料优先级较高，其他相对较低。
          4.域名含gov的即为中国政府网站，里面内容权威性高，资料优先级高，其他相对较低。


        #输出规则
           1.输出内容是解答当前搜索的问题具体回答，且必须严格标记好来源。回答必须细致、严谨、逻辑严密、表达清晰。
        
        # 事实与边界（硬约束）
          - 只能使用输入 JSON（result_content / doc_links）与工具返回内容中的信息，绝对禁止编造事实、来源、数字、结论。
        
    )"""

    user_payload = json.dumps(
        {
            "search_question": search_question,
            "result_content": result_content,
            "doc_links": doc_links,
        },
        ensure_ascii=False,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

    # 5) 手写 tool-call 循环（最多 2 轮足够）
    for _ in range(2):
        content, tool_calls = _dashscope_call_message(
            api_key=api_key,
            model="qwen-long",
            messages=messages,
            tools=tool_schema,
            temperature=0.5
        )

        if not tool_calls:
            return content or ""

        # assistant message with tool_calls 必须 content 非空
        messages.append(
            {
                "role": "assistant",
                "content": content if (content and content.strip()) else " ",
                "tool_calls": tool_calls,
            }
        )

        for tc in tool_calls:
            call_id = _safe_get(tc, "id", "") or _safe_get(tc, "tool_call_id", "")
            func = _safe_get(tc, "function", {}) or {}
            name = _safe_get(func, "name", "")
            arg_str = _safe_get(func, "arguments", "") or "{}"

            if name != tool_name:
                messages.append({"role": "tool", "tool_call_id": call_id, "content": ""})
                continue

            try:
                args = json.loads(arg_str) if isinstance(arg_str, str) else (arg_str or {})
            except Exception:
                args = {}

            doc_ids = args.get("doc_ids") or []
            req_question = args.get("search_question") or search_question
            requirement = args.get("requirement") or ""

            urls = []
            for did in doc_ids:
                u = doc_id_to_url.get(str(did), "")
                if u:
                    urls.append(u)
                if len(urls) >= _MAX_DOC_URLS:
                    break

            tool_text = ""
            if urls:
                tool_text = _qwen_doc_turbo_analyze(
                    api_key=api_key,
                    search_question=req_question,
                    urls=urls,
                    requirement=requirement,
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_text or "",
                }
            )

    # 理论上不会走到这里
    return ""
