# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pathlib import Path

import pdfplumber
import dashscope
import requests
import shutil
import subprocess
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
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
            content_type = (resp.headers.get("content-type", "") or "").lower()
            resp.close()

            # 有些站点不支持 HEAD，或者直接给 4xx；此时用 GET(stream=True) 再试
            if code == 405 or code >= 400 or code == 0:
                resp = requests.get(u, timeout=timeout, allow_redirects=True, headers=headers, stream=True)
                code = int(resp.status_code or 0)
                if not content_type:
                    content_type = (resp.headers.get("content-type", "") or "").lower()
                resp.close()

            if 200 <= code < 400:
                # 避免把“伪装成 .pdf 但实际返回 HTML/文本”的链接放进 doc_url
                if content_type and ("text/html" in content_type or "text/plain" in content_type):
                    continue
                ok.append(u)
        except Exception:
            # 按你的要求：失败就直接过滤掉，不记录也不返回
            continue

    return ok


def _ensure_download_dir() -> Path:
    """确保 download 目录存在，返回其 Path。"""
    d = Path("download")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_filename_from_url(url: str, fallback_ext: str = ".pdf") -> str:
    """从 URL 生成一个尽量可读且安全的文件名。"""
    try:
        p = urlparse(url).path
    except Exception:
        p = ""

    name = (p.split("/")[-1] if p else "") or (uuid.uuid4().hex + fallback_ext)
    name = name.split("?")[0].split("#")[0]

    if not name.lower().endswith((".pdf", ".doc", ".docx")):
        name = name + fallback_ext

    # 清理非法字符
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
    if not name:
        name = uuid.uuid4().hex + fallback_ext
    return name


def _download_doc(url: str, timeout: int = 30) -> Path:
    """
    下载文档到 download 目录，返回本地文件路径。
    说明：此处不再做可达性校验（上游已校验），只负责“下载落盘”。
    """
    download_dir = _ensure_download_dir()
    fname = _safe_filename_from_url(url)
    # 避免同名覆盖：若已存在则追加 uuid
    dst = download_dir / fname
    if dst.exists():
        dst = download_dir / f"{dst.stem}_{uuid.uuid4().hex}{dst.suffix}"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; DocDownloader/1.0)"}
    with requests.get(url, timeout=timeout, allow_redirects=True, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    return dst


def _read_pdf_by_page(local_path: Path) -> str:
    """
    使用 pdfplumber 逐页读取 PDF 文本，并生成“逐页结构化 JSON 字符串”。

    注意：
    - 不处理图片（按你的要求忽略图片类内容）。
    - 每页保留：全文 text、lines（逐行）、tables（若可抽取）、简单 meta。
    - 返回的是 JSON 字符串（ensure_ascii=False），便于直接喂给 qwen-long 总结。
    """
    data: Dict[str, Any] = {
        "file_type": "pdf",
        "local_path": str(local_path),
        "page_count": 0,
        "pages": [],
    }

    with pdfplumber.open(str(local_path)) as pdf:
        total = len(pdf.pages)
        data["page_count"] = total

        for i, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            # 表格（若抽取失败则置空）
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            raw_lines = [ln.rstrip("\n") for ln in (page_text.splitlines() if page_text else [])]
            lines_clean = [ln.strip() for ln in raw_lines if ln.strip()]

            heading_guesses: List[str] = []
            for ln in lines_clean[:12]:
                if re.match(r"^\d+(\.\d+)*\s+\S+", ln):
                    heading_guesses.append(ln)
                elif len(ln) <= 120 and ln.isupper() and any(ch.isalpha() for ch in ln):
                    heading_guesses.append(ln)

            data["pages"].append(
                {
                    "page": i,
                    "text": page_text,
                    "lines": lines_clean,
                    "tables": tables,
                    "meta": {
                        "line_count": len(lines_clean),
                        "char_count": len(page_text),
                        "heading_guesses": heading_guesses[:5],
                    },
                }
            )

    return json.dumps(data, ensure_ascii=False)



def _iter_docx_blocks(doc: Document):
    """
    以“保持原始顺序”的方式遍历 docx 中的段落与表格。
    返回的元素类型为 Paragraph 或 Table。
    """
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, doc)
        elif child.tag.endswith('}tbl'):
            yield Table(child, doc)


def _read_docx_by_block(docx_path: Path) -> str:
    """
    读取 .docx，生成结构化 JSON 字符串（按块：heading/paragraph/table）。
    """
    doc = Document(str(docx_path))
    blocks: List[Dict[str, Any]] = []
    para_idx = 0
    table_idx = 0

    for blk in _iter_docx_blocks(doc):
        if isinstance(blk, Paragraph):
            text = (blk.text or "").strip()
            if not text:
                continue
            para_idx += 1
            style_name = ""
            try:
                style_name = (blk.style.name or "")
            except Exception:
                style_name = ""

            is_heading = False
            level = None
            m = re.search(r"(\d+)", style_name or "")
            if style_name.lower().startswith("heading") and m:
                is_heading = True
                level = int(m.group(1))
            elif ("标题" in style_name) and m:
                is_heading = True
                level = int(m.group(1))

            blocks.append(
                {
                    "type": "heading" if is_heading else "paragraph",
                    "index": para_idx,
                    "style": style_name,
                    "heading_level": level,
                    "text": text,
                }
            )

        elif isinstance(blk, Table):
            table_idx += 1
            rows: List[List[str]] = []
            try:
                for row in blk.rows:
                    rows.append([(cell.text or "").strip() for cell in row.cells])
            except Exception:
                rows = []

            blocks.append({"type": "table", "index": table_idx, "rows": rows})

    data = {
        "file_type": "docx",
        "local_path": str(docx_path),
        "block_count": len(blocks),
        "blocks": blocks,
    }
    return json.dumps(data, ensure_ascii=False)


def _read_doc_as_text(doc_path: Path) -> str:
    """
    读取 .doc（Word 97-2003）。优先使用 antiword / catdoc；
    若都不可用则尝试 LibreOffice(soffice/libreoffice) 转 docx 再提取文本。
    """
    antiword = shutil.which("antiword")
    catdoc = shutil.which("catdoc")
    soffice = shutil.which("soffice") or shutil.which("libreoffice")

    if antiword:
        p = subprocess.run([antiword, str(doc_path)], capture_output=True, text=True)
        return (p.stdout or "")
    if catdoc:
        p = subprocess.run([catdoc, str(doc_path)], capture_output=True, text=True)
        return (p.stdout or "")

    if soffice:
        out_dir = _ensure_download_dir()
        subprocess.run(
            [soffice, "--headless", "--convert-to", "docx", "--outdir", str(out_dir), str(doc_path)],
            capture_output=True,
            text=True,
        )
        cand = out_dir / (doc_path.stem + ".docx")
        if cand.exists():
            data = json.loads(_read_docx_by_block(cand))
            texts: List[str] = []
            for b in data.get("blocks", []):
                t = (b.get("text") or "").strip()
                if t:
                    texts.append(t)
                if b.get("type") == "table":
                    for row in (b.get("rows") or []):
                        row = row or []
                        texts.append("\t".join([(c or "").strip() for c in row]))
            return "\n".join(texts)

    raise RuntimeError("无法解析 .doc：未找到 antiword/catdoc/soffice。请安装其中之一，或改为 .docx。")


def _read_doc_by_block(doc_path: Path) -> str:
    """
    读取 .doc，生成结构化 JSON 字符串：
    - 若可转换为 docx，则按 docx 结构化；
    - 否则按“段落块”结构化（纯文本解析）。
    """
    out_dir = _ensure_download_dir()
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "docx", "--outdir", str(out_dir), str(doc_path)],
            capture_output=True,
            text=True,
        )
        cand = out_dir / (doc_path.stem + ".docx")
        if cand.exists():
            data = json.loads(_read_docx_by_block(cand))
            data["file_type"] = "doc"
            data["converted_to_docx"] = str(cand)
            data["source_doc"] = str(doc_path)
            return json.dumps(data, ensure_ascii=False)

    text = _read_doc_as_text(doc_path)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    blocks = [{"type": "paragraph", "index": i + 1, "text": p} for i, p in enumerate(paras)]
    data = {
        "file_type": "doc",
        "local_path": str(doc_path),
        "block_count": len(blocks),
        "blocks": blocks,
    }
    return json.dumps(data, ensure_ascii=False)


def _read_doc_or_docx_by_block(path: Path) -> str:
    ext = (path.suffix or "").lower()
    if ext == ".docx":
        return _read_docx_by_block(path)
    if ext == ".doc":
        return _read_doc_by_block(path)
    raise RuntimeError(f"不支持的文档类型: {ext}")

def _qwen_long_summarize_doc_from_json(
    api_key: str,
    search_question: str,
    requirement: str,
    doc_url: str,
    local_path: Path,
    doc_json: str,
    doc_type: str,
) -> str:
    """
    用 qwen-long 对“本地抽取的结构化 JSON（由代码生成）”做细致总结。
    注意：这里不让模型生成结构化数据；模型只负责基于 JSON 内容做总结说明。
    """
    prompt = f"""# 任务
你将得到一份文档的结构化 JSON（由本地解析器生成：PDF=pdfplumber逐页抽取；DOCX=python-docx按块抽取；DOC=本地解析/转换后抽取）。
请你严格基于 JSON 内容，细致总结该文档“具体讲了什么”，不得遗漏关键细节。

# 文档信息
- doc_url: {doc_url}
- local_path: {str(local_path)}
- doc_type: {doc_type}

# 关联问题（用于理解侧重点，但不得为迎合问题而遗漏文档内容）
- search_question:
{search_question}

# 总结要求
{requirement}

# 强制输出规范（必须遵守）
1) 必须覆盖文档的：背景/目的、核心概念定义、整体结构、关键流程/接口/参数、约束与注意事项、示例或结论（若文档包含）。
2) 必须“可追溯”：每条关键结论都要标注来源位置：
   - PDF：标注 Page N（对应 JSON.pages[].page）
   - DOC/DOCX：标注 Block #K（对应 JSON.blocks[].index，表格也算 Block）
3) 禁止编造：JSON 里没有的内容不要补；遇到缺失就明确说明“未在文档中找到”。
4) 需要非常细致：优先按章节/主题展开，逐点罗列；尽量把每章关键点都覆盖。

# 输入（结构化 JSON）
```json
{doc_json}
```
"""
    messages = [
        {"role": "system", "content": "你是一个严谨的技术文档分析助手。"},
        {"role": "user", "content": prompt},
    ]

    content, _ = _dashscope_call_message(
        api_key=api_key,
        model="qwen-long",
        messages=messages,
        tools=None,
        temperature=0.2,
    )
    return content or ""

def _qwen_doc_turbo_analyze_doc_url_only(
    api_key: str,
    search_question: str,
    urls: List[str],
    requirement: str,
) -> str:
    """仅执行 qwen-doc-turbo 的 doc_url 分析；失败则抛出异常（由上层做 400 回退）。"""
    urls = urls[:_MAX_DOC_URLS]
    prompt_text = f"""# 搜索的问题
{search_question}

# 总结归纳的要求（若无参考搜索的问题）
{requirement}

# 总结归纳的规范
1) 必须严格基于事实归纳，若涉及相关数据、规范、定义、公式等等必须严格标记来源。
2) 必须对时间、地点、人物/主体、特性、前提条件、适用范围、场景等关键信息保持敏感；凡与问题答案相关的细节不得遗漏、不得模糊化。
3) 总结必须细致、严谨、不能遗漏文章中的细节。
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
        temperature=0.5,
    )
    return content or ""

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

    严格按你的要求：
    - 先尝试调用 qwen-doc-turbo（doc_url）解析文档；
    - 若出现 400（InvalidParameter / format not supported 等），则对触发问题的文档：
        1) 下载到 download 目录；
        2) 本地解析并生成结构化 JSON：
           - PDF：pdfplumber 逐页抽取 text/lines/tables；
           - DOC/DOCX：本地读取按块抽取 paragraph/heading/table；
        3) 把上述 JSON 交给 qwen-long 做“细致、不遗漏细节”的总结（并按 Page/Block 标注来源）。
    """
    urls = urls[:_MAX_DOC_URLS]
    if not urls:
        return ""

    # 1) 先尝试批量 doc_url
    try:
        return _qwen_doc_turbo_analyze_doc_url_only(
            api_key=api_key,
            search_question=search_question,
            urls=urls,
            requirement=requirement,
        )
    except RuntimeError as e:
        msg = str(e)

        # 只对 400 做回退；其他错误继续抛出
        if "status_code=400" not in msg:
            raise

        # 2) 逐个 URL 分析：能用 doc-turbo 的继续用；不支持的走“下载+pdfplumber+qwen-long”回退
        out_parts: List[str] = []
        for u in urls:
            try:
                one = _qwen_doc_turbo_analyze_doc_url_only(
                    api_key=api_key,
                    search_question=search_question,
                    urls=[u],
                    requirement=requirement,
                )
                out_parts.append(f"【文档】{u}\n{one}")
                continue
            except RuntimeError as e2:
                msg2 = str(e2)
                if "status_code=400" not in msg2:
                    raise

            # 400 回退路径：下载 -> 本地解析(生成结构化JSON) -> qwen-long 细致总结
            local_path = _download_doc(u)
            ext = (local_path.suffix or "").lower()

            if ext == ".pdf":
                doc_json = _read_pdf_by_page(local_path)  # 返回 JSON 字符串（逐页）
                doc_type = "pdf"
            elif ext in (".doc", ".docx"):
                doc_json = _read_doc_or_docx_by_block(local_path)  # 返回 JSON 字符串（按块）
                doc_type = ext.lstrip(".")
            else:
                # 兜底：按纯文本封装为 JSON
                try:
                    raw = local_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    raw = ""
                doc_json = json.dumps(
                    {
                        "file_type": ext.lstrip(".") or "unknown",
                        "local_path": str(local_path),
                        "text": raw,
                    },
                    ensure_ascii=False,
                )
                doc_type = ext.lstrip(".") or "unknown"

            analyzed = _qwen_long_summarize_doc_from_json(
                api_key=api_key,
                search_question=search_question,
                requirement=requirement,
                doc_url=u,
                local_path=local_path,
                doc_json=doc_json,
                doc_type=doc_type,
            )
            out_parts.append(f"【文档】{u}\n{analyzed}")

        return "\n\n".join(out_parts).strip()

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
                try:
                    tool_text = _qwen_doc_turbo_analyze(
                        api_key=api_key,
                        search_question=req_question,
                        urls=urls,
                        requirement=requirement,
                    )
                except Exception:
                    # doc_url 解析失败不应阻断整体流程
                    tool_text = ""

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_text or "",
                }
            )

    # 理论上不会走到这里
    return ""
