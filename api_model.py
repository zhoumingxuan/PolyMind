# -*- coding: utf-8 -*-
import json
import re
import time
from datetime import datetime

import dashscope
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError as RequestsConnectionError
from urllib3.exceptions import ProtocolError

from config import ConfigHelper
from search_service import web_search

config = ConfigHelper()

DASHSCOPE_API_KEY = config.get("qwen_key", None)


class AIStream:
    def __init__(self):
        self.buffer = []

    def process_chunk(self, chunk):
        print(chunk, end="", flush=True)


class QwenModel:
    def __init__(self, model_name):
        self.api_key = DASHSCOPE_API_KEY
        self.model = model_name
        self.total_tokens_count = 0
        self.tools = self._build_search_tool()

    @staticmethod
    def _build_search_tool():
        return [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": """
【工具定位】
- 根据 question_list 中的指令批量执行 1~10 条网络搜索。
- 搜索结果必须来自真实网页，并保留可追溯的来源。

【调用规范】
1. 问题需覆盖用户提供的时间、地点、人物与限制条件。
2. question_list 必须是 JSON 数组，所有标点使用英文符号。
3. 严禁生成重复或语义接近的问题，禁止含糊表述。
""",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question_list": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "GUID，必须全局唯一。"
                                    },
                                    "question": {
                                        "type": "string",
                                        "minLength": 5,
                                        "description": "明确的中文问题，需包含上下文约束，禁止模糊或重复内容。"
                                    },
                                    "time": {
                                        "type": "string",
                                        "enum": ["none", "week", "month", "semiyear", "year"],
                                        "default": "none",
                                        "description": "按网页发布时间筛选：none（不限）、week（7 天）、month（30 天）、semiyear（180 天）、year（365 天）。"
                                    }
                                },
                                "required": ["id", "question", "time"]
                            },
                            "description": """
必填。数组中每个元素描述一条搜索需求：
- id：GUID，确保唯一。
- question：5 字以上的明确中文问题，必须体现用户需求里的关键条件。
- time：时间筛选范围（none/week/month/semiyear/year）。

请确保：
1. 最多 10 个问题；
2. 问题之间完全不重复；
3. 语义清晰、可直接执行；
4. 所有标点均使用英文符号。
"""
                        }
                    },
                    "required": ["question_list"]
                }
            }
        }]

    @staticmethod
    def _parse_tool_arguments(raw_arguments: str):
        start_index = raw_arguments.find("[")
        end_index = raw_arguments.rfind("]")
        if start_index != -1 and end_index != -1:
            raw_arguments = raw_arguments[start_index:end_index + 1]
        raw_arguments = re.sub(r"}\s*，\s*{", "},{", raw_arguments)
        raw_arguments = raw_arguments.strip() or "[]"
        return json.loads(raw_arguments)

    @staticmethod
    def _safe_msg_attr(msg, attr, default=None):
        """
        安全读取 message 上的属性，兼容对象形式和 dict 形式。
        """
        if msg is None:
            return default

        # dict 形式：优先用 get
        if isinstance(msg, dict):
            return msg.get(attr, default)

        # 对象形式：用 getattr
        try:
            return getattr(msg, attr)
        except Exception:
            return default

    def do_tool_calls(self, tool_calls, messages):
        """
        处理所有 tool_calls：逐一执行并补回 tool 消息。
        注意：tool 消息的 content 必须是字符串，这里统一 json.dumps。
        """
        all_data = []
        all_refs = []

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            args_data = tool_call["function"].get("arguments", "[]")
            tool_id = tool_call["id"]

            parsed_args = self._parse_tool_arguments(args_data)
            args = {"question_list": parsed_args}

            data_content = None
            refs = []

            if func_name == "web_search":
                data_content, ref_data = search_list(**args)
                refs.extend(ref_data)

            # content 必须为字符串
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(data_content or [], ensure_ascii=False)
            })

            all_data.extend(data_content or [])
            all_refs.extend(refs or [])

        return all_data, all_refs

    def send_messages(
        self,
        messages,
        stream: AIStream = None,
        temperature: float = 0.5,
        result_format: str = "message",
        no_search: bool = False,
        inner_search: bool = False,
    ):
        """
        发送消息到 DashScope 并以流式方式解析返回结果。

        关键点：
        - 支持思考模型(enable_thinking=True, incremental_output=True)。
        - 兼容“部分 chunk 只有 usage、没有 choices”的情况。
        - 对网络中断、限流等异常做整体重试（指数回退）。
        - 对 400/InvalidParameter（如 content 缺失）不做无意义重试，直接抛出。
        """

        max_retries = 6
        base_backoff = 5

        # 简单的节流（防止瞬时 QPS 过高）
        time.sleep(3)

        tools_data = None if no_search else self.tools

        attempt = 0
        last_exception = None

        while attempt < max_retries:
            try:
                response = dashscope.Generation.call(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024 * 16,
                    thinking_budget=1024 * 32,
                    enable_thinking=True,
                    tools=tools_data,
                    # enable_search=True if inner_search else False,
                    stream=True,
                    include_usage=True,
                    incremental_output=True,
                    result_format=result_format,
                )

                reasoning_content = []
                answer_content = []
                total_tokens = 0
                tool_response = {
                    "role": "assistant",
                    "content": " ",
                    "tool_calls": []
                }
                toolcall_infos = []

                for chunk in response:
                    status_code = getattr(chunk, "status_code", None)
                    if status_code and status_code != 200:
                        err_code = getattr(chunk, "code", None)
                        err_msg = getattr(chunk, "message", None)
                        raise RuntimeError(
                            f"DashScope 流式块错误: status_code={status_code}, code={err_code}, message={err_msg}"
                        )

                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        if isinstance(usage, dict):
                            total_tokens = usage.get("total_tokens", total_tokens)
                        else:
                            total_tokens = getattr(usage, "total_tokens", total_tokens)

                    output = getattr(chunk, "output", None)
                    if output is None:
                        continue

                    if isinstance(output, dict):
                        choices = output.get("choices")
                    else:
                        choices = getattr(output, "choices", None)

                    if not choices:
                        continue

                    choice0 = choices[0]
                    msg = None
                    if isinstance(choice0, dict):
                        msg = choice0.get("message") or choice0.get("delta") or {}
                    else:
                        msg = getattr(choice0, "message", None) or getattr(choice0, "delta", None) or {}

                    msg_reasoning = self._safe_msg_attr(msg, "reasoning_content", "")
                    msg_content = self._safe_msg_attr(msg, "content", "")
                    msg_tool_calls = self._safe_msg_attr(msg, "tool_calls", None)

                    if msg_reasoning and not msg_content:
                        reasoning_content.append(msg_reasoning)

                    if msg_tool_calls:
                        for tool_call in msg_tool_calls:
                            index = tool_call.get("index", 0)
                            while len(toolcall_infos) <= index:
                                toolcall_infos.append({"id": "", "name": "", "arguments": ""})

                            if "id" in tool_call:
                                toolcall_infos[index]["id"] += tool_call.get("id", "")

                            func = tool_call.get("function") or {}
                            if "name" in func:
                                toolcall_infos[index]["name"] += func.get("name", "")
                            if "arguments" in func:
                                toolcall_infos[index]["arguments"] += func.get("arguments", "")

                    if msg_content:
                        if stream:
                            stream.process_chunk(msg_content)
                        answer_content.append(msg_content)

                self.total_tokens_count = total_tokens
                if self.total_tokens_count > 50000:
                    print("\n累计 tokens 已超过 50000，休眠 120s 再继续使用 DashScope")
                    time.sleep(120)
                    self.total_tokens_count = 0

                if toolcall_infos:
                    for t_index, tool_call in enumerate(toolcall_infos):
                        item = {
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                            },
                            "id": tool_call["id"],
                            "index": t_index,
                            "type": "function",
                        }
                        tool_response["tool_calls"].append(item)

                answer_text = "".join(answer_content)
                reasoning_text = "".join(reasoning_content)

                # 关键：当存在 tool_calls 且答案文为空时，给一个最小非空占位，避免 400
                if tool_response["tool_calls"] and not answer_text.strip():
                    answer_text = " "  # 单空格即可满足“非空字符串”要求

                tool_response["content"] = answer_text
                tool_response["reasoning_content"] = reasoning_text
                tool_response["usage"] = {"total_tokens": total_tokens}

                return answer_text, reasoning_text, tool_response

            except (ProtocolError, ChunkedEncodingError, RequestsConnectionError) as exc:
                attempt += 1
                last_exception = exc
                backoff = min(120, base_backoff * (2 ** (attempt - 1)))
                print(
                    f"DashScope streaming 响应异常终止（{exc}），将在 {backoff}s 后进行第 {attempt}/{max_retries} 次重试"
                )
                time.sleep(backoff)
                continue
            except Exception as exc:
                # 对明显的 400/InvalidParameter（content 缺失/为空）不做重试
                msg = str(exc)
                if ("status_code=400" in msg) or ("InvalidParameter" in msg) or ("content field is a required field" in msg):
                    raise

                attempt += 1
                last_exception = exc
                backoff = min(120, base_backoff * (2 ** (attempt - 1)))
                print(
                    f"DashScope 调用/解析异常（{exc}），将在 {backoff}s 后进行第 {attempt}/{max_retries} 次重试"
                )
                time.sleep(backoff)
                continue

        raise RuntimeError(f"DashScope 多次重试后仍然失败: {last_exception}")

    def do_call(
        self,
        system_prompt,
        user_prompt,
        stream: AIStream = None,
        temperature: float = 0.5,
        no_search: bool = False,
        inner_search: bool = False,
        result_format: str = "message",
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        web_content_list = []
        answer, reasoning, tool_response = self.send_messages(
            messages,
            stream,
            temperature=temperature,
            no_search=no_search,
            inner_search=inner_search,
            result_format=result_format,
        )

        references = []
        while tool_response["tool_calls"]:
            # 直接把 tool_response 作为消息加入（沿用你早期写法的习惯）
            messages.append(tool_response)

            web_search_list, web_references = self.do_tool_calls(
                tool_response["tool_calls"], messages
            )
            web_content_list.extend(web_search_list or [])
            references.extend(web_references or [])

            answer, reasoning, tool_response = self.send_messages(
                messages,
                stream,
                temperature=temperature,
                no_search=no_search,
                inner_search=inner_search,
                result_format=result_format,
            )

        return answer, reasoning, web_content_list, references



def search_list(question_list):
    seen = set()
    unique_questions = []

    for item in question_list:
        question = item.get("question")
        if not question:
            continue
        time_range = item.get("time", "none")
        key = f"Question:`{question}`====Time:`{time_range}`"
        if key not in seen:
            seen.add(key)
            unique_questions.append({"question": question, "time": time_range})

    if not unique_questions:
        return [], []

    refs = []
    results = []
    for item in unique_questions:
        question = item.get("question")
        if not question:
            continue
        time_range = item.get("time", "none")

        print("需要搜索的问题:", question, "\n")
        web_content, refs_items = web_search(question, time_range)

        results.append({
            "question": question,
            "result": web_content
        })
        refs.extend(refs_items)

    return results, refs
