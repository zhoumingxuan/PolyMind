import json
import re
import time
from datetime import datetime

import dashscope
import requests
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
        try:
            return getattr(msg, attr)
        except Exception:
            return default

    def do_tool_calls(self, tool_calls, messages):
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

            messages.append(
                {"role": "tool", "tool_call_id": tool_id, "content": data_content}
            )

            return data_content, refs

    def send_messages(
        self,
        messages,
        stream: AIStream = None,
        temperature: float = 0.5,
        result_format: str = "message",
        no_search: bool = False,
        inner_search: bool = False,
    ):
        max_stream_retries = 3
        attempt = 0

        time.sleep(5)

        tools_data=None if no_search else self.tools

        while True:
            response = None
            while response is None:
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
                        # enable_search=True if inner_search else False,  # 开启联网搜索的参数
                        # search_options={
                        #     "forced_search": True,  # 强制开启联网搜索
                        #     "enable_source": True,  # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                        #     "enable_citation": False,  # 开启角标标注功能
                        #     "search_strategy": "pro"  # 模型将搜索10条互联网信息
                        # } if inner_search else None,
                        stream=True,
                        include_usage=True,
                        incremental_output=True,
                        result_format=result_format,
                    )
                except Exception:
                    response = None
                    time.sleep(60)
                    continue

            reasoning_content = []
            answer_content = []
            total_tokens = 0
            tool_response = {
                "role": "assistant",
                "content": "",
                "tool_calls": []
            }

            toolcall_infos = []

            try:
                for chunk in response:
                    msg = chunk.output.choices[0].message

                    if chunk.get("usage"):
                        total_tokens = chunk.usage.total_tokens

                    msg_reasoning = self._safe_msg_attr(msg, "reasoning_content", "")
                    msg_content = self._safe_msg_attr(msg, "content", "")
                    msg_tool_calls = self._safe_msg_attr(msg, "tool_calls", None)

                    if msg_reasoning and not msg_content:
                        reasoning_content.append(msg_reasoning)

                    if msg_tool_calls:
                        for tool_call in msg_tool_calls:
                            index = tool_call["index"]

                            while len(toolcall_infos) <= index:
                                toolcall_infos.append({"id": "", "name": "", "arguments": ""})

                            if "id" in tool_call:
                                toolcall_infos[index]["id"] += tool_call.get("id", "")

                            if "function" in tool_call:
                                func = tool_call["function"]
                                if "name" in func:
                                    toolcall_infos[index]["name"] += func.get("name", "")
                                if "arguments" in func:
                                    toolcall_infos[index]["arguments"] += func.get("arguments", "")

                    if msg_content:
                        chunk_content = msg_content
                        if stream:
                            stream.process_chunk(chunk_content)
                        answer_content.append(chunk_content)
            except Exception:
                attempt += 1
                if attempt >= max_stream_retries:
                    raise RuntimeError("DashScope streaming响应多次异常终止，请稍后重试。")
                time.sleep(60)
                continue

            self.total_tokens_count += total_tokens
            if self.total_tokens_count > 50000:
                time.sleep(60)
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
            tool_response["content"] = answer_text
            tool_response["reasoning_content"] = reasoning_text
            tool_response["usage"] = {"total_tokens": total_tokens}

            return answer_text, reasoning_text, tool_response

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


def create_webquestion_from_user(
    qwen_model: QwenModel, user_message, history_search, now_date, search_focus=None
):
    system_prompt = f"""
你是搜索任务规划师，负责为多智能体系统生成高质量的网络检索问题。

## 基本信息
- 当前日期：{now_date}

## 任务目标
1. 逐条解析用户需求，提炼必须查证的事实、定义、规范、数据等要点。
2. 结合输入中出现的时间、地点、人物、事件与特别强调内容，生成可直接执行的搜索问题。
3. 若提供历史搜索结果，需基于其中的事实识别信息缺口，避免重复查询，强化深入性问题。

    ## 编写规则
    - 问题需按“一个问题解决一个信息缺口”的原则组织，禁止冗余。
    - 优先生成：行业规范/定义 → 关键背景/数据 → 深度洞察/争议点。
    - 问题必须用中文，语义明确，避免泛化或含糊词。
    - 若用户需求中包含时间约束，需与当前日期核对后再写入问题。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
优先检索通用定义/经典做法；若来源存疑或无法核验，请在回答中注明不确定性。

    ## 输出格式
    - 严格输出原始 JSON 文本，不得附加 Markdown 代码块符号、反引号、注释、自然语言前后缀或任何非 JSON 内容。
    - 返回 JSON 数组，每个元素包含 id（GUID）、question、time（none/week/month/semiyear/year）。
    - 若暂无法生成搜索问题，直接输出空数组 `[]`，同样不得添加解释。
"""

    if history_search:
        system_prompt += f"""
## 历史搜索数据
以下为已获取的搜索结果，可直接引用，不要重复生成同义问题：
```json
{json.dumps(history_search, ensure_ascii=False, indent=2)}
```
"""

    if search_focus:
        system_prompt += f"""
## 搜索关注要素
```
{search_focus}
```
- 所有检索约束都要结合该配置给出的时效性、规范性、经验性、创新性与效率要求，禁止遗漏或混淆。
"""

    user_prompt = user_message

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, user_prompt, temperature=0.5, no_search=True
    )

    answer = answer.strip()
    start_index = answer.find("[")
    end_index = answer.rfind("]")

    if start_index != -1 and end_index != -1:
        answer = answer[start_index:end_index + 1]
    else:
        raise ValueError("无法从回答中提取JSON对象")

    data = json.loads(answer)

    refs = []
    results = []
    for item in data:
        question = item["question"]
        time_range = item["time"]

        print("需要搜索的问题:", question, "\n\n")
        web_content, ref_items = web_search(question, time_range)

        refs.extend(ref_items)

        results.append({
            "question": question,
            "result": web_content
        })

    return results, refs


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

        print("需要搜索的问题:", question, "\n\n")
        web_content, refs_items = web_search(question, time_range)

        results.append({
            "question": question,
            "result": web_content
        })
        refs.extend(refs_items)

    return results, refs


