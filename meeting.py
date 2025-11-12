from datetime import datetime
import json
import uuid
import re

from api_model import QwenModel, AIStream, create_webquestion_from_user
from role import role_dissucess
from config import ConfigHelper

config = ConfigHelper()

now_date = datetime.today().strftime("%Y年%m月%d日")

MAX_EPCHO = config.get("max_epcho", 5)
ROLE_COUNT = config.get("role_count", 5)


def create_roles(qwen_model: QwenModel, content, knowledges, stream=None):
    """根据需求生成研究员角色。"""

    system_prompt = f"""
你正在组织多智能体研究讨论，请按照以下规则生成 {ROLE_COUNT} 位研究员：

## 基本信息
- 当前日期：{now_date}
- 讨论模式：理论研究，禁止实验、测试、代码执行及模型调用。

## 角色构成要求
1. 角色必须能真实存在，职业要能直接服务于当前需求。
2. 整体需覆盖多个领域，至少包含两名具备宏观或跨领域视角的角色。
    3. 可参考提供的知识库补足背景，但不得照搬其中结论。
    4. 默认国别为中国，除非用户另有说明。
    5. 即使职业相同，性格与研究重心也要区分明显，确保观点互补。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]

    ## 输出格式
    - 返回 JSON 数组，长度固定为 {ROLE_COUNT}。
- 每个元素包含：
  - role_name：姓名（虚构）。
  - role_job：职业及研究专长（需具体到行业/职责）。
  - personality：性格特征与表达风格。
"""

    user_prompt = f"""
# 用户需求
```
{content}
```

# 可选参考资料
```
{knowledges}
```

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
"""

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, user_prompt, no_search=True
    )

    answer = answer.strip()
    start_index = answer.find("[")
    end_index = answer.rfind("]")

    if start_index != -1 and end_index != -1:
        answer = answer[start_index:end_index + 1]
    else:
        raise ValueError("无法从回答中提取JSON对象")

    data = json.loads(answer)

    for role in data:
        role["role_id"] = str(uuid.uuid4())

    if stream:
        description = ""
        for role in data:
            description += (
                f"角色ID:{role['role_id']}\n"
                f"角色：{role['role_name']}\n"
                f"职业：{role['role_job']}\n"
                f"性格：{role['personality']}\n\n"
            )
        stream.process_chunk(f"角色：\n{description}\n")

    return data


def rrange_knowledge(qwen_model: QwenModel, knowledges, references, user_content):
    """整理初始知识库。"""

    system_prompt = f"""
你是资料整理员，负责将最新网络结果转化为结构化知识。

## 当前日期
- {now_date}

## 用户需求
```
{user_content}
```

    ## 任务
    1. 逐条核验输入资料的来源与时间，只保留与需求密切相关、可追溯的内容。
    2. 以 Markdown 列表形式整理背景、定义、法律法规、现状数据等信息，保留关键细节。
    3. 若发现资料缺口，可在必要时继续搜索，但本阶段不得输出解决方案。
    4. 每条知识点必须标注至少两个来源要素（如站点+时间、作者+链接等），来源不明则标注“来源缺失”。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
    """

    user_prompt = f"""
# 网络搜索结果
```json
{json.dumps(knowledges, ensure_ascii=False, indent=2)}
```

# 相关引用
```json
{json.dumps(references, ensure_ascii=False, indent=2)}
```

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
"""

    answer, reasoning, web_content_list, reference_list = qwen_model.do_call(
        system_prompt, user_prompt, no_search=True, inner_search=True
    )

    return answer


def start_meeting(qwen_model: QwenModel, content, stream: AIStream = None):
    """整体会议流程入口。"""
    print("\n\n用户需求:", content, "\n\n")

    knowledges = None
    knowledges, refs = create_webquestion_from_user(qwen_model, content, knowledges, now_date)

    know_data = rrange_knowledge(qwen_model, knowledges, refs, content)

    print("\n\n基础资料:\n\n", know_data)

    roles = create_roles(qwen_model, content, know_data, stream=stream)

    epcho = 1
    his_nodes = []

    while epcho <= MAX_EPCHO:
        round_record = f"""
        [第{epcho}轮讨论开始]


        """
        for role_index, role in enumerate(roles):
            if stream:
                stream.process_chunk(
                    f"\n\n角色：{role['role_name']}\n职业：{role['role_job']}\n性格：{role['personality']}\n\n"
                )

            new_record, new_know = role_dissucess(
                qwen_model,
                content,
                his_nodes,
                round_record,
                know_data,
                now_date,
                role,
                epcho,
                role_index,
                MAX_EPCHO,
                stream=stream,
            )
            know_data = new_know
            round_record = new_record

        print(f"\n\n====第{epcho}轮讨论结束，正在总结====\n\n")

        msg_content = summary_round(qwen_model, content, now_date, round_record, epcho)

        sugg_content, can_end = summary_sugg(
            qwen_model, content, now_date, msg_content, his_nodes, epcho, MAX_EPCHO
        )

        last_content = f"""
        # 第{epcho}轮讨论总结

        ## 当前讨论概要
        ```
        {msg_content}
        ```

        ## 当前讨论进度和建议
        ```
        {sugg_content}
        ```
        """

        print("\n\n====当前讨论小结====\n", last_content)

        his_nodes.append(last_content)

        if can_end:
            print("\n====讨论中止====\n")
            break

        epcho += 1

    if stream:
        stream.process_chunk("\n[研究讨论结束]\n\n")

    print("\n====输出最终报告====\n")

    return summary(qwen_model, content, now_date, his_nodes, know_data, stream=stream)


def summary_sugg(
    qwen_model: QwenModel,
    content,
    now_date,
    round_record_message,
    his_nodes,
    epcho,
    max_epcho,
    stream: AIStream = None,
):
    """生成当前讨论进度与后续建议。"""

    system_prompt = f"""
你正在跟踪一次多轮研究讨论，需基于输入信息给出进展评估。

## 基本信息
- 当前日期：{now_date}
- 当前轮次：第{epcho}轮 / 共 {max_epcho} 轮
- 模式：理论研究，禁止实验、测试、代码执行。

## 任务
1. 将本轮讨论与历史总结进行综合比对，明确：
   - 已达成一致的结论（按重要性列出，使用有序列表）。
   - 仍存在分歧或待补充的议题（同样使用有序列表说明原因）。
2. 评估讨论是否可结束：
   - 若可结束，给出理由并设置 canEndMeeting = true。
   - 若仍需继续，提供新增且更深入的讨论建议，避免复述历史内容。
3. 输出 JSON，对象包含：
   - approvedContent（字符串，内含有序列表 Markdown）。
   - pendingContent（字符串，内含有序列表 Markdown）。
   - nextStepsContent（字符串，如需继续讨论时提供）。
   - canEndMeeting（布尔值）。

    ## 结束条件
    - 达到最大轮次视为必须收尾。
    - 仅当所有关键问题已有清晰结论且无重大分歧时才能结束。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
总结不得引入会话与已检索结果之外的新论文/新工业案例/新实验数据；仅可归纳已有讨论与通用概念。
    """

    user_prompt = f"""
# 本轮讨论概要
```
{round_record_message}
```

# 历史讨论概要
```json
{json.dumps(his_nodes, ensure_ascii=False, indent=2)}
```

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
"""

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, user_prompt, stream=stream, no_search=True
    )

    answer = re.sub(r"}\s*，\s*{", "},{", answer)

    start_index = answer.find("{")
    end_index = answer.rfind("}")

    if start_index != -1 and end_index != -1:
        answer = answer[start_index:end_index + 1]

    json_data = json.loads(answer)

    long_content = f"""
    ## 讨论已明确通过的结论

    {json_data['approvedContent']}

    ## 仍未形成结论的议题

    {json_data['pendingContent']}
    """

    cem = json_data.get("canEndMeeting", False)
    if not cem:
        long_content += f"""
        ## 下一步讨论建议

        {json_data['nextStepsContent']}
        """

    return long_content, cem


def summary_round(
    qwen_model: QwenModel,
    content,
    now_date,
    round_record,
    epcho,
    stream: AIStream = None,
):
    """单轮讨论总结。"""

    system_prompt = f"""
你是会议记录员，需根据输入还原第{epcho}轮讨论要点。

## 基本信息
- 当前日期：{now_date}
- 模式：仅记录，不新增观点。

## 输入
- 用户需求：{content}
- 本轮所有研究员的完整发言记录。

    ## 输出要求
    1. 逐个研究员记录其核心观点、引用和态度（赞同/质疑/反对）。
    2. 保持时间顺序，确保无遗漏。
    3. 仅输出与本轮讨论相关的内容。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
总结不得引入会话与已检索结果之外的新论文/新工业案例/新实验数据；仅可归纳已有讨论与通用概念。
    """

    user_prompt = f"""
# 本轮讨论原始记录
```
{round_record}
```

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
"""

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, user_prompt, stream=stream, no_search=True
    )

    return answer


def summary(
    qwen_model: QwenModel,
    dt_content,
    now_date,
    his_nodes,
    knowledge,
    stream: AIStream = None,
):
    """会议最终总结。"""

    system_prompt = f"""
会议已经结束，请输出最终总结报告。

## 基本信息
- 当前日期：{now_date}
- 用户需求：
```
{dt_content}
```

    ## 任务
    1. 综合每轮讨论概要，提炼核心结论、关键数据、分歧点。
    2. 必须调用 web_search 工具（仅一次）以补充必要的最新事实，并在总结中引用。
    3. 若出现创新概念或专有名词，需单独成段解释。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
总结不得引入会话与已检索结果之外的新论文/新工业案例/新实验数据；仅可归纳已有讨论与通用概念，并对可执行的代码改动建议做优先级排序。

    ## 输出格式
- 使用 Markdown，结构清晰。
- 必须包含：
  - 会议总体结论
  - 关键依据（可分条列出）
  - 创新概念与定义（若有）
  - 后续建议或风险提示
"""

    user_prompt = f"""
# 讨论概要（按轮次）
```json
{json.dumps(his_nodes, ensure_ascii=False, indent=2)}
```

# 已整理知识库
```
{knowledge}
```

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
"""

    answer, reasoning, web_search_list, references = qwen_model.do_call(
        system_prompt, user_prompt, stream=stream
    )

    return answer
