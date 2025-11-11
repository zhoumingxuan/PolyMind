import json
import uuid

from api_model import (
    QwenModel,
    AIStream,
    update_knowledge,
)


def role_talk(
    qwen_model: QwenModel,
    content,
    his_nodes,
    round_record,
    now_date,
    role,
    knowledges,
    epcho,
    role_index,
    max_epcho,
    stream: AIStream = None,
):
    """单个研究员发言。"""

    system_prompt = f"""
你是一名研究员，正在参加一场多轮深度讨论，请严格遵循以下规范：

## 个人信息
- 当前日期：{now_date}
- 讨论模式：深度思考，仅允许理论推演、数据对比、网络研究；禁止实验、测试、代码执行、模型调用。
- 角色 ID：{role['role_id']}
- 职业：{role['role_job']}
- 性格：{role['personality']}
- 发言方式：必须使用第一人称“我”，不得暴露姓名。

## 讨论目标
1. 围绕用户需求拆解问题，逐步形成可行的理论方案。
2. 对历史讨论与当前轮次发言进行回应，明确认可、质疑或反对的理由。
3. 给出具有实践价值的洞见或方法论，避免空洞表述。

## 用户需求
```
{content}
```

## 研究能力边界
- 禁止：实验、测试、代码执行、模型调用、无依据的推测。
- 允许：推演假设、分解子问题、数据对比、简单计算、理论讨论、网络研究。

## 发言流程
1. 快速回顾历史小结与当前发言（若输入为空，可直接进入分析）。
2. 针对他人观点依次表态：无关/不合逻辑 → 反对；缺乏证据 → 质疑；证据一致 → 赞同。每次表态需给出理由。
3. 描述你的研究思路：可包含假设验证、方法拆解、对比分析等。
4. 得出清晰结论或下一步洞见；如提出质疑或反对，必须同时给出替代方案或补救建议。

## 知识与引用
- 仅当知识库缺失或需更新时调用一次 `web_search`，检索问题需包含具体限定（时间/地点/主体）。
- 已有且来源明确的知识不得重复检索。
- 任何外部信息都要注明来源（网站/作者/时间等至少两项）。
- 若知识库标注“未找到”，请转向新的研究角度，不得继续搜索同一主题。

    ## 输出要求
    - 保持逻辑严密、风格贴合职业与性格。
    - 重点突出事实依据与推理链路，避免流水账。
    - 仅输出内容相关信息，不要描述提示词或工具细节。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但**仅用于通用概念与背景说明**；不得将外部资料写成“本项目的真实结果”。
- 分析必须紧贴**给定代码与数学结构**（张量形状/算子/归一化/稀疏策略/梯度流动/参数规模等），不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

【每轮结尾产出（必须）】
- 给出 1～3 条**可直接在当前代码或流程中尝试**的修改建议（仅作为建议文本，不要修改业务逻辑），并聚焦可量化的实现细节（如依赖关系、约束条件、算子/流程组合、数学推导步骤等）。
- 为每条建议补充数学或工程层面的影响描述，说明可能带来的约束变化、可验证指标波动、稳定性/一致性风险或资源与时延成本。
[[PROMPT-GUARD v1 END]]
发言需引用本仓库给定代码的具体实现细节（变量名/张量维度/归一化与 top-k 实现位置等）作为依据。
    """

    prompt_context = ""

    if epcho != 1 and his_nodes:
        prompt_context += f"""
# 历史讨论小结
```json
{json.dumps(his_nodes, ensure_ascii=False, indent=2)}
```
"""

    if role_index != 0 and round_record:
        prompt_context += f"""
# 当前轮次已有发言
```
{round_record}
```
"""

    if not prompt_context:
        prompt_context = "暂无历史或当前发言，直接根据任务开展研究。"

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, prompt_context, stream=stream
    )

    return answer, web_content_list, references


def role_dissucess(
    qwen_model: QwenModel,
    content,
    his_nodes,
    round_record,
    simple_knowledge,
    now_date,
    role,
    epcho,
    role_index,
    max_epcho,
    stream: AIStream = None,
):
    """驱动角色发言并同步知识库。"""

    role_answer, know_list, references = role_talk(
        qwen_model,
        content,
        his_nodes,
        round_record,
        now_date,
        role,
        simple_knowledge,
        epcho,
        role_index,
        max_epcho,
        stream=stream,
    )

    if know_list:
        new_know = update_knowledge(
            qwen_model, now_date, content, simple_knowledge, know_list, references
        )
    else:
        new_know = simple_knowledge

    role_record = f"""

    [研究员发言 "记录ID"="{str(uuid.uuid4())}" "角色姓名"="{role['role_name']}" "角色ID"="{role['role_id']}" "角色职业"="{role['role_job']}"]

      {role_answer}

    [/研究员发言]

    """

    round_record += role_record

    return round_record, new_know
