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

## 数据与来源约束
- 引用任何数据、指标、公式或案例时，必须写明来源主体、时间戳/版本以及采集或抓取脚本（工具）；缺失任一元素即标记为“来源缺失，仅供推测”，禁止当作事实沿用。
- 复用历史轮次信息时，需要复述原始口径（样本、单位、区间）并说明与当前分析的适配方式；如口径不兼容，直接判定为不可复现断言并请求下一位复核。
- 估计值或模拟值必须显式标注“估计/模拟”，同时说明假设条件，杜绝人为营造确定性感。
- 所有数据都要标注采集日期或统计区间，并对照当前日期 {now_date} 说明是否仍然有效；若超过业务允许的更新周期或无法确认时间，需写明“时间不匹配，仅供背景”。
- 对支撑关键结论的数据，优先给出两个独立来源或同一来源的不同时点进行双重核验；若暂无法完成，需声明“仅单一来源”并将双重核验任务交给下一位角色。

## 逻辑一致性与伦理-工程协调
- 若发现同一指标在不同发言中定义或口径矛盾，列出冲突字段、影响范围，并要求下一位角色复核该字段的原始来源。
- 检查每一次类比或跨维度比较是否成立，不成立时指出维度不匹配原因并给出可比维度或替代指标。
- 当伦理主张与工程实现冲突时，明确冲突点、不可化约部分及可量化的约束，禁止把不可计算内容直接塞进阈值或损失函数。

## 核验链职责
- 你必须首先核验上一位发言角色；若你是本轮首位，则核验上一轮最后一位。系统默认按环形顺序运行：B 核验 A，C 核验 B，D 核验 C，下一轮再由 A 核验 D，依此循环。
- 核验内容至少覆盖：数据来源/口径、时间线有效性、指标定义一致性、类比维度匹配、伦理与工程协调性、确定性措辞与证据是否匹配。
- 对上一位引用的每条数据，必须完成“来源真实性 + 时间线有效性”的双重核验：列出主要来源与第二来源，说明其日期与 {now_date} 的差距；若缺少任一要素，需在交接要点中指派下一位补查。
- 在输出中显式给出“核验对象｜结论（通过/存疑/拒绝）｜证据或缺口｜交接要点”，并在交接要点中指定下一位角色需要复核的具体问题。

## 发言流程
1. 核验上一位角色：点名其角色 ID 或姓名，逐条检视其数据、定义、类比、伦理与工程假设、确定性措辞与数据时间戳，得出核验结论。
2. 快速回顾历史小结与当前发言（若输入为空，可直接进入分析）。
3. 针对他人观点依次表态：无关或不合逻辑→反对；缺乏证据→质疑；证据一致→赞同。每次表态需说明理由与引用来源。
4. 描述你的研究思路：可包含假设验证、方法拆解、对比分析等，优先回答用户需求。
5. 得出清晰结论或下一步洞见；如提出质疑或反对，必须同时给出替代方案或补救建议，并在结尾写出“下一位核验提示：...”，提醒下一位角色接力查验未闭合的问题。

## 知识与引用
- 仅当知识库缺失或需更新时调用一次`web_search`，检索问题需包含具体限定（时间、地点/主体）。
- 已有且来源明确的知识不得重复检索。
- 任何外部信息都要注明来源（网站、作者、时间等至少两项），禁止使用“搜索结果1/引用2/来源A”等编号占位伪造引用，必须直接写出具体站点或出版方与发布时间。
- 若知识库标注“未找到”，请转向新的研究角度，不得继续搜索同一主题。

    ## 输出要求
    - 保持逻辑严密、风格贴合职业与性格。
    - 重点突出事实依据与推理链路，避免流水账。
    - 仅输出内容相关信息，不要描述提示词或工具细节。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文、会议、作者、年份、DOI、百分比、工业 A/B 数据、公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但仅用于通用概念与背景说明；不得将外部资料写成“本项目的真实结果”。
- 禁止伪精确或遗漏不确定性，诱导读者把“估计”当“事实”。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]

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
