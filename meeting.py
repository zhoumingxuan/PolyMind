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
    """整理初始知识库并生成统一检索策略与需求解读。"""

    structure_example = """```
{
  "knowledgeBase": "Markdown 结构化内容，突出概念、定义、法规、现状数据与案例",
  "userNeedInsights": {
    "targetDeliverable": "用户期望的交付形态，例如《xx对比报告》",
    "coreObjectives": ["目标1", "目标2"],
    "detailConstraints": ["必须回答的细节、指标口径、量化要求"],
    "givenEvidence": ["用户提供的既有知识或数据来源（若有），可作为可靠信息，必须详细、细致的描述，不得遗漏用户需求中任何细节"],
    "researchAngles": ["用户指定的参考方向/方法/范围/比较维度（若有），可作为可靠信息，必须详细、细致的描述，不得遗漏用户需求中任何细节"],
    "riskAlerts": ["潜在争议或限制"]
  },
  "searchFocusProfile": {
    "timeliness": {"level": "高", "reason": "说明原因"},
    "compliance": {"level": "中", "reason": "说明原因"},
    "experience": {"level": "低", "reason": "说明原因"},
    "innovation": {"level": "中", "reason": "说明原因"},
    "efficiency": {"level": "高", "reason": "说明原因"},
    "extraNotes": ["提醒1", "提醒2"]
  },
  "removalHints": ["需要剔除的原文句子，可为空数组"]
}
```"""

    system_prompt = f"""
你是资料整理与需求解读专家，需要把最新检索结果沉淀为可复核的知识基线，并同步输出检索策略与交付预期。

## 当前日期
- {now_date}

## 用户需求
```
{user_content}
```

## 知识基线分栏要求
- **指标与规范定义**：逐条写明指标名称、口径、计算方法、适用场景与来源时间，杜绝混用。
- **监管 / 行业约束**：法规、标准、政策条款，注明发布机构与生效时间。
- **现状数据与趋势**：仅保留帮助理解问题的数据，写清采集区间、单位、来源脚本或接口。
- **典型案例 / 风险**：概述事实、相关主体与时间，说明对本课题的启示或争议。
- **待补充信息**：列出仍未知但必须查证的字段。
- 禁止写“推荐/建议/方案/行动计划”等指令性内容，基础知识库只保留理解所需的事实与概念。

## 工作内容
1. 过滤掉所有主观推断、推荐与无法追溯的描述，仅保留能让后续模型理解问题的背景、定义与事实。
2. 细致拆解用户需求：除了交付形式、目标、细节，还需记录用户已给出的知识、引用、研究方向或假设，保证 `userNeedInsights` 足够支撑后续讨论而无需再引用原始需求。
3. 基于知识缺口给出 `searchFocusProfile`，说明检索维度的优先级，并提出额外提醒（如“必须补全日期”）。
4. 若检索结果中存在被证伪或口径冲突的内容，写入 `removalHints`，用于后续剔除。

## 输出结构
严格输出 UTF-8 JSON（不可包含反引号），参考下方示例：
{structure_example}

## 数据约束
- 严禁编造来源、作者、年份、DOI、百分比、公司名称等“看似真实”的细节，禁止伪造工具调用记录。
- 若证据不足，使用“信息不足，以下为合理推测”，并说明假设条件。
- 不得引用与任务无关的故事或推荐性结论。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文、会议、作者、年份、DOI、百分比、工业 A/B 数据、公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但仅用于通用概念与背景说明；不得将外部资料写成“本项目的真实结果”。
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
- 禁止编造具体论文、会议、作者、年份、DOI、百分比、工业A/B 数据、公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但仅用于通用概念与背景说明；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
"""

    answer, reasoning, web_content_list, reference_list = qwen_model.do_call(
        system_prompt, user_prompt, no_search=True, inner_search=True
    )

    cleaned = answer.strip()
    start_index = cleaned.find("{")
    end_index = cleaned.rfind("}")
    if start_index != -1 and end_index != -1 and end_index >= start_index:
        cleaned = cleaned[start_index:end_index + 1]

    payload = json.loads(cleaned)
    knowledge_text = payload.get("knowledgeBase", "").strip()
    user_need_profile = payload.get("userNeedInsights", {})
    search_focus_profile = payload.get("searchFocusProfile", {})
    removal_hints = payload.get("removalHints", []) or []

    if removal_hints and knowledge_text:
        knowledge_text = prune_knowledge_sections(knowledge_text, removal_hints)

    knowledge_text = sanitize_knowledge_base(knowledge_text)

    return (
        knowledge_text,
        json.dumps(search_focus_profile, ensure_ascii=False, indent=2),
        json.dumps(user_need_profile, ensure_ascii=False, indent=2),
    )

def prune_knowledge_sections(knowledge_text, removal_snippets):
    """根据提示剔除已被证伪或存疑的知识片段。"""

    if not knowledge_text or not removal_snippets:
        return knowledge_text

    updated_text = knowledge_text
    for snippet in removal_snippets:
        fragment = (snippet or "").strip()
        if not fragment:
            continue
        pattern = re.escape(fragment)
        updated_text, _ = re.subn(pattern, "", updated_text, count=1)

    return updated_text


def sanitize_knowledge_base(knowledge_text):
    """剔除推荐、方案类内容，仅保留理解所需信息。"""

    if not knowledge_text:
        return knowledge_text

    drop_keywords = ("推荐", "建议", "方案", "行动", "路径", "推广", "部署", "购买", "投资")
    lines = []
    for line in knowledge_text.splitlines():
        raw_line = line.strip()
        if not raw_line:
            continue
        if any(keyword in raw_line for keyword in drop_keywords):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_removal_hints_from_answer(role_answer):
    """从研究员发言中提取需剔除的知识片段。"""

    if not role_answer:
        return []

    hints = []
    pattern = re.compile(r"\[需剔除知识](.*?)\[/需剔除知识]", re.S)
    for block in pattern.findall(role_answer):
        for line in block.splitlines():
            cleaned = line.strip().lstrip("-•").strip()
            if not cleaned:
                continue
            if "原文" in cleaned:
                cleaned = cleaned.split("原文：", 1)[-1].strip()
            if cleaned:
                hints.append(cleaned)
    return hints

def start_meeting(qwen_model: QwenModel, content, stream: AIStream = None):
    """整体会议流程入口。"""
    print("\n\n用户需求:", content, "\n\n")

    knowledges = None
    knowledges, refs = create_webquestion_from_user(
        qwen_model, content, knowledges, now_date
    )

    know_data, search_focus, user_need_profile = rrange_knowledge(
        qwen_model, knowledges, refs, content
    )

    print("\n\n基础资料:\n\n", know_data)
    print("\n\n重点关注要素:\n\n", search_focus)
    print("\n\n用户需求解读:\n\n", user_need_profile)

    roles = create_roles(qwen_model, content, know_data, stream=stream)

    epcho = 1
    his_nodes = []
    pending_removals = []

    while epcho <= MAX_EPCHO:
        round_record = f"""
        [第{epcho}轮讨论开始]


        """
        for role_index, role in enumerate(roles):
            if stream:
                stream.process_chunk(
                    f"\n\n角色：{role['role_name']}\n职业：{role['role_job']}\n性格：{role['personality']}\n\n"
                )

            new_record, role_answer = role_dissucess(
                qwen_model,
                content,
                his_nodes,
                round_record,
                know_data,
                search_focus,
                user_need_profile,
                now_date,
                role,
                epcho,
                role_index,
                MAX_EPCHO,
                stream=stream,
            )
            round_record = new_record
            removal_hints = extract_removal_hints_from_answer(role_answer)
            if removal_hints:
                pending_removals.extend(removal_hints)

        print(f"\n\n====第{epcho}轮讨论结束，正在总结====\n\n")

        msg_content = summary_round(
            qwen_model, user_need_profile, now_date, round_record, epcho, search_focus
        )

        sugg_text, can_end, sections_to_prune = summary_sugg(
            qwen_model,
            content,
            now_date,
            msg_content,
            his_nodes,
            know_data,
            epcho,
            MAX_EPCHO,
            search_focus,
            user_need_profile,
        )

        last_content = f"""
        # 第{epcho}轮讨论总结

        ## 当前讨论概要
        ```
        {msg_content}
        ```

        ## 当前讨论进度和建议
        ```
        {sugg_text}
        ```
        """

        print("\n\n====当前讨论小结====\n", last_content)

        his_nodes.append(last_content)

        prune_candidates = []
        if pending_removals:
            prune_candidates.extend(pending_removals)
        if sections_to_prune:
            prune_candidates.extend(sections_to_prune)
        if prune_candidates:
            unique_prunes = []
            seen = set()
            for snippet in prune_candidates:
                cleaned = (snippet or "").strip()
                if not cleaned or cleaned in seen:
                    continue
                seen.add(cleaned)
                unique_prunes.append(cleaned)
            if unique_prunes:
                know_data = prune_knowledge_sections(know_data, unique_prunes)
                know_data = sanitize_knowledge_base(know_data)
                print("\n====知识库修剪====\n")
                for idx, snippet in enumerate(unique_prunes, start=1):
                    print(f"- 移除片段{idx}: {snippet[:80]}")
        pending_removals = []

        if can_end:
            print("\n====讨论中止====\n")
            break

        epcho += 1

    if stream:
        stream.process_chunk("\n[研究讨论结束]\n\n")

    print("\n====输出最终报告====\n")

    return summary(
        qwen_model,
        content,
        now_date,
        his_nodes,
        know_data,
        search_focus,
        user_need_profile,
        stream=stream,
    )


def summary_sugg(
    qwen_model: QwenModel,
    content,
    now_date,
    round_record_message,
    his_nodes,
    knowledge_snapshot,
    epcho,
    max_epcho,
    search_focus,
    user_need_profile,
    stream: AIStream = None,
):
    """生成当前讨论进度、核验结果及下一步建议。"""

    verification_schema = """```
{
  "approvedContent": "有序列表 Markdown",
  "pendingContent": "有序列表 Markdown",
  "nextStepsContent": "如需继续讨论时的建议",
  "canEndMeeting": false,
  "verificationChecklist": [
    {
      "topicGroup": "approved|pending",
      "topicTitle": "议题标题",
      "question": "具体检索问题",
      "time": "week|month|semiyear|year|none",
      "status": "verified|disputed|pending",
      "evidenceSummary": "依据摘要，说明结论是否成立",
      "sourceCitations": [
        {
          "title": "文章或网页标题",
          "author": "作者或机构",
          "publishedAt": "时间",
          "url": "链接"
        }
      ],
      "pruneSnippet": "若需从知识库移除的原文句子，没有则留空"
    }
  ],
  "sectionsToPrune": ["知识库中需删除的片段，可为空"]
}
```"""

    system_prompt = f"""
你正在跟踪一次多轮研究讨论，需要基于输入信息给出进展评估并完成证据核验。

## 基本信息
- 当前日期：{now_date}
- 当前轮次：第{epcho}轮 / 共 {max_epcho} 轮
- 模式：理论研究，禁止实验、测试、代码执行。
- 检索关注要素：所有判断需符合下方配置，必要时提醒研究员遵守对应的时效性、规范性、经验性、创新性、效率性等级。

```
{search_focus}
```

## 核验与搜索要求
1. 先根据讨论内容判定“已通过结论”与“仍待结论”的议题。
2. 针对两类议题分别生成 1~4 条高质量检索问题（question_list），合并为一次 `web_search` 调用；禁止重复或含糊问题。
3. 核验结果需说明：
   - 检索问题
   - 证据结论（通过 / 存疑 / 待定）
   - 主要来源（标题、作者或机构、时间、链接）
   - 是否需要从知识库剔除对应片段
4. 若无法获取有效结果，需在 checklist 中写明“来源不足”和下一步计划。

## 输出 JSON 结构（仅输出 JSON）：
{verification_schema}

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文、会议、作者、年份、DOI、百分比、工业 A/B 数据、公司名称等“看似真实”的细节。
- 若证据不足，必须写明“信息不足，以下为合理推测”并阐明前提。
- 检索与引用内容均需可追溯，禁止使用“搜索结果/引用2/来源A”等占位符。
[[PROMPT-GUARD v1 END]]
总结不得引入会话与已检索结果之外的新结论，仅可提出需要补充的核验动作。
    """

    user_prompt = f"""
# 用户需求解读
```
{user_need_profile}
```

# 本轮讨论概要
```
{round_record_message}
```

# 历史讨论概要
```json
{json.dumps(his_nodes, ensure_ascii=False, indent=2)}
```

# 当前知识库摘要
```
{knowledge_snapshot}
```

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】与上方一致。
【信息不足时的处理】与上方一致。
[[PROMPT-GUARD v1 END]]
    """

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, user_prompt, stream=stream, no_search=False
    )

    normalized = re.sub(r"}\s*，\s*{", "},{", answer)
    start_index = normalized.find("{")
    end_index = normalized.rfind("}")
    if start_index != -1 and end_index != -1 and end_index >= start_index:
        normalized = normalized[start_index:end_index + 1]

    try:
        json_data = json.loads(normalized)
    except json.JSONDecodeError:
        return answer.strip(), False, []

    verification_checklist = json_data.get("verificationChecklist", []) or []
    sections_to_prune = json_data.get("sectionsToPrune", []) or []

    approved_section = json_data.get("approvedContent", "- （无）")
    pending_section = json_data.get("pendingContent", "- （无）")
    next_steps = json_data.get("nextStepsContent", "- （未提供）")

    verified_report = format_verification_section(verification_checklist, "approved")
    pending_report = format_verification_section(verification_checklist, "pending")

    long_content = f"""
    ## 讨论已明确通过的结论

    {approved_section}

    ### 证据核验（已通过议题）
    {verified_report}

    ## 仍未形成结论的议题

    {pending_section}

    ### 证据核验（待定议题）
    {pending_report}
    """.strip()

    if not json_data.get("canEndMeeting", False):
        long_content += f"""

    ## 下一步讨论建议
    {next_steps}
    """

    return long_content, json_data.get("canEndMeeting", False), sections_to_prune

def summary_round(
    qwen_model: QwenModel,
    user_need_profile,
    now_date,
    round_record,
    epcho,
    search_focus,
    stream: AIStream = None,
):
    """单轮讨论总结。"""

    system_prompt = f"""
你是会议记录员，需根据输入还原第{epcho}轮讨论要点。

## 基本信息
- 当前日期：{now_date}
- 模式：仅记录，不新增观点。
- 检索关注要素：请对照下方配置，保持时效性/规范性/经验性/创新性/效率性的约束一致。

```
{search_focus}
```

## 输入
- 用户需求解读：{user_need_profile}
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
    search_focus,
    user_need_profile,
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
- 检索关注要素：所有最终结论都要说明如何满足下方约束，必要时提示残留风险。
```
{search_focus}
```
 - 用户需求解读：请确保交付形式、目标、细节关注点与下方解析保持一致。
```
{user_need_profile}
```

    ## 任务
    1. 综合每轮讨论概要，提炼核心结论、关键数据、分歧点。
    2. 必须调用 web_search 工具（仅一次）以补充必要的最新事实，并在总结中引用。
    3. 若出现创新概念或专有名词，需单独成段解释。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文、会议、作者、年份、DOI、百分比、工业 A/B 数据、公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但仅用于通用概念与背景说明；不得将外部资料写成“本项目的真实结果”。
- 不得引用与任务无关的行业案例。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。

[[PROMPT-GUARD v1 END]]
总结不得纳入会议之外的新观点、新行业案例或未实证的数据；可以提醒读者哪些结论基于讨论或通用共识，并注明待验证项。

    ## 输出格式
- 使用 Markdown，结构清晰。
- 必须包含：
  - 用户需求解读
  - 会议总体结论
  - 关键依据（可分条列出）
  - 创新概念与定义（若有）
  - 仍存争议点
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
 
# 用户需求解读
```
{user_need_profile}
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
