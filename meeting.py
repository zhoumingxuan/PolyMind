from datetime import datetime
import json
import time
import uuid
import re

from api_model import QwenModel, AIStream, create_webquestion_from_user
from role import role_dissucess
from config import ConfigHelper

config = ConfigHelper()

now_date = datetime.today().strftime("%Y年%m月%d日")

MAX_EPCHO = config.get("max_epcho", 5)
ROLE_COUNT = config.get("role_count", 5)


def create_roles(qwen_model: QwenModel, content,user_need_profile, knowledges, stream=None):
    """根据需求生成研究员角色。"""

    system_prompt = f"""
你正在组织多智能体研究讨论，请按照以下规则生成 {ROLE_COUNT} 位研究员：

## 基本信息
- 当前日期：{now_date}
- 讨论模式：理论研究，禁止实验、测试、代码执行及模型调用。

## 角色构成要求
1. 角色必须能真实存在，职业要能直接服务于当前需求。
2. 角色职业必须严格根据研究范围来确定；角色职业可重复，但是若角色职业重复，则性格必须不重复。
3. 整体需覆盖多个领域，至少包含两名具备宏观或跨领域视角的角色。
4. 可参考提供的知识库补足背景，但不得照搬其中结论。
5. 默认国别为中国，除非用户另有说明。
6. 即使职业相同，性格与研究重心也要区分明显，确保观点互补。

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

#用户需求解读
```
{user_need_profile}
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
  "知识整理": "Markdown 结构化内容，突出概念、定义、法规、现状数据与案例",
  "userNeedInsights": {
    "期望交付形态": "用户期望的交付形态，例如《xx对比报告》",
    "研究核心目标": ["目标1", "目标2"],
    "研究范围":"根据用户需求明确严格限定讨论的范围",
    "必须回答的细节": ["必须回答的细节、指标口径、量化要求"],
    "用户提供的既有知识或数据来源": ["用户提供的既有知识或数据来源（若有），可作为可靠信息，必须详细、细致的描述，不得遗漏用户需求中任何细节"],
    "用户提供的参考": ["用户指定的参考方向/方法/范围/比较维度（若有），可作为可靠信息，必须详细、细致的描述，不得遗漏用户需求中任何细节"],
    "风险提示": ["潜在争议或限制"]
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
    knowledge_text = payload.get("知识整理", "").strip()
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


def start_meeting(qwen_model: QwenModel, content, stream: AIStream = None):
    """整体会议流程入口。"""
    print("\n\n用户需求:", content, "\n\n")

    knowledges = None
    knowledges, refs = create_webquestion_from_user(
        qwen_model, content, knowledges, now_date
    )
    
    while True:
     try:
      know_data, search_focus, user_need_profile = rrange_knowledge(
        qwen_model, knowledges, refs, content
      )
      break
     except Exception:
      print("\n出现返回错误，资料重新整理\n")
      time.sleep(60)
      continue
      

    print("\n\n基础资料:\n\n", know_data)
    print("\n\n重点关注要素:\n\n", search_focus)
    print("\n\n用户需求解读:\n\n", user_need_profile)

    roles = create_roles(qwen_model, content,user_need_profile, know_data, stream=stream)

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

        print(f"\n\n====第{epcho}轮讨论结束，正在总结====\n\n")

        msg_content = summary_round(
            qwen_model, user_need_profile, now_date, round_record, epcho, search_focus
        )

        sugg_text, can_end = summary_sugg(
            qwen_model,
            user_need_profile,
            now_date,
            msg_content,
            his_nodes,
            know_data,
            epcho,
            MAX_EPCHO,
            search_focus
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
    user_need_profile,
    now_date,
    round_record_message,
    his_nodes,
    knowledge_snapshot,
    epcho,
    max_epcho,
    search_focus,
    stream: AIStream = None,
):
    """生成当前讨论进度、核验结果及下一步建议。"""


    system_prompt = f"""
你负责跟踪本次理论研究讨论，需依据所有输入信息完成**本轮研究进展评估、证据核验与下一阶段讨论规划**；语言应专业、克制，并与用户需求保持高度对齐。

## 基本信息
 - 当前日期：{now_date}
 - 当前轮次：第{epcho}轮 / 共{max_epcho}轮
 - 工作模式：仅在既有资料基础上做理论分析与规划，严禁安排或假定开展实验、测试、部署、业务试点等实际操作。
 - 检索关注要素：所有判断与规划都要对照下列配置，在必要处点明主要受到哪些维度（时效性、规范性、经验性、创新性、效率性）的约束。

 ```
 {search_focus}
 ```

## 用户需求解读
 ```
 {user_need_profile}
 ```
- 若解读中包含用户提供的既有知识、数据或研究方向，可视为**待核验的线索**；可以引用，但在最终结论中需要说明其来源，或在规划中安排后续核验。

## 网络搜索使用原则
1. 你可以为本轮需要核验的关键说法或重要参数设计至少3条高质量检索问题**，并通过一次 `web_search` 调用批量检索。
2. 每条检索问题应尽量包含：时间范围（或最近若干年）、主要对象（地区/行业/人群/技术类型等）以及关键限制条件，避免语义高度重叠。
3. 网络检索仅用于补充**通用事实与背景**，不得被包装成“本项目内部实验结果”。
4. 若当前环境无法实际调用检索，你需要在结论中明确说明“存在信息缺口”，并把相应内容归入“仍待讨论议题”或在下一步中安排后续核验任务。

## 本轮需要完成的任务
综合“本轮讨论概要”“历史讨论概要”“当前知识库快照”与“用户需求解读”，完成以下工作：

1. **证据与依据梳理**
   - 从本轮讨论中提取被多次引用或对结论影响较大的事实性说法、经验判断或参数设定。
   - 结合网络检索（如可用）与已有知识库，将这些内容划分为：
     - 可以视为相对可靠、且与用户需求方向一致的依据 → 进入 `verifiedBasis`；
     - 明显与公开常识或可靠资料不符、或与用户需求边界冲突的依据 → 进入 `rejectBasis`。
   - 表达时以**连续自然语言**写成一段或若干段，而不是简单列表。

2. **阶段性结论归类**
   - 依据上一步得到的“已核验依据”与“被否决依据”，对本轮出现的主要结论型表述进行分类：
     - 在当前证据与逻辑链条下，可以视为阶段性成立、且对后续研究有正向指导作用的 → 写入 `approvedContent`；
     - 基于明显不可靠依据、逻辑上存在明显漏洞、或与用户需求明显不符的 → 写入 `rejectContent`；
     - 逻辑上尚可自洽，但受限于证据不足、口径不统一或研究员之间尚未形成共识的 → 写入 `pendingContent`。
   - 三个字段都用自然语言连贯描述，可分段，但不要使用 JSON 或列表嵌套。

3. **判断是否可以结束讨论**
   - 如果用户需求中的主要研究目标已经在前几轮 + 本轮讨论中**得到较为完整的回答**，且 `pendingContent` 中仅剩一些可以在实施阶段再处理的细节，则可以将 `canEndMeeting` 设为 `True`。
   - 若仍存在与核心研究目标直接相关的重大不确定性或分歧，应将 `canEndMeeting` 设为 `False`，并在 `pendingContent` 中说明原因。

4. **下一阶段讨论规划**
   - 在 `nextStepsContent` 中，用一段或若干小段说明若继续讨论，应该围绕哪些子问题推进，例如：
     - 需要进一步统一的概念或口径；
     - 需要对比的不同技术路线或参数设定；
     - 需要从外部公开资料中补充的关键事实类型。
   - 规划内容必须是**讨论与理论分析层面**的安排，不得出现“开展实验/落地部署/上线系统/收集真实用户数据”等具体行动指令。

5. **知识库修剪建议**
   - 在 `sectionsToPrune` 中，列出若干**建议从知识库中剔除或弱化的片段**，这些片段应与 `rejectBasis` 或 `rejectContent` 中的问题直接相关，可使用原句或高度相似的摘录。
   - 每个元素为一个字符串，代表一段需要剔除或修剪的文本，便于后续进行精确删除。

## 输出格式（必须严格为 JSON）
 1.只输出一个 JSON 对象，字段固定为：
   ```json
   {{
     "verifiedBasis": "",
     "rejectBasis": "",
     "approvedContent": "",
     "rejectContent": "",
     "pendingContent": "",
     "canEndMeeting": False,
     "nextStepsContent": ""
   }}
   ```

 2.各字符串字段若无内容，可填入类似“无明显新增内容”之类的中文说明，避免留空。

[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
禁止编造具体论文、会议、作者、年份、DOI、百分比、工业 A/B 数据、公司名称等“看似真实”的细节。
若证据不足，必须写明“信息不足，以下为合理推测”并阐明前提。
所有检索与引用内容都需可追溯，禁止使用“搜索结果引用2/来源A”等无意义占位符。
[[PROMPT-GUARD v1 END]]

总结不得引入会话与已检索结果之外的全新结论，只能在此基础上提出需要补充核验或进一步讨论的动作。

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
    
    verified_basis = json_data.get("verifiedBasis", "- （无）")
    reject_basis = json_data.get("rejectBasis", "- （无）")
    approved_section = json_data.get("approvedContent", "- （无）")
    reject_section = json_data.get("rejectContent", "- （无）")
    pending_section = json_data.get("pendingContent", "- （无）")
    next_steps = json_data.get("nextStepsContent", "- （未提供）")
    canEndMeeting = bool(json_data.get("canEndMeeting", False))

    long_content = f"""
## 已核验的依据
   
   {verified_basis}

##  被否决的依据
   
   {reject_basis}

## 讨论已明确通过的结论
   
   {approved_section}

## 讨论被否决的结论
   
   {reject_section}

## 仍待讨论议题
   
   {pending_section}

""".strip()

    if not canEndMeeting:
        long_content += f"""

    ## 下一步讨论建议
    {next_steps}
    """

    return long_content, canEndMeeting


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
你是本轮讨论的记录与梳理负责人，任务是把第{epcho}轮讨论整理成一份**结构化研究小结**，供后续轮次和最终报告引用。

## 基本信息
 - 当前日期：{now_date}
 - 工作模式：只做内容归纳与结构化整理，不新增任何个人观点或外部结论。
 - 检索关注要素：下方配置可视为本课题的“研究约束背景”，在表述时保持时效性、规范性、经验性、创新性、效率性等维度的一致性。

 ```
 {search_focus}
 ```

## 输入说明
- 用户需求解读（结构化）：用于理解本课题的研究目标、研究范围与需要完成的任务。
- 本轮所有研究员的完整发言记录。

## 具体任务
1. 先用一个简短小节概括本轮讨论相对于前几轮的**主要推进点**：围绕了哪些子问题、与用户需求中的哪些目标直接相关。
2. 按时间顺序，对每位研究员的关键发言进行提炼，建议包含：
   - 该轮次中提出的主要观点或假设；
   - 引用了哪些依据或经验（如有，可用模糊表述“有研究指出/通常认为”等描述来源类型）；
   - 对他人观点的态度：支持、补充、保留意见或明确反对。
3. 归纳本轮已经形成的**初步共识**（哪类问题上观点趋于一致），以及仍存在的**主要分歧或尚未澄清的问题**。
4. 用一个简短小节总结“本轮讨论对后续研究最有价值的线索或假设”，但不要给出具体行动指令（如开展实验、部署系统等）。

## 输出格式
- 使用中文 Markdown 或普通段落均可，整体风格偏**研究记录**而不是流水式会议纪要。
- 建议的结构（可适当微调标题用语）：
  1. 本轮讨论焦点概述
  2. 各研究员主要观点与互动
  3. 本轮初步共识
  4. 本轮分歧与待澄清问题
  5. 对后续研究有价值的线索

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文/会议/作者/年份/DOI/百分比提升/工业A/B数据/公司名称等“看似真实”的细节。
- 仅可用“有研究指出/可能/推测/一般做法是……”等模糊表述指代外部工作；不得出现具体标题或精确数字。
- 允许使用网络搜索/外部资料，但仅用于通用概念与背景说明；不得将外部资料写成“本项目的真实结果”。
- 不得讲与任务无关的行业故事或案例。

【信息不足时的处理】
- 若证据不足，请明确写“信息不足，以下为合理推测”，而非下确定结论。
[[PROMPT-GUARD v1 END]]

总结不得引入本轮会话与已检索结果之外的**新论文/新工业案例/新实验数据**，仅可归纳已有讨论与通用概念。
    """

    user_prompt = f"""
#用户需求解读
```
{user_need_profile}
```

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
会议已经结束，请根据全部讨论与整理后的资料，撰写一份**研究型最终结论报告**。

## 基本信息
 - 当前日期：{now_date}
 - 本次任务仅限在既有资料与讨论基础上进行理论分析与综合，不得假定开展了新的实验或业务试点。
 - 检索关注要素：下方配置只作为约束条件，所有结论与建议应在必要处说明大致如何满足或未完全满足这些约束。 
 ```
 {search_focus}
 ```


## 写作总原则
1. 报告必须以“用户需求解读”中给出的**研究目标、研究范围和需要完成的任务**为主线，优先回答：本次研究在这些目标上取得了哪些进展，还有哪些空白。
2. 不需要逐条对照需求列出清单，但每一节内容都应在逻辑上对应到某一类需求（如：背景与动机、模型或方法设计、参数或策略选择、评估口径、风险等）。
3. 采用研究报告的写法：围绕“问题—分析—结论”的结构展开，避免仅罗列参数或零散事实。
4. 对证据不足或存在明显不确定性的地方，要主动说明原因，而不是给出过于确定的表述。

## 具体任务
1. 在理解“用户需求解读”的基础上，用一个简短小节概括本次课题的研究背景、研究对象以及需要解决的核心问题（不用逐字复述原文，但要覆盖关键约束和目标）。
2. 综合各轮讨论概要与已整理知识库，从研究者视角组织正文，建议（可根据课题适当改写标题用语）包含以下内容：
   - **研究背景与需求概述**：说明问题产生的背景、当前痛点以及本次研究聚焦的范围。
   - **研究目标与任务拆解**：按主题或子问题归纳本次需要完成的主要工作。
   - **研究思路与技术路线**：说明采用了怎样的分析框架（例如：模型，算法思路等等），以及不同方案的大致比较。
   - **阶段性研究结论**：围绕若干关键问题或子任务，总结已经比较清晰的结论或共识。
   - **关键依据与不确定性说明**：列出支撑主要结论的重要依据（可来自讨论或外部通用资料），并标记仍不完全确定、需要进一步验证的部分。
   - **仍存问题与局限**：梳理目前无法充分回答或存在明显假设前提的问题。
   - **后续工作建议与风险提示**：给出后续可以继续推进的研究方向或实施时需要注意的风险点。
3. 至少构造 3 条高质量检索问题，通过一次 `web_search` 调用补充必要的通用事实或最新背景信息；只在确有需要支撑结论时引用检索结果，并在文中以自然语言点明这是基于外部公开资料的补充说明。
4. 明确区分：
   - 来自本次讨论过程的内部观点与共识；
   - 来自外部公开资料的通用背景；
   - 在一定前提下的推断性结论（需要说明前提和不确定性）。

## 网络搜索与知识库使用说明
1. 已有知识库仅用于帮助理解课题与减少重复检索，并不等同于完整、最新的事实，因此仍须使用 `web_search` 进行补充。
2. 必须将多个检索问题打包为**一次** `web_search` 调用执行，禁止多次零散调用。
3. 不得假定已经执行了检索；若因任何原因无法检索，应在报告中说明对应结论存在信息缺口。

[[PROMPT-GUARD v1 START]]
【严禁虚构与跑题（硬性约束）】
- 禁止编造具体论文、会议、作者、年份、DOI、百分比、工业 A/B 数据、公司名称等“看似真实”的细节。
- 外部工作只能以“有研究指出 / 一些公开资料显示 / 行业内通常认为”等模糊表述引用，不得给出具体标题或精确数字。
- 允许使用网络搜索 / 外部资料，但仅用于通用概念与背景说明；不得将外部资料写成“本项目已经实证得到的结果”。
- 不得引入与当前课题无关的案例或故事。

【信息不足时的处理】
- 若证据不足，请明确写明“信息不足，以下为在若干前提下的合理推测”，并列出关键前提。
[[PROMPT-GUARD v1 END]]

## 输出格式
- 使用中文 Markdown，整体风格偏“研究型报告”。
- 一级结构至少包括（标题文字可根据课题稍作调整，但必须覆盖对应含义）：
  1. 报告标题
  2. 研究背景与需求概述
  3. 研究目标与任务拆解
  4. 研究思路与技术路线
  5. 阶段性研究结论
  6. 关键依据与不确定性说明
  7. 仍存问题与局限
  8. 后续工作建议与风险提示

"""


    user_prompt = f"""
# 原始用户需求（原文）
{dt_content}

# 讨论概要（按轮次）
```json
{json.dumps(his_nodes, ensure_ascii=False, indent=2)}
```

# 已整理知识库（用户理解背景，不代表最终事实）
```
{knowledge}
```
 
# 用户需求解读(结构化摘要)
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
        system_prompt, user_prompt, stream=stream,no_search=False
    )

    return answer
