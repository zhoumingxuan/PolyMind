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

def create_plan(qwen_model: QwenModel, content, user_need_profile, knowledges, roles, stream=None):
    """根据需求制定多轮讨论大纲"""

    system_prompt = f"""
你正在作为研究主持人，为一次多智能体“理论研究讨论”制定完整的讨论大纲。

【基本信息】
- 当前日期：{now_date}
- 最大讨论轮数：{MAX_EPCHO}
- 讨论模式：仅限理论研究与分析推理。
  严禁在任何描述中假定开展或参与以下行为：
  - 实验、测试、运行或修改代码
  - 操作或调用任何外部系统
  - 部署服务、执行交易或其他现实业务操作

【大纲制定原则】
1. 所有讨论目标必须严格围绕用户需求展开。
   - “用户需求解读”仅用于帮助你结构化理解需求，不得据此随意扩展、改变或替换用户的原始目标。
2. 必须为每一轮（从第 1 轮到第 {MAX_EPCHO} 轮）中每一个角色给出讨论计划，避免出现“某轮无人推进关键问题”的情况。
3. 每一轮、每个角色的计划只需给出本轮的**阶段性研究目标或探索方向**，而不是详细操作步骤或预设结论。
4. 计划要服务于“逐步收敛”的过程：
   - 前几轮聚焦于澄清问题、拆解结构、枚举可能的解释或路径；
   - 中间轮聚焦于比较、筛选和修正这些候选路径；
   - 最后几轮聚焦于收敛：形成理论框架、结论边界和后续待研究清单，而不是在计划中直接写出具体结论内容。

【粒度与不确定性要求】
1. 这份计划是“研究导航图”，而不是“详细操作说明书”：
   - 不要提前写出具体结论；
   - 不要在目标中堆砌大量技术细节（例如具体算法名称、参数取值、文件格式细节、设备型号等）。

2. 每个角色在每一轮的 "讨论目标"：
   - 数量为 1～2 条，禁止超过 2 条；
   - 每条目标控制在一句相对简短的中文句子内，聚焦一个核心方向，例如：
     “澄清关键概念及其相互关系”、“提出并整理若干候选假设”、“初步搭建用于后续论证的理论分析框架”等。

3. 每一轮必须给出一个 "本轮讨论目标"：
   - 数量为至少 5 条，禁止超过 10 条；
   - 这是这一轮在整体研究上希望达到的阶段性目标；
   - 用一个中文句子描述，粒度高于单个角色的目标；
   - 只描述“本轮要把问题推进到什么状态”，不直接写出具体结论。

4. 鼓励在目标中显式标记不确定性，但只描述“要探索什么”和“探索方向”，不写出“探索结果”：
   - 可以使用类似“提出若干假设并初步评估可行性”、“梳理不同可能机制的优劣，不作最终判断”
     “识别当前最重要的未知量并讨论其可能范围”等表述。

【角色使用要求】
1. “讨论角色”中给出的每个角色，都应在每一轮的“计划列表”中出现一次。
2. 输出中的 "角色名称" 必须与“讨论角色”中已有的名称或标识严格对应，不得凭空新增角色，也不得改名合并角色。
3. 不同角色的“讨论目标”应体现分工与互补：
   - 例如，有的角色偏理论推导，有的角色偏经验与案例，有的角色偏方法论与结构化整理等；
   - 禁止所有角色在同一轮给出几乎相同的目标。

【输出格式与硬性约束】
1. 你的回答必须是一个 JSON 数组，长度必须为 {MAX_EPCHO}。
   - 数组中每个元素代表一轮讨论的计划。
2. 数组中每个元素必须是结构完全一致的对象，且字段不得增删或更名：
   - "讨论轮次序号": 整数，表示讨论轮次，从 1 开始，依次递增到 {MAX_EPCHO}。
   - "计划列表": 数组，数组中每个元素是一个对象，表示本轮中某个角色的讨论计划。该对象的字段必须如下，且不得增删或更名：
       - "角色名称": 字符串，对应该角色在“讨论角色”中的名称或标识。
       - "讨论目标": 字符串数组，每个元素是一条该角色在本轮的“阶段性研究目标或探索方向”，
         用中文描述，不要带序号或前缀，数量为 1～2 条。
3. 所有字符串内容必须使用简体中文，禁止使用 Markdown 语法（如 #、-、* 等）以及任何形式的注释。
   - 即使用户提供的内容中包含 Markdown 标记，你的输出中也不得使用这些标记。

【输出规范】
 1.输出为 JSON数组结构（UTF-8编码），以下为输出的示例（仅供你理解，不要在回答中输出本示例）：
 ```json
 [
   {{
     "讨论轮次序号": 1,
     "计划列表": [
       {{
         "角色名称": "示例角色A",
         "讨论目标": ["澄清与用户需求直接相关的核心概念与问题边界"]
       }},
       {{
         "角色名称": "示例角色B",
         "讨论目标": ["提出若干候选解释或模型方向，暂不做优劣判断"]
       }}
     ],
     "本轮阶段性目标":["",""]
   }}
  ]
 ```

请依据“用户需求”，“用户需求解读”“可选参考资料”和“讨论角色”生成满足以上全部约束条件的完整 JSON 数组作为唯一输出。
"""

    user_prompt = f"""
# 用户需求（原始表述）
{content}
# 用户需求解读（供你理解结构与重点，不得改变原始需求目标）
{user_need_profile}
# 可选参考资料（仅用于理解背景与术语，可酌情参考，绝对不得作为研究方向参考）
{knowledges}
# 讨论角色
{roles}
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

    new_data=[]

    for item in data:
       new_data.append({
           "讨论轮次序号":item["讨论轮次序号"],
           "本轮阶段性目标":item["本轮阶段性目标"]
       })

    return new_data


def create_report_template(qwen_model: QwenModel, content, user_need_profile, plan, knowledges, stream=None):
    """根据需求生成最终报告模板"""

    system_prompt = f"""
你正在作为研究主持人，需要根据一次多轮理论研究讨论的整体设计，生成一份“最终研究报告”的通用章节框架。

【基本信息】
- 当前日期：{now_date}
- 讨论模式：仅限理论研究与分析推理。
  严禁在任何描述中假定开展或参与以下行为：
  - 实验、测试、运行或修改代码
  - 操作或调用任何外部系统
  - 部署服务、执行交易或其他现实业务活动

【任务目标】
1. 该模板仅用于之后撰写理论研究结论报告的“整体章节框架”，而不是现在就给出结论。
2. 你只需要规划有哪些章节、章节之间的大致顺序，以及每个章节在整份报告中的功能定位。
3. 本步骤发生在正式讨论之前，不能也不应该预设任何具体研究结论、技术路线或方向选择。

【章节设计原则】
1. 章节设计应适配当前的用户需求和多轮讨论计划，使得未来讨论产生的结果可以自然地填入这些章节中。
2. 每个章节必须有清晰的“内容要点”，用一到两句简短中文说明该章节在整份报告中的功能和作用：
   - 仅描述“本章主要承担什么信息组织或总结任务”，例如“用于说明研究背景和问题提出的缘由”；
   - 不写具体研究方法名称、不写具体研究方向、不写具体结论。
3. 可以根据需要设置若干章节，但应保持结构简洁、层次清晰，避免过细拆分。
4. 章节标题和内容要点必须具有通用性，适用于不同课题：
   - 禁止在标题或内容要点中出现具体领域术语、产品名称、公司机构名称、数据指标、设备型号等。
5. 章节整体设计必须以能够解决用户需求为**最终目标**而设计，因此需要包含有研究结论含义的章节；需要让人一看就知道，这是我想要的。

【如何利用输入信息】
1. “用户需求”与“用户需求解读”用于把握本次研究的大致问题类型和预期交付形态（例如偏概念梳理、偏方法评估、偏方案比较等），从而影响章节侧重，但仍需保持通用表述。
2. “多轮讨论计划”用于推断未来讨论大致会经历哪些阶段，从而映射为报告中哪些章节需要承接这些阶段的成果。
3. “可选参考资料”仅用于理解背景与术语，不得直接照搬其中的结论或结构到模板中。

【输出格式与硬性约束】
1. 你的回答必须是一个 JSON（UTF-8）数组,输出格式示例如下,请严格遵守如下示例的格式：
   ```json
     [
      {{
          "章节序号":1,
          "章节标题":"",
          "内容要点":""
      }}
     ]
   ```
   
2. 数组中每个元素必须是结构完全一致的对象，字段不得增删或更名：
   - "章节序号": 整数，从 1 开始递增。
   - "章节标题": 字符串，使用中性、可复用的中文标题，例如“研究背景与问题提出”“理论基础与核心概念”“分析路径与论证思路”“结论概要与研究局限”“后续研究方向与建议”等，不得包含具体领域名称或技术名词。
   - "内容要点": 字符串，用一到两句简短中文概括本章节在整份报告中的功能与目标，
     只描述“该章用于做什么类型的整理或总结”，不写具体结论，不写具体研究方向或方法名称。
3. 所有字符串内容必须使用简体中文，禁止使用 Markdown 语法（如 #、-、* 等）以及任何形式的注释。
4. 模板中不得出现具体研究结论、具体数值、真实机构名称或虚构的研究成果，只能描述“章节应该承担什么类型的内容功能”。

请依据“用户需求”“用户需求解读”“多轮讨论计划”和“可选参考资料”生成满足以上全部约束条件的报告模板 JSON 数组作为唯一输出。
"""

    user_prompt = f"""
# 用户需求（原始表述）
{content}

# 用户需求解读（供你理解结构与重点，不得改变原始需求目标）
{user_need_profile}

# 多轮讨论计划（阶段性研究目标与角色分工大纲）
{plan}

# 可选参考资料（仅用于理解背景与术语，可酌情参考）
{knowledges}
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

    return data






def create_roles(qwen_model: QwenModel, content,user_need_profile, knowledges, stream=None):
    """根据需求生成研究员角色。"""
    
    system_prompt = f"""
你正在组织一次多智能体理论研究讨论，需要按照以下约束生成 {ROLE_COUNT} 位研究员角色，用于后续多轮推理和观点碰撞。

## 基本信息
- 当前日期：{now_date}
- 讨论模式：仅限理论研究与分析推理，**禁止**假定开展或参与任何形式的实验、测试、运行代码、操作系统、部署或现实业务活动。

## 角色构成约束
1. 角色职业必须为在现实中合理存在被大众所知的职业，禁止任何虚构；且职业范围必须在“用户需求”所界定的主题和范围内。
2. 角色的职业方向必须从“用户需求”中**归纳推导**，不得脱离需求随意虚构与当前任务明显无关的领域。
3. 整体角色集合在视角和方法上应尽量形成互补，在问题理解方式、关注重点、推理习惯等方面存在差异，而不是简单重复同一种思考方式。
4. 允许不同角色具有相同或相近的职业称谓，但在以下方面必须有明显区分：
   - 性格特征与表达风格；
   - 关注的问题侧重点或分析路径。
5. 所有角色的职责应以“思考、分析、论证、评估、解释”为核心，**不得**以“执行实验、开展测试、落地实施、操作系统或设备”等现实行动为主要职责。
6. 可以参考提供的知识库理解背景，但不得直接照搬其中已有结论，将其包装成角色的既定立场。
7. 默认使用中文姓名和中文语境；如用户需求中明确指定了语言或地区要求，则角色背景可以适度贴合，但仍需服务于当前任务。
8. 所有角色必须基于当前理论研究尚未有任何结论的情况下制定，绝对不得产生会使得理论研究具有偏向性的角色。
9. 角色性格特征，主要包含以下几个方面：
   a.能够深化研究细节。
   b.能够推进研究进度，避免研究出现迟滞。
   c.能够对研究有谨慎的态度，提示研究不易被察觉的风险和注意点。
   d.能够有一些创新思维。

## 输出格式
- 只输出 JSON 数组，长度固定为 {ROLE_COUNT}，不得添加任何额外说明文字。
- 数组中每个元素为一个对象，字段严格限定为：
  - "role_name"：姓名（虚构）。
  - "role_job"：职业（必须是现实中存在的职业，且经历资深的，不必说明研究专长）。
  - "personality"：性格特征（主要是体现研究风格，要求：只要是现实中存在的，无其他特殊禁忌）。
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
    "用户提供的参考": ["用户指定的参考方向/方法/范围/比较维度（若有），可作为可靠信息，必须详细、细致的描述，不得遗漏用户需求中任何细节"]
  },
  "searchFocusProfile": {
    "timeliness": {"level": "高", "reason": "说明原因"},
    "compliance": {"level": "中", "reason": "说明原因"},
    "experience": {"level": "低", "reason": "说明原因"},
    "innovation": {"level": "中", "reason": "说明原因"},
    "efficiency": {"level": "高", "reason": "说明原因"},
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
- 禁止写“推荐/建议/方案/行动计划”等指令性内容，基础知识库只保留理解所需的事实与概念。

## 工作内容
 1. 过滤掉所有主观推断、推荐与无法追溯的描述，仅保留能让后续模型理解问题的背景、定义与事实。
 2. 细致拆解用户需求：除了交付形式、目标、细节，还需记录用户已给出的知识、引用、研究方向或假设，保证 `userNeedInsights` 足够支撑后续讨论而无需再引用原始需求。

## 输出结构
严格输出 UTF-8 JSON（不可包含反引号），参考下方示例：
{structure_example}

## 特别说明（防止网络资料篡改用户需求）
1. 在生成 `"userNeedInsights"` 时，必须以上方“用户需求”原文为唯一依据：
   - 只允许对用户原文进行**忠实的概括与结构化拆分**；
   - 绝对禁止根据“网络搜索结果”或“相关引用”擅自增加、删减或改写用户的目标、范围、约束和交付形态。
2. 当用户原文与网络资料存在不一致或冲突时：
   - 一律以用户原文为准；
   - 你可以在“知识整理”中说明“与部分公开资料存在差异”，但**严禁**据此修改 `"userNeedInsights"` 中的任何字段。
3. “网络搜索结果”和“相关引用”仅能用于：
   - 帮助你理解用户提到的术语、背景和常见做法；
   - 补充客观事实到“知识整理”部分；
   它们**不能**被写入 `"userNeedInsights"` 当作“用户的真实需求”或“用户明确提出的约束条件”。
4. `"userNeedInsights"` 中的每一条内容都必须满足：
   - 能在用户原始表述中找到对应句子，或是对此的中性、保守概括；
   - 对于仅来自你推断或网络资料的内容，禁止写入 `"userNeedInsights"`。如确有必要，可在“知识整理”或“待补充信息”中以“推测/常见情况”前缀标注。

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

    plan = create_plan(qwen_model,content,user_need_profile,know_data,roles,stream=stream)

    print("\n\n用户讨论大纲:\n\n", plan)

    report_template = create_report_template(qwen_model,content,user_need_profile,know_data,plan,stream=stream)
    
    print("\n\n最终报告模板:\n\n", report_template)

    epcho = 1
    his_nodes = []
    pending_removals = []

    while epcho <= MAX_EPCHO:
        round_record = f"""
        [第{epcho}轮讨论开始]


        """
        plan_item = next((p for p in plan if p["讨论轮次序号"] == epcho), None)
            
        if plan_item:
            round_goals = plan_item.get("本轮阶段性目标", [])
        else:
            print(f"未能获取当前轮次:{epcho}讨论目标，讨论中止")
            break

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
                round_goals,
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
   - 在 `nextStepsContent` 中，分点描述下一步的计划，要求如下：
     1. 所列出的计划必须是有助于进一步推进研究进度的。
     2. 列出计划之前必须注意，当前讨论轮数和最大讨论轮数，尽可能的在讨论结束之前完成研究。
     3. 不必重复验证已有核验的事实和结论，，不必研究无助于解决用户需求的任何细节，必须时刻关注研究进展。
     4. 若讨论中，研究员出现一些无助于解决用户需求的细节讨论，必须明确强调，避免下一步仍然讨论这些，导致研究进展迟滞。
     5. 若讨论中，有部分资料未查找到，则明确无法找到该资料，不必在下一步要求继续查找。
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
