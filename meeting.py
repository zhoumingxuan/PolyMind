from datetime import datetime
import json
import time
import uuid
import re
from knowledge import create_webquestion_from_user,rrange_knowledge
from api_model import QwenModel, AIStream
from role import role_dissucess
from config import ConfigHelper

config = ConfigHelper()

now_date = datetime.today().strftime("%Y年%m月%d日")

MAX_EPCHO = config.get("max_epcho", 5)
ROLE_COUNT = config.get("role_count", 5)


def create_initial_solution(qwen_model: QwenModel, content, knowledge):
    """
    根据用户需求与基础知识库，生成一个“能够解决用户需求”的初步方案框架，
    供后续研究员在多轮讨论中进一步完善与讨论。
    返回值：纯文本（Markdown 风格），只描述方案本身。
    """

    system_prompt = f"""
# 任务
  1. 你需要基于用户需求构建一个解决该需求的初步方案框架。
  2. 该方案仅作为后续研究和完善的起点，不是最终结论。

# 禁止约束
  1. 绝对严格禁止产生任何“下一步计划”“后续建议”“实施步骤”等具有行动导向含义的内容。
  2. 绝对严格禁止提及实验、测试、验证、上线、部署、运行或修改代码、调用外部系统等实际操作行为。
  3. 绝对严格禁止出现与用户需求无关的内容或扩展话题。
  4. 绝对严格禁止给出确定性的最终结论，例如“已经证明……”“可以确定……”“结论是……等表述。
  5. 避免使用带有强烈指令或要求色彩的语气（如“必须”“务必”“一定要”“需要先……”等），
     更适合使用中性描述，比如“可以被设计为……”“可以被划分为……”。
  6. 绝对严格禁止与初步方案无关的任何多余表述。

# 网络搜索工具说明
  1. 基础知识只能用于理解用户需求，鼓励在构建初步方案时调用网络搜索工具补充信息。
  2. 如需检索，尽量将检索问题成组设计，一次性或少量批量调用，避免频繁零散调用。
  3. 每个检索问题应尽量明确时间范围、地域/主体和关键限制条件，避免语义重叠或含义重复。
  4. 严禁提出含义相同或高度等价的问题进行重复搜索。
  5. 如引用网络数据，在信息本身包含明确来源时，必须标注来源名称，时间；严禁虚构来源或捏造具体数字。
     对缺乏可靠来源支撑的细节，可以保持模糊或不写，禁止强行补全。
# 基本信息
  1. 当前日期：{now_date}
  2. 工作模式：仅限理论研究与分析推理。
  3. 在所有表述中，都视为尚未开展任何实际业务或技术活动，你只描述一个可供讨论的方案结构。


# 方案生成目标
  1. 方案需要紧贴用户需求，以解决用户在原文中提出的问题为主要参照，不得偏离或改写用户原始诉求。
  2. 你需要围绕同一需求构建若干可能的思路或路线（至少五个），需遵循以下要求:
        a. 它们互不相同且逻辑自洽。
        b. 它们能为完整解决用户需求提供研究方向/路线，而不是仅解决用户需求的一部分。
        c. 尽可能保证视角、场景、维度覆盖全面，可行性高，不必强行收敛为唯一路径。

  3. 在构思时，可以从不同视角或维度切入，形成若干具有代表性的备选思路，以减少潜在盲区，使对用户需求的回应更为全面和立体。
  4. 你可以根据内容需要，自主选择合适的组织方式（如小标题、分段、列表等），总体上只需让人能够大致看出：
     - 目前可能的几条主要思路或切入方向；
     - 每条思路中相对核心的环节、关注点；
     - 哪些地方明显存在不确定性、分歧空间或特别适合后续重点讨论。
  5. 上述条目仅作为引导，不要求完全套用。你可以根据具体需求自由调整表达方式，只要整体上形成一个
     “便于讨论和修改的初步方案框架”，而不是固定不变的最终结构。

    """

    user_prompt = f"""
# 用户需求原文
```
{content}
```

# 用于理解需求的基础知识（仅包含理解需求所必需的定义与背景）
```
{knowledge}
```
"""

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, user_prompt, no_search=False
    )

    solution_text = answer.strip()

    return solution_text



def summarize_and_consolidate_solutions(
    qwen_model: QwenModel,
    content: str,
    knowledge: str,
    current_solution: str,
    round_discussion: str,
    stream: AIStream | None = None,
):
    """
    第一轮讨论总结：将多个初步方案收敛为两个候选方案。
    """
    system_prompt = """
# 任务
  1.作为本次会议主席，你的任务是在讨论结束后，对所有初步想法进行总结、提炼和收敛。
  2.你需要将讨论中出现的多个研究方案，根据讨论的反馈，整合成**两个**逻辑清晰、具有代表性的候选方案。

# 工作流程
 1.  **分析讨论内容**：理解研究员们对每个初步方案的评价、补充和质疑。
 2.  **展开数据核验**：
     由于可能存在引用错误导致结论偏差，必须严格核验讨论中涉及的数据引用：
     a. 对于研究员们赞同的方案，说明理由时引用的**每一条数据**都必须进行核对，不得遗漏。
     b. 对于研究员们提出的建议，涉及的**每一条引用数据**都必须进行核对，不得遗漏。

 3.  **争议处理**：
     如遇研究员讨论存在争议，必须严格按照以下逻辑链处理：
     a. **事实核查**：发起网络搜索，查明数据是真实还是虚构；只采纳引用数据为真实的研究员描述。
     b. **视角比对**：若争议双方数据均为真实，对比思考视角，择优选取视角更全面的一方。
     c. **深度研判**：若数据均为真实且视角均全面，发起进一步的网络搜索进行深度研究，从而判断选择哪个方向来解决争议。

 4.  **数据来源补充标记**：若核对之后发现数据来源缺少，需将相关数据来源进行补充填写。

 5.  **方案参考**： 回顾参考之前初步方案框架中提到的五个研究方案及其风格。

 6.   **方案整合与构建**：
      在执行上述步骤后，构建**两个**候选方案，必须严格遵循：
      a. **完整性**：每个方案都必须是解决用户需求的细致且完整路径，确保逻辑严密且符合事实。
      b. **差异化**：两个方案必须基于事实构建（禁止虚构），且两者必须有非常明显的区别（技术路线、侧重点或哲学不同）。
      c. **迭代**：吸收讨论中的有效建议，对原始方案进行修订和完善。

7.   **清晰呈现**：
     用细致、清晰、逻辑严密的结构描述这两个方案，以便下一轮进行对比。

# 禁止约束（Critical）
  1. **绝对禁止行动导向**：严禁产生“下一步计划”、“操作指令”、“实施步骤”等内容（如“接下来需要做…”、“首先…然后…”）。
  2. **绝对禁止实操行为**：严禁提及实验、测试、验证、上线、部署、运行代码或调用外部系统。
  3. **绝对禁止无关内容**：严禁加入与用户需求无关的话题，即使讨论中曾出现。
  4. **绝对禁止终局定论**：
     - 严禁暗示某方案是最终答案。
     - 严禁使用“已证明”、“最终结论是”等措辞。
     - 应使用“当前阶段的共识”、“目前倾向于假设”等中性描述。
  5. **语气约束**：避免强烈指令色彩（如“务必”、“必须”），使用客观陈述语气。

# 网络搜索工具说明
  1. 如需检索，尽量将检索问题成组设计（每组搜索问题数量不限），一次性或少量批量调用，避免频繁零散调用。
  2. 每个检索问题应尽量明确时间范围、地域/主体和关键限制条件，避免语义重叠或含义重复。
  3. 严禁提出含义相同或高度等价的问题进行重复搜索。
  4. 如引用网络数据，在信息本身包含明确来源时，可简要标注来源名称和时间；严禁虚构来源或捏造具体数字。
     对缺乏可靠来源支撑的细节，可以保持模糊或不写，禁止强行补全。

# 输出格式
  1. 输出为一份完整的Markdown文档。
  2. 文档应包含两个明确区分的方案，例如使用“方案一”和“方案二”作为标题。
  3. 不需要解释你是如何得出这两个方案的，直接输出方案本身。
  4. 绝对不得出现与方案无关的任何描述。
"""  

    user_prompt = f"""
# 用户需求原文
```
{content}
```

# 基础知识(仅用于理解用户需求)
```
{knowledge}
```

# 初步方案框架（讨论前）
```
{current_solution}
```

# 第一轮讨论内容
```
{round_discussion}
```
"""
    answer, _, _, _ = qwen_model.do_call(system_prompt, user_prompt, no_search=False)
    new_solution = answer.strip()
    return new_solution


def summarize_and_generate_report(
    qwen_model: QwenModel,
    content: str,
    knowledge: str,
    current_solution: str,
    round_discussion: str,
    stream: AIStream | None = None,
):
    """
    第二轮讨论总结：在两个方案中选择其一，并完善成初步研究报告。
    """
    system_prompt = """
# 任务
作为会议主席，在第二轮讨论（对比两个候选方案）结束后，你的任务是：
1.  **决策选择**：根据讨论内容，判断哪个方案得到了更多的支持或被证明更具优势。
2.  **融合完善**：选择胜出的方案作为基础，并吸收讨论中对该方案的优化建议，以及另一方案中的合理部分。
3.  **生成报告**：将最终完善的方案，撰写成一份结构清晰的**初步研究报告**。

# 报告要求
-   **结构完整**：报告应包含背景、问题定义、核心方案、关键论据、潜在风险等要素。
-   **逻辑清晰**：清晰地阐述方案的内部逻辑和外部边界。
-   **保持开放性**：虽然是初步报告，但应为后续的完善和修订留出空间，明确指出当前方案的假设和待定细节。

# 禁止约束
1.  **避免终局性**：不要将报告描述为“最终报告”或“最终结论”。使用“初步研究报告”、“当前版本”等词语。
2.  **禁止行动指令**：报告内容应停留在理论分析层面。

# 输出格式
- 输出为一份完整的Markdown文档，格式为一份研究报告。
- 直接输出报告内容，无需解释决策过程。
"""
    user_prompt = f"""
# 用户需求原文
{content}

# 基础知识
{knowledge}

# 候选方案（讨论前）
{current_solution}

# 第二轮讨论内容
{round_discussion}
"""
    answer, _, _, _ = qwen_model.do_call(system_prompt, user_prompt, no_search=True)
    report = answer.strip()
    if stream:
        stream.process_chunk("\n\n[第二轮总结完成，已生成初步研究报告]\n\n" + report + "\n")
    return report


def refine_report(
    qwen_model: QwenModel,
    content: str,
    knowledge: str,
    current_report: str,
    round_discussion: str,
    stream: AIStream | None = None,
):
    """
    后续轮次：根据讨论内容，不断完善研究报告。
    """
    system_prompt = """
# 任务
作为会议记录员和编辑，你的任务是根据本轮的讨论内容，对现有的**研究报告**进行修订和完善。

# 工作流程
1.  **识别修订点**：分析讨论内容，找出对报告中特定部分的修正、补充、质疑或深化。
2.  **整合更新**：将这些讨论成果整合到报告的相应章节中，使报告内容更精确、更深入、更完善。
3.  **保持一致性**：确保更新后的报告整体逻辑一致，结构清晰。

# 禁止约束
-   **不要颠覆性修改**：除非讨论中明确达成了需要重构的共识，否则应在现有报告框架上进行增量修改。
-   **忠于讨论**：所有修改都应有本轮讨论作为依据。

# 输出格式
- 输出一份**完整的、更新后的研究报告**（Markdown格式）。
- 直接输出新版报告全文，无需标出修改痕迹。
"""
    user_prompt = f"""
# 用户需求原文
{content}

# 基础知识
{knowledge}

# 当前研究报告（修订前）
{current_report}

# 本轮讨论内容
{round_discussion}
"""
    answer, _, _, _ = qwen_model.do_call(system_prompt, user_prompt, no_search=True)
    updated_report = answer.strip()
    if stream:
        stream.process_chunk("\n\n[报告修订完成]\n\n" + updated_report + "\n")
    return updated_report



def create_roles(qwen_model: QwenModel, content, knowledges, stream=None):
    """根据用户需求生成研究员角色（性格 + 思维特长，通用化表述）。"""
    
    system_prompt = f"""
你正在组织一次多智能体理论研究讨论，需要生成固定数量（{ROLE_COUNT} 位）的研究员角色，
用于后续多轮推理和观点碰撞。

## 基本信息
- 当前日期：{now_date}
- 讨论模式：仅限理论研究与分析推理。
- 禁止在任何角色设定中假定开展或参与实验、测试、运行代码、操作系统、部署或现实业务活动。

## 角色设计原则
1. 所有角色的共同目标是：从不同思路出发，帮助更好地理解并解决当前用户需求。
2. 整体角色集合在视角和方法上应尽量形成互补：
   - 在问题理解方式、关注重点、推理习惯等方面存在差异；
   - 避免产生一组思维方式高度类似、只会重复的角色。
3. 每个角色通过“性格特征”和“思维特长”的组合来区分：
   - 性格特征体现其稳定的思考方式和沟通风格；
   - 思维特长体现其在推理、分析或整合方面相对突出的能力。
4. 思维特长需要有助于解决用户需求，但应保持为抽象的能力维度，不得是具体行业技能或岗位职能。
5. 严禁使用职业、职称、学科、研究方向、行业标签等描述角色，不得出现任何岗位称谓。
6. 所有角色的职责都以思考、分析、论证、评估、解释为核心，不围绕现实世界中的执行行为设定。
7. 可以参考提供的知识库理解背景，但不得把其中已有结论写成角色的预设立场或偏见。
8. 所有角色都应在“当前尚无最终结论”的前提下被设计，不预设任何固定答案或路线选择。

## 关于 personality 字段的约束
1. personality 只描述角色的性格特征和思维特长，使用通用语言，不绑定任何具体领域或问题场景。
2. personality 不得包含用户需求或参考资料中的专有名词、产品名称、格式名称、系统名称、算法名称等任务特定术语。
3. personality 不得描述与当前任务中具体对象、方案、路径相关的细节情景，不出现“在分析某对象时会怎样”这类表述。
4. personality 不得包含对当前任务中任何方案、路径或结论的评价、倾向或判断，不得暗含“已经更偏好某条路线”。
5. personality 中的思维特长应体现为稳定的认知能力和处理问题的方式，例如是否更偏向结构化、抽象化、整合、多角度对比、关注前提条件等，
   但不需要也不允许引用具体任务内容。
6. 性格与思维特长应同时满足：
   - 有能力把研究细节向更深层推进；
   - 有能力推动讨论继续向前发展；
   - 能保持一定的谨慎，关注容易被忽略的风险和前提条件；
   - 保留开放性和创造力，为后续讨论留出新的角度。

## 输出格式
- 只输出 JSON 数组，长度固定为 {ROLE_COUNT}，不得添加任何额外说明文字。
- 数组中每个元素必须是一个对象，字段严格限定为：
  - "role_name"：姓名（虚构、中文语境即可）。
  - "personality"：性格与思维特长的综合描述，使用一小段自然语言完成表述。
- "personality" 必须为通用表述，不得包含具体研究领域、技术路线、行业名称、岗位称谓或任务中特定名词。
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
"""

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, user_prompt, no_search=True
    )

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
        role["role_job"] = ""  # 占位以兼容后续流程，不提供任何职业信息
        role["personality"] = role.get("personality", "")

    if stream:
        description = ""
        for role in data:
            description += (
                f"角色ID:{role['role_id']}\n"
                f"角色：{role['role_name']}\n"
                f"性格：{role['personality']}\n\n"
            )
        stream.process_chunk(f"角色：\n{description}\n")

    return data


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



def start_meeting(qwen_model: QwenModel, content, stream: AIStream = None):
    """整体会议流程入口。"""
    print("\n\n用户需求:", content, "\n\n")
    
    print("\n====正在解读用户需求，构建基础知识库====\n")
    know_list,refs=create_webquestion_from_user(qwen_model,content,now_date)

    knowledge_content=rrange_knowledge(qwen_model,know_list,refs,now_date,content)

    print(knowledge_content)

    print("\n====基础知识库构建完成====\n")
    time.sleep(5)

    print("\n====初步方案构建====\n")

    solution_text = create_initial_solution(qwen_model,content,knowledge_content)

    print(solution_text)

    print("\n====初步方案构建完成====\n")

    print("\n====构建角色====\n")

    roles = create_roles(qwen_model, content, knowledge_content, stream=stream)

    print("\n====角色构建完成====\n")
    
    time.sleep(5)
    print("\n====开始讨论====\n")

    
    epcho = 1
    his_nodes = []
    # pending_removals = []

    while epcho <= MAX_EPCHO:
        round_record = f"""
        [第{epcho}轮讨论开始]
        """
        print(f"\n====第{epcho}轮讨论开始====\n")
        if stream:
            stream.process_chunk(f"\n\n====第{epcho}轮讨论开始====\n")

        for role_index, role in enumerate(roles):
            if stream:
                stream.process_chunk(
                    f"\n\n角色：{role['role_name']}\n性格和思维特长：{role['personality']}\n\n"
                )
            
            round_record, role_answer=role_dissucess(qwen_model,content,round_record,solution_text,now_date,role,knowledge_content,epcho,role_index,len(roles),MAX_EPCHO,stream)
        
        print(f"\n\n====第{epcho}轮讨论结束，正在总结====\n\n")
        if epcho == 1:
            # 第一轮总结：将多个初步方案收敛为两个候选方案
            solution_text = summarize_and_consolidate_solutions(
                qwen_model, content, knowledge_content, solution_text, round_record, stream
            )
            print("\n\n====第一轮总结完成，已形成两个候选方案====\n\n" + solution_text + "\n")
            break
            
        elif epcho == 2:
            # 第二轮总结：在两个方案中选择其一，并完善成初步研究报告
            print("\n====第二轮总结：正在生成初步研究报告====\n")
            solution_text = summarize_and_generate_report(
                qwen_model, content, knowledge_content, solution_text, round_record, stream
            )
            print(solution_text)
        else:
            # 后续轮次：根据讨论内容，不断完善研究报告
            print(f"\n====第{epcho}轮总结：正在修订研究报告====\n")
            solution_text = refine_report(
                qwen_model, content, knowledge_content, solution_text, round_record, stream
            )
            print(solution_text)

        his_nodes.append(round_record)
        epcho+=1
            

    
    print("\n\n====所有讨论已结束====\n\n")
    if stream:
        stream.process_chunk("\n\n====所有讨论已结束====\n\n")

    return solution_text

    # knowledges = None
    # knowledges, refs = create_webquestion_from_user(
    #     qwen_model, content, knowledges, now_date
    # )
    
    # while True:
    #  try:
    #   know_data, search_focus, user_need_profile = rrange_knowledge(
    #     qwen_model, knowledges, refs, content
    #   )
    #   break
    #  except Exception:
    #   print("\n出现返回错误，资料重新整理\n")
    #   time.sleep(60)
    #   continue
      

    # print("\n\n基础资料:\n\n", know_data)
    # print("\n\n重点关注要素:\n\n", search_focus)
    # print("\n\n用户需求解读:\n\n", user_need_profile)

    # roles = create_roles(qwen_model, content,user_need_profile, know_data, stream=stream)

    # plan = create_plan(qwen_model,content,user_need_profile,know_data,roles,stream=stream)

    # print("\n\n用户讨论大纲:\n\n", plan)

    # report_template = create_report_template(qwen_model,content,user_need_profile,know_data,plan,stream=stream)
    
    # print("\n\n最终报告模板:\n\n", report_template)

    # epcho = 1
    # his_nodes = []
    # pending_removals = []

    # while epcho <= MAX_EPCHO:
    #     round_record = f"""
    #     [第{epcho}轮讨论开始]


    #     """
    #     plan_item = next((p for p in plan if p["讨论轮次序号"] == epcho), None)
            
    #     if plan_item:
    #         round_goals = plan_item.get("本轮阶段性目标", [])
    #     else:
    #         print(f"未能获取当前轮次:{epcho}讨论目标，讨论中止")
    #         break

    #     for role_index, role in enumerate(roles):
    #         if stream:
    #             stream.process_chunk(
    #                 f"\n\n角色：{role['role_name']}\n职业：{role['role_job']}\n性格：{role['personality']}\n\n"
    #             )
            

    #         new_record, role_answer = role_dissucess(
    #             qwen_model,
    #             content,
    #             his_nodes,
    #             round_record,
    #             know_data,
    #             search_focus,
    #             user_need_profile,
    #             now_date,
    #             role,
    #             epcho,
    #             role_index,
    #             MAX_EPCHO,
    #             round_goals,
    #             stream=stream,
    #         )
    #         round_record = new_record

    #    

    #     msg_content = summary_round(
    #         qwen_model, user_need_profile, now_date, round_record, epcho, search_focus
    #     )

    #     sugg_text, can_end = summary_sugg(
    #         qwen_model,
    #         user_need_profile,
    #         now_date,
    #         msg_content,
    #         his_nodes,
    #         know_data,
    #         epcho,
    #         MAX_EPCHO,
    #         search_focus
    #     )

    #     last_content = f"""
    #     # 第{epcho}轮讨论总结

    #     ## 当前讨论概要
    #     ```
    #     {msg_content}
    #     ```

    #     ## 当前讨论进度和建议
    #     ```
    #     {sugg_text}
    #     ```
    #     """

    #     print("\n\n====当前讨论小结====\n", last_content)

    #     his_nodes.append(last_content)

    #     if can_end:
    #         print("\n====讨论中止====\n")
    #         break

    #     epcho += 1

    # if stream:
    #     stream.process_chunk("\n[研究讨论结束]\n\n")

    # print("\n====输出最终报告====\n")

    # return summary(
    #     qwen_model,
    #     content,
    #     now_date,
    #     his_nodes,
    #     know_data,
    #     search_focus,
    #     user_need_profile,
    #     stream=stream,
    # )


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
