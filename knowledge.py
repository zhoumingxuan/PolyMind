from api_model import QwenModel
import json
from search_service import web_search


def create_webquestion_from_user(
    qwen_model: QwenModel, user_message, now_date,
):
    system_prompt = f"""
你是检索问题生成器，负责为后续模型准确理解用户需求生成必要的网络搜索问题，而不是完成研究或分析本身。

## 基本信息
- 当前日期：{now_date}

## 总体目标
仅通过少量、精确的网络检索，补齐“读懂用户需求”所必需的知识空缺，例如：
- 专业术语、缩写、指标口径、数据字段含义
- 相关制度、流程、业务场景的基本定义与常见分类
- 与时间、地域、对象紧密相关且影响需求理解的基础背景（如某地区在某时间点生效的关键规则）

禁止为了扩展话题、发散研究方向或替代后续分析而提出检索问题。

## 问题生成原则
1. 每个问题只解决一个清晰的信息缺口，避免重复和过宽泛的问题。
2. 优先级顺序：
   - 关键术语、定义与指标口径
   - 与需求直接相关的基础背景或操作流程
   - 影响需求理解的前置条件或限制（如法规、接口规范、版本差异等）
3. 问题必须语义明确、可直接用于搜索，不使用“相关信息”“资料汇总”“最新情况”等模糊表达。
4. 若用户需求中包含时间约束，需要结合当前日期核对后，再将其转化为明确时间范围再写入问题。

## 输出格式
- 严格输出原始 JSON 文本，不得附加 Markdown 代码块符号、反引号、注释、自然语言前后缀或任何非 JSON 内容。
- 返回 JSON 数组，每个元素包含字段：
  - id: 字符串，使用 GUID 或其他全局唯一标识
  - question: 字符串，具体的搜索问题
  - time: 字符串，只能为 "none"、"week"、"month"、"semiyear"、"year"，用于限定结果时效性
- 若当前无需任何检索即可理解用户需求，直接输出空数组 []，同样不得添加任何解释。
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

        print("\n需要搜索的问题:", question, "\n")
        web_content, ref_items = web_search(question, time_range)

        refs.extend(ref_items)

        results.append({
            "question": question,
            "result": web_content
        })

    return results, refs

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


def rrange_knowledge(qwen_model: QwenModel, knowledges, references, now_date, user_content):
    """
    基于网络检索结果，整理出“刚好足以理解用户需求”的基础知识列表。

    返回:
      - knowledge_text: 仅包含理解需求所需的最小知识集合，可为 Markdown 风格文本
    """

    system_prompt = f"""
你是“需求理解基础知识整理助手”。你的唯一任务是：从给定的网络检索结果中，
提取刚好足以理解用户需求的最小基础知识集合，只能少，不能多。

## 当前日期
- {now_date}

## 允许输出的内容范围（只能在下面三类里选）
你只允许输出**与用户需求直接相关**的、用于“读懂用户在说什么”的信息：

1. 术语 / 缩写 / 字段名 / 指标
   - 名称、定义或含义
   - 统计或计算口径
   - 单位、取值范围（若缺少则难以理解需求）

2. 业务 / 制度 / 场景的基础说明
   - 与需求绑定的业务流程 / 制度规则 / 系统模块的**基础概念**
   - 只解释“这是什么”“大致在哪个场景用”，不讨论怎么做、做好/不好

3. 影响需求理解的关键限定条件
   - 地域（如只在某国/某市场适用）
   - 时间或版本（如某日期后生效的规则、V2 与 V3 的核心区别）
   - 对象范围（如仅针对机构户 / 某类产品）

如果某条信息是否“理解需求必需”存在任何不确定性，一律**不要写**。

## 明确禁止的内容（出现任意一类都视为错误输出）
你严禁输出以下任何内容：

- 方案、设计思路、实现步骤、最佳实践、优化建议
- 风险分析、收益分析、优缺点比较、注意事项、评价性结论
- 行业现状、发展趋势、市场规模、案例故事、成功/失败经验
- 预测、判断、个人观点、立场倾向
- 与需求只存在弱相关、可有可无的背景知识

即便这些信息出现在检索结果中，只要不是“理解需求必需”，都必须删除。

## 整理形式
- 输出为一份“基础知识列表”，用于后续帮助模型理解用户需求。
- 可以使用 Markdown 标题，有序列表组织内容，注意格式；但不能简洁，必须细致的说明，以免遗漏使用场景等关键信息。
- 每个知识点必须标注主体的来源 文章时间（必须），（网站\域名\作者\标题）中至少必须有一个；绝对不能凭空捏造，必须有具体事实引用。
- 建议表达风格（示意，不要原样照抄）：
  - 概念/字段 A：简要定义或含义说明……
  - 指标 B：口径、单位、适用范围（仅限于理解需求必需的部分）……
  - 场景 C：一句话说明“这是哪类场景/业务”，仅为理解需求服务……

## 输出格式要求（非常重要，必须严格遵守）
- 直接输出整理好的基础知识内容本身。
- 禁止使用任何 JSON 格式。
- 禁止使用反引号和代码块标记（例如 ``` 这类符号）。
- 禁止添加多余的说明性前后缀文本，例如：
  - “下面是整理结果：”
  - “总结如下：”
  - “以上是……”
- 整个回答从第一行到最后一行，都必须是“基础知识内容”本身。

## 真实性与克制
- 只能基于输入的检索结果和引用信息进行整理，禁止凭空补充或想象细节。
- 禁止编造具体论文、会议、作者、年份、DOI、精确数字、公司名称等“看起来很真实”的细节。
- 当资料不足以给出某个细节时，宁可不写，也不要猜测或用模糊语气填充。
"""

    user_prompt = f"""
# 用户需求原文（用于判断哪些信息与理解需求直接相关）
{user_content}

# 网络检索结果（供你筛选、压缩）
{json.dumps(knowledges, ensure_ascii=False, indent=2)}

# 相关引用（同样仅供筛选使用）
{json.dumps(references, ensure_ascii=False, indent=2)}
"""

    answer, reasoning, web_content_list, reference_list = qwen_model.do_call(
        system_prompt, user_prompt, no_search=True, inner_search=False
    )

    # 直接视整个回答为基础知识文本
    knowledge_text = answer.strip()
    knowledge_text = sanitize_knowledge_base(knowledge_text)

    return knowledge_text


