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
2. 提问语句形式，示例如下（不必完全照搬，理解该用怎样的语句形式即可，******仅表示示例填充文本，不得出现在提问中，也不得被原样输出为问题）：
       - 什么是 ******
       - ****** 是什么（在 ****** 领域/语境下）
       - ****** 的定义/边界是什么（与 ****** 的区别）
       - ****** 的英文全称/常见缩写/别名是什么
       - ****** 指标的统计口径/计算方法是什么（单位/是否年化/是否含税等）
       - ****** 数据字段（字段名/英文名）的含义与取值范围是什么（来源/接口文档）
       - ****** 的枚举值/分类标准是什么（代码表/字段字典）
       - ****** 的标准流程/关键步骤是什么（输入/输出/参与方）
       - ****** 在 ****** 场景下的适用条件/前置要求是什么
       - ****** 在 ****** 地区/监管口径下的规定是什么（自 YYYY-MM-DD 起/截至 YYYY-MM）
       - ****** 在 ****** 版本/标准号中的差异或变更点是什么（vX.Y / 标准版本）
       - ****** 的最新要求是什么（发布主体/文号/生效时间/适用范围）
       - ****** 的最新规范是什么（标准名称/版本号/发布主体/发布日期）
       - ****** 的最新法规内容是什么（标准名称/版本号/发布主体/发布日期）

3. 优先级顺序：
   - 关键术语、定义与指标口径
   - 与需求直接相关的基础背景或操作流程
   - 影响需求理解的前置条件或限制（如法规、接口规范、版本差异等）
4. 问题必须语义明确、可直接用于搜索，不使用“相关信息”“资料汇总”“最新情况”等模糊表达。
5. 若用户需求中包含时间约束，需要结合当前日期核对后，再将其转化为明确时间范围再写入问题。

## 输出格式
- 严格输出原始 JSON 文本，不得附加 Markdown 代码块符号、反引号、注释、自然语言前后缀或任何非 JSON 内容。
- 返回 JSON 数组，长度必须大于等于3，每个元素包含字段：
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

    results = []
    for item in data:
        question = item.get("question","")
        time_range = item.get("time","none")

        print("\n正在搜索的问题:", question, "\n")
        web_content = web_search(question, time_range)

        results.append({
            "question": question,
            "result": web_content
        })

    return results

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


def rrange_knowledge(qwen_model: QwenModel, knowledges, now_date, user_content):
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


## 明确禁止的内容（出现任意一类都视为错误输出）
你严禁输出以下任何内容：

- 方案、设计思路、实现步骤、最佳实践、优化建议
- 风险分析、收益分析、优缺点比较、注意事项、评价性结论
- 行业现状、发展趋势、市场规模、案例故事、成功/失败经验
- 预测、判断、个人观点、立场倾向
- 与需求只存在弱相关、可有可无的背景知识

即便这些信息出现在检索结果中，只要不是“理解需求必需”，都必须删除。

## 整理流程
- 第一步（筛选）：拿到"网络检索结果"提取与用户需求有关的信息（尽可能的所有保留细节，禁止遗漏），并且**必须原样保留每一项与用户信息相关中来源（包括来源主体和时间）**。
- 第二步（拆分）：将第一步拿到的信息做个拆分，确保每一条为一个独立的知识点，同时把每个知识点的来源（包括来源主体和时间）**原样照搬**过来。
- 第三步（精炼）：将第二步拿到的信息进一步筛选，只保留恰好用于理解用户需求的知识点，剔除额外信息。
- 第四步（归类和加工）：综合第三步的情况，进行归类，看哪些知识点可以放在一块综合描述，则合并为一个知识点，同时必须保留相关来源（包括来源主体和时间);知识点的描述风格改为类似"字典"风格的描述说明。
- 第五步（去重和检查）：剔除第四步描述重复的知识点；另外，若第四步的来源出现遗漏，则从"网络检索结果"查找对应含义的文字描述，找到对应的来源（包括来源主体和时间）并从中**原样照搬**，补充回知识点。

## 整理格式
- 必须使用 Markdown 格式，有序列表组织内容，注意格式；但不能简洁，必须细致的说明，以免遗漏使用场景等关键信息。

## 输出要求（非常重要，必须严格遵守）
- 从第一步到第五步必须严格按照顺序执行，中间不得遗漏任何步骤，输出内容为执行完第五步的内容。
- 禁止输出与"基础知识"无关的任何描述或文本，也无需任何强调提示。
- 禁止使用反引号和代码块标记（例如 ``` 这类符号）。
- 禁止添加多余的说明性前后缀文本，例如：
  - “下面是整理结果：”
  - “总结如下：”
  - “以上是……”
- 整个回答从第一行到最后一行，都必须是“基础知识内容”本身。
- 删除或者过滤掉的内容的绝对禁止输出。
- 最终整理好的内容，序号必须从1开始，严格按顺序依次编号。

## 特别说明（非常重要）
1. 只能基于输入的检索结果和引用信息进行整理，禁止凭空补充或想象细节。
2. 绝对禁止使用“综合/推断/网络检索结果第N条”来替代。
3. 宁可不写，也不要猜测或用模糊语气填充。
4. 严禁在输出中出现任何“删除/已删除/故此条删除/未找到对应条目”等字样。
5. 严禁解释你为什么不写某条内容。
"""

    user_prompt = f"""
# 用户需求原文（用于判断哪些信息与理解需求直接相关）
```
{user_content}
```

# 网络检索结果(用于提取关键信息)
```
{json.dumps(knowledges, ensure_ascii=False, indent=2)}
```

"""

    print("\n\n知识条目:",knowledges)

    answer, reasoning, web_content_list, reference_list = qwen_model.do_call(
        system_prompt, user_prompt, no_search=True, inner_search=False
    )

    # 直接视整个回答为基础知识文本
    knowledge_text = answer.strip()
    knowledge_text = sanitize_knowledge_base(knowledge_text)

    return knowledge_text


