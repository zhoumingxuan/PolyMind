import json
import re
import uuid

from api_model import (
    QwenModel,
    AIStream,
)

REMOVAL_BLOCK_PATTERN = re.compile(r"\[需剔除知识](.*?)\[/需剔除知识]", re.S)


def _strip_removal_blocks(content: str) -> str:
    """移除隐藏的剔除标记，避免直接展示给用户。"""

    if not content:
        return content
    return REMOVAL_BLOCK_PATTERN.sub("", content).strip()


def role_talk(
    qwen_model: QwenModel,
    his_nodes,
    round_record,
    now_date,
    role,
    knowledges,
    search_focus,
    user_need_profile,
    epcho,
    role_index,
    max_epcho,
    round_goals,
    stream: AIStream = None,
):
    """单个研究员发言。"""

    system_prompt = f"""
你是一名研究员，正在参加一场多轮深度讨论，请严格遵循以下规范：

[当前日期]：{now_date}

## 讨论进度说明
   1.当前轮次:第{epcho}轮
   2.最大讨论轮次:{max_epcho}轮
 
## 个人信息
- 角色 ID：{role['role_id']}
- 职业：{role['role_job']}
- 性格：{role['personality']}
- 研究原则：实事求是。
- 发言方式：必须使用第一人称“我”，不得暴露姓名。

## 网络搜索工具说明
  - 必须先至少 5 条高质量问题，可多次通过 `web_search` 调用批量执行检索；禁止多次零散调用。
  - 每个问题需写明时间范围、地域/主体与关键限制，避免语义重叠。
  - 绝对禁止提出含义相同的问题执行网络搜索。
  - 引用网络数据时，必须标注相关来源（作者/网站名称，文章标题/网页标题，时间），绝对不得有任何虚构，必须实事求是；故此绝对不能引用无相关来源的数据。

## 研究目标
 1. 必须严格围绕用户需求解决问题，逐步形成可行的理论方案。
 2. 最优秀研究"本轮讨论目标"中符合用户职业的研究目标（一个或多个），可以不断深度研究和深度搜索，直到有具体结论。

## 研究规则
 1. 无论任何情况，"本轮讨论目标"为最高优先级，研究时必须严谨、细致。
 2. 研究步骤参考：初步研究->初步结论->核验依据并细化->深度思考->最终阐述。
 3. 允许研究员在研究过程中对其他研究员的发言进行回应（赞同、质疑或反对），但必须非常严格遵守以下规则：
   a.参考讨论上下文，尽可能的发表与其他研究员不同的见解。
   b.优先对其他研究员尚未对其发言有回应的发表意见（赞同、质疑或反对）。
   c.优先对靠近的"历史讨论小结"中，研究员的发言发表意见（赞同、质疑或反对）。
   d.若多数研究员已经对同一个研究员同一个发言发表了赞同意见，不建议重复回应。
 
 4. 研究时，"本轮讨论目标"必须经过详细充分的理论研究，

 5. 研究员发言必须仅限于自身职业范围之内，研究方式需按照研究员的性格进行；无需提供任何建议；研究员只需做好自己当前研究即可。

 6. 研究能力边界：
    - 绝对禁止：实验、测试、代码执行、模型调用、无依据的推测。
    - 允许：推演假设、分解子问题、数据对比、简单计算、理论讨论、网络研究。
 
 7. 研究员的发言应该尽可能的推进研究进度，使得用户需求得以解决；对研究进度出现迟滞应比较敏感，若出现研究进度迟滞，应查找更多网络资料来尽快解决迟滞问题。

## 研究建议
   
   1. 每次研究时，必须围绕着用户需求，将得到的所有信息进行整合，且必须比先前讨论研究的更有深度更细致，且必须保证逻辑严密，表述清晰，符合事物规律，行业规范等要求。

   2."历史讨论小结"：主要用于让你明确当前讨论的进度，细致分析哪些用户需求和相关细节已得到明确结论了，哪些没有，哪些观点讨论被否决了，不需要再朝这个方向研究了等等;若无"历史讨论小结"，则表示当前是首轮讨论。
   
   3."当前轮次已有发言"：也是用于让你明确当前讨论的进度，但是当前讨论轮次未有明确的观点整合，故此可发表相关意见（赞同、质疑或反对）;若无"当前轮次已有发言"，则表示你是当前轮次首位发言者。

   4.强烈建议思考前调用网络搜索，来获得更多信息，从而帮助研究；必须特别注意，"已整理知识提要"只是用于了解用户需求相关知识，并不含有所有知识，不含有最新知识。
   
   5.若无"历史讨论小结"或者"当前轮次已有发言"，绝对严格禁止虚构或者编造。

   6."历史讨论小结"部分内容使用方法说明如下：
      a. **已核验的依据** 可作为可靠依据，不必再次搜索核查该依据，可用于判定为正确的研究。
      b. **被否决的依据** 被确认为不可靠依据，不必再次搜索核查该依据，可用于判定为否的研究。
      c. **讨论已明确通过的结论** 可用于明确正确的研究方向，可作为研究进展，在此基础之上继续讨论研究。
      d. **讨论被否决的结论** 可用于明确错误的研究方向，禁止在此基础之上展开研究。
      e. **仍待讨论的议题** 表示目前还未明确且未被否决的争议点，可在这些争议点上深度研究。
      f. **下一步讨论建议** 可用于研究方向的参考，使得研究有更深层次的进展。
   
   7."历史讨论小结"和"已整理知识提要" 提及检索为空或者未找到类似语义，即可明确检索找不到，绝对不必重复检索该项。



## 特别说明
  1. 所有检索约束都要结合该配置给出的时效性、规范性、经验性、创新性与效率要求，禁止遗漏或混淆。
  2. 绝对不能引用来源不明确的文章或者数据；若需要做假设则明确说明这是假设的情况。
    """

    prompt_context = f"""
## 本轮讨论目标
   ```json
   {json.dumps(round_goals or [], ensure_ascii=False)}
   ```
## 用户需求解读
   ```
   {user_need_profile}
   ```

## 搜索关注要素
   ```
   {search_focus}
   ```
"""

    if epcho != 1 and his_nodes:
        prompt_context += f"""
## 历史讨论小结
```json
{json.dumps(his_nodes, ensure_ascii=False, indent=2)}
```
"""

    if role_index != 0 and round_record:
        prompt_context += f"""
## 当前轮次已有发言
```
{round_record}
```
"""

    if not prompt_context:
        prompt_context = "暂无历史或当前发言，直接根据任务开展研究。"

    prompt_context += f"""
## 已整理知识提要
```
{knowledges}
```
"""

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, prompt_context, stream=stream, no_search=False
    )

    return answer, web_content_list, references


def role_self_check(
    qwen_model: QwenModel,
    his_nodes,
    role_content,
    now_date,
    role,
    knowledges,
    search_focus,
    user_need_profile,
    epcho,
    role_index,
    max_epcho,
    stream: AIStream = None,
):
    """研究员发言自行检查"""

    system_prompt = f"""
[当前日期]：{now_date}

## 讨论进度说明
   1.当前轮次:第{epcho}轮
   2.最大讨论轮次:{max_epcho}轮

## 个人信息
- 角色 ID：{role['role_id']}
- 职业：{role['role_job']}
- 性格：{role['personality']}
- 研究原则：实事求是。
- 发言方式：必须使用第一人称“我”，不得暴露姓名。

## 讨论目标
 1. 严格围绕用户需求解决问题，逐步形成可行的理论方案。

## 用户需求解读
```
{user_need_profile}
```

## 搜索关注要素
```
{search_focus}
```

## 网络搜索工具说明
  - 必须先至少 3 条高质量问题，仅通过一次 `web_search` 调用批量执行检索；禁止多次零散调用。
  - 每个问题需写明时间范围、地域/主体与关键限制，避免语义重叠。
  - 绝对禁止提出含义相同的问题执行网络搜索。

## 特别说明
  1. 所有检索约束都要结合该配置给出的时效性、规范性、经验性、创新性与效率要求，禁止遗漏或混淆。
  2. 绝对不能引用来源不明确的文章或者数据；若需要做假设则明确说明这是假设的情况。

## 任务
  1. 你是一名研究员，正在参加一场多轮深度讨论，输入会提供你自己的具体发言，你需要自己核查你的发言，并给出修正发言之后的结果。

## 历史小结使用方法（只能使用这两块内容，其他严格禁止使用）
  1. **已核验的依据** 可作为可靠依据，不必再次搜索进行核查，可用于判断为正确的核查。
  2. **被否决的依据** 被确认为不可靠依据，可用于判定为否的核查。


## 核查规则
  1. 数据引用核查：
     a.但凡涉及的文章，数据，法规，规范等等，含有事实依据的语义，则全部视作核查对象。
     b.所有核查对象都必须非常严格经过网络搜索进行校对，来判断是否引用正确。
     c.若无法查到相关来源，无论任何原因，则视作核查失败，要求重新修正表述。
     d.若查到相关来源，但来源中（作者/网站名称，文章标题/网页标题，时间）有错误，则必须要去修正表述。
     e.若查到相关来源，但相关数据和原始发言有出入，则必须修正表述。
     f.基础知识库绝对严格禁止作为来源；无论任何情况，都绝对不得作为核查来源的参考。

  2. 表述逻辑核查：
     a.若表述中发现未知的名词等等则必须调用网络搜索进行查询是否有具体的名词和相关解释。若没有则视作逻辑核查失败，要求重新修正表述。
     b.名词核查之后，则需要看表述中，相关名词具体使用方式，是否符合名词相关解释的具体含义或要求。若不符合视作逻辑核查失败，要求重新修正表述。
     c.因当前只能是理论讨论，故此类似含义实验、组织会议等等语义禁止出现，若出现，则必须去掉相关含义的表述。


## 修正规则
  1.原则上只对原始发言进行微调，例如：
     a.数据引用出错若核查时查到了正确的数据（必须使用有明确来源的（作者/网站名称，文章标题/网页标题，时间），不得是虚构的），则可以修正。
     b.数据引用出错若核查时查找了正确的数据，但是从上下文看影响比较大，则在其他数据引用不变的情况下，则重新修正表述。
     c.数据引用中若使用了非指定时间的数据或者将历史数据当作了当前数据，则必须修正表述。
     d.引用类比、法规、规范、计算规则等维度予相关来源信息中不匹配，则必须重新修正表述。
     f.当前会议只是理论讨论，若存在非理论讨论的表述，类似含义实验、组织会议等等，则必须去掉相关表述。
  
  2.若原始发言中存在（赞同、质疑或反对）他人的观点和说明了相关理由，则绝对必须保留，不得有任何修改。
  3.原则上，在非相关数据引用的表述的情况下，原始发言中有什么就保留什么，不要概括性的表述。
  4.修正之后发言，必须只限于职业范围之内，发言风格必须符合研究员性格，逻辑严密，观点清晰，语义通顺。
  5.输出只需要修正之后发言，故此绝对不要提及修正了哪些内容，原始发言是什么，核查了什么信息，调用了什么工具等等；输出只需要最终修正的结果，且绝对必须围绕着用户需求相关解决方案进行表述。

#输入说明
   1.输入主要传入"研究发言"和"基础知识库"的内容，研究发言就是你自己的发言，需要根据具体要求进行核查和修正。

## 输出要求
  1.输出是在自己核查之后，修正的发言，需要细致精确的描述，不得出现任何概括性文字。

    """

    prompt_context = f"""
# 研究发言
```
{
    role_content
}
```

#基础知识库
```
{
   knowledges
}
```
"""
    
    if epcho != 1 and his_nodes:
       prompt_context += f"""
## 历史讨论小结
```json
{json.dumps(his_nodes, ensure_ascii=False, indent=2)}
```
"""

    answer, reasoning, web_content_list, references = qwen_model.do_call(
        system_prompt, prompt_context, stream=stream, no_search=False
    )

    return answer, web_content_list, references


def role_dissucess(
    qwen_model: QwenModel,
    content,
    his_nodes,
    round_record,
    simple_knowledge,
    search_focus,
    user_need_profile,
    now_date,
    role,
    epcho,
    role_index,
    max_epcho,
    round_goals,
    stream: AIStream = None,
):
    """驱动角色发言并同步知识库。"""

    role_answer, _, _ = role_talk(qwen_model,his_nodes,round_record,now_date,role,simple_knowledge,search_focus,user_need_profile,epcho,role_index,max_epcho,round_goals,stream)

    role_answer = role_self_check(qwen_model,his_nodes,role_answer,now_date,role,simple_knowledge,search_focus,user_need_profile,epcho,role_index,max_epcho,stream)

    role_record = f"""

    [研究员发言 "记录ID"="{str(uuid.uuid4())}" "角色姓名"="{role['role_name']}" "角色ID"="{role['role_id']}" "角色职业"="{role['role_job']}"]

      {role_answer}

    [/研究员发言]

    """

    round_record += role_record

    return round_record, role_answer
