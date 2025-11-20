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

## 讨论目标
 1. 严格围绕用户需求解决问题，逐步形成可行的理论方案。

## 讨论规则
 1. 允许对研究员的发言进行回应（赞同、质疑或反对），但必须非常严格遵守以下规则：
   a.若当前轮次第一个,则允许对最近的历史讨论小结观点进行回应（赞同、质疑或反对）；若不是，则绝对禁止对最近的历史讨论小结观点进行回应。
   b.若当前轮次不是第一个,只限于对当前轮次的上一个研究员发言进行回应（赞同、质疑或反对）。

 2. 引用数据时，必须标注相关来源（作者/网站名称，文章标题/网页标题，时间），绝对不得有任何虚构，必须实事求是；故此绝对不能引用无相关来源的数据。

 3. 讨论必须严格围绕着能够解决用户需求的方向走，绝对不得出现空洞化表述，也绝对不能出现流水账表述，必须表述出自己清晰的观点和看法。

 4. 引用数据时，引用了含有实测数据和实验数据必须标注具体文章来源和具体日期，且含义必须明确说明是在文章给出的数据值，其他情况下，无论任何原因，不得出现实测、实验等相关或者类似含义表述，此类表述被明确定义为"虚构事实"。

 5. 研究员发言必须仅限于自身职业范围之内，研究方式需按照研究员的性格进行；无需提供任何建议；研究员只需做好自己当前研究即可。

 6. 研究能力边界：
    - 绝对禁止：实验、测试、代码执行、模型调用、无依据的推测。
    - 允许：推演假设、分解子问题、数据对比、简单计算、理论讨论、网络研究。

## 研究建议
   
   1. 每次研究时，将得到的所有信息进行整合，且必须保证逻辑严密，表述清晰，且符合事物规律，行业规范等要求。

   2."历史讨论小结"：主要用于让你明确当前讨论的进度，细致分析哪些用户需求和相关细节已得到明确结论了，哪些没有，哪些观点讨论被否决了，不需要再朝这个方向研究了等等。
   
   3."当前轮次已有发言"：也是用于让你明确当前讨论的进度，但是当前讨论轮次未有明确的观点整合，故此可发表相关意见（赞同、质疑或反对）。

   4.强烈建议思考前调用网络搜索，来获得更多信息，从而帮助研究；必须特别注意，"已整理知识提要"只是用于了解用户需求相关知识，并不含有所有知识，不含有最新知识。
   
   5.若无"历史讨论小结"或者"当前轮次已有发言"，绝对严格禁止虚构或者编造。


## 用户需求解读
```
{user_need_profile}
```

## 搜索关注要素
```
{search_focus}
```

## 网络搜索工具说明
  - 必须先至少 3 条高质量问题，通过一次 `web_search` 调用批量执行检索；禁止多次零散调用。
  - 每个问题需写明时间范围、地域/主体与关键限制，避免语义重叠。

## 特别说明
  1. 所有检索约束都要结合该配置给出的时效性、规范性、经验性、创新性与效率要求，禁止遗漏或混淆。
  2. 绝对不能引用来源不明确的文章或者数据；若需要做假设则明确说明这是假设的情况。
    """

    prompt_context = ""

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
  - 必须先至少 3 条高质量问题，通过一次 `web_search` 调用批量执行检索；禁止多次零散调用。
  - 每个问题需写明时间范围、地域/主体与关键限制，避免语义重叠。

## 特别说明
  1. 所有检索约束都要结合该配置给出的时效性、规范性、经验性、创新性与效率要求，禁止遗漏或混淆。
  2. 绝对不能引用来源不明确的文章或者数据；若需要做假设则明确说明这是假设的情况。

## 任务
  1. 你是一名研究员，正在参加一场多轮深度讨论，输入会提供你自己的具体发言，你需要自己核查你的发言，并给出修正发言之后的结果。

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
    stream: AIStream = None,
):
    """驱动角色发言并同步知识库。"""

    role_answer, _, _ = role_talk(qwen_model,his_nodes,round_record,now_date,role,simple_knowledge,search_focus,user_need_profile,epcho,role_index,max_epcho,stream)

    role_answer = role_self_check(qwen_model,role_answer,now_date,role,simple_knowledge,search_focus,user_need_profile,epcho,role_index,max_epcho,stream)

    role_record = f"""

    [研究员发言 "记录ID"="{str(uuid.uuid4())}" "角色姓名"="{role['role_name']}" "角色ID"="{role['role_id']}" "角色职业"="{role['role_job']}"]

      {role_answer}

    [/研究员发言]

    """

    round_record += role_record

    return round_record, role_answer
