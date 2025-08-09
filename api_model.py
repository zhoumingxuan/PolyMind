import os
import dashscope
import json
import time
import re
import time
from datetime import datetime
from search_service import  web_search
from config import ConfigHelper

config = ConfigHelper()


DASHSCOPE_API_KEY=config.get("qwen_key", None)

class AIStream:
    def __init__(self):
        self.buffer=[]

    def process_chunk(self,chunk):
        print(chunk,end='',flush=True)


class QwenModel:

    def __init__(self, model_name):
        self.api_key = DASHSCOPE_API_KEY
        self.model = model_name
        self.total_tokens_count = 0

        self.tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": """
              **函数说明**
               1.用于查处网络资料的函数，支持多个网络搜索问题，但必须确保每个问题都不重复。
            """,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "question_list": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "GUID类型，每个都必须不重复"
                                    },
                                    "question": {
                                        "type": "string",
                                        "minLength": 5,
                                        "description": "需要在网络中查找的具体问题，必须明确、清晰;且必须严格确保每个问题都不重复，包括含义也都不重复。特别强调，生成的问题尽可能使用中文。"
                                    },
                                    "time": {
                                        "type": "string",
                                        "enum": ["none", "week", "month", "semiyear", "year"],
                                        "default": "none",
                                        "description": "按网页发布时间筛选：none 不限；week 最近7天；month 最近30天；semiyear 最近180天；year 最近365天。"
                                    }
                                },
                                "required": ["id", "question", "time"]
                            },
                            "description": """
               需要在互联网中查找的问题列表,数组格式，禁止生成重复的问题或者含义相同的问题。

               - 该参数为数组类型，请必须严格注意数组格式，相关格式涉及的标点符号都必须使用英文符号。

               - 最多只有10个问题。
            """
                        }
                    },
                    "required": ["question_list"]
                }
            }
        }]

    def do_tool_calls(self,tool_calls, messages):
     for tool_call in tool_calls:
      func_name = tool_call['function']['name']
      args_data = tool_call['function']['arguments']
      t_id=tool_call['id']

      startIndex = args_data.find("[")
      endIndex = args_data.rfind("]")

      if startIndex != -1 and endIndex != -1:
        args_data = args_data[startIndex:endIndex+1]

      args_data = re.sub(r'}\s*，\s*{', '},{', args_data)

      args = json.loads(args_data)

      args = {
          "question_list": args
      }

      data_content = None
      refs=[]

      if func_name == "web_search":
          data_content,ref_data = search_list(**args)
          refs.extend(ref_data)

      messages.append(
          {"role": "tool", "tool_call_id": t_id, "content": data_content})

      return data_content,refs


    def send_messages(self, messages,stream:AIStream=None,temperature=0.5,result_format="message",no_search=False,inner_search=False):
        response = None
        while response is None:
            try:
                response = dashscope.Generation.call(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024*16, # 最大输出长度
                    thinking_budget=1024*32, # 思考预算，单位是token
                    enable_thinking = True,
                    tools=None if no_search else self.tools,
                    enable_search = True if inner_search else False, # 开启联网搜索的参数
                    search_options =  {
                        "forced_search": True, # 强制开启联网搜索
                        "enable_source": False, # 使返回结果包含搜索来源的信息，OpenAI 兼容方式暂不支持返回
                        "enable_citation": False, # 开启角标标注功能
                        "search_strategy": "pro" # 模型将搜索10条互联网信息
                    } if inner_search else None,
                    stream=True,
                    include_usage=True,
                    incremental_output=True,
                    result_format=result_format
                )
            except Exception as e:
                response = None
                time.sleep(60)
                continue
        
        reasoning_content=[]
        answer_content=[]

        total_tokens=0
        tool_response={
            "role":"assistant",
            "content": "",
            "tool_calls": []
        }
        
        toolcall_infos=[]
        for chunk in response:
            # 如果思考过程与回复皆为空，则忽略
            msg = chunk.output.choices[0].message

            if chunk.get("usage"):
                total_tokens = chunk.usage.total_tokens

            if (msg.reasoning_content != "" and msg.content == ""):
                reasoning_content.append(msg.reasoning_content)
            if 'tool_calls' in msg and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    index = tool_call['index']

                    while len(toolcall_infos) <= index:
                        toolcall_infos.append({'id': '', 'name': '', 'arguments': ''})  # 初始化所有字段

                    if 'id' in tool_call:
                        toolcall_infos[index]['id'] += tool_call.get('id', '')
                        
                    # 增量更新函数信息
                    if 'function' in tool_call:
                        func = tool_call['function']
                        # 增量更新函数名称
                        if 'name' in func:
                            toolcall_infos[index]['name'] += func.get('name', '')
                        # 增量更新参数
                        if 'arguments' in func:
                            toolcall_infos[index]['arguments'] += func.get('arguments', '')

            if msg.content != "":
                chunk_content=msg.content
                if stream:
                    stream.process_chunk(chunk_content)
                answer_content.append(chunk_content)

        self.total_tokens_count += total_tokens

        if self.total_tokens_count>50000:
            time.sleep(60)
            self.total_tokens_count=0

        if len(toolcall_infos)>0:
            for t_index,tool_call in enumerate(toolcall_infos):
                item={
                    "function": {
                        "name": tool_call['name'],
                        "arguments": tool_call['arguments']
                    },
                    "id": tool_call['id'],
                    "index": t_index,
                    "type": "function"
                }
                tool_response['tool_calls'].append(item)
               
        return "".join(answer_content), "".join(reasoning_content),tool_response

    def do_call(self, system_prompt,user_prompt, stream:AIStream=None, temperature=0.5,no_search=False,inner_search=False,result_format="message"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        web_content_list=[]
        answer, reasoning,tool_response = self.send_messages(messages, stream, temperature=temperature,no_search=no_search,inner_search=inner_search,result_format=result_format)
        
        references=[]
        while len(tool_response['tool_calls'])>0:
            messages.append(tool_response)
            web_search_list,web_references=self.do_tool_calls(tool_response['tool_calls'], messages)
            web_content_list.extend(web_search_list)
            references.extend(web_references)
            answer, reasoning,tool_response = self.send_messages(messages, stream, temperature=temperature,no_search=no_search,inner_search=inner_search,result_format=result_format)

        return answer, reasoning,web_content_list,references


def create_webquestion_from_user(qwen_model:QwenModel,user_message,history_search,now_date):
   
   system_prompt = f"""

    当前日期: "{now_date}"

    # 任务

     1. 严格根据用户输入的需求（尤其注意提及的时间、地点、人物、事件、以及特别强调的内容），生成需要网络搜索的问题，确保能够对用户输入的需求有更深刻和专业的理解。关键在于根据用户提供的信息，理解其核心需求，并通过问题引导深入搜索。

     2. 若提供历史搜索数据，必须利用历史数据中的信息，围绕当前用户需求生成更有深度的问题，推动搜索结果更贴合用户的目标。历史数据应被视为已知知识库，并在问题生成过程中发挥作用。

     3. 问题生成优先级：
        - 先根据行业规范、专业定义或相关名称解释生成问题；
        - 然后生成能够进一步提升对用户需求理解的深度问题。
    
    # 强制要求与规范

     1. 严格依据用户输入中涉及的时间、地点、人物、事件、特别强调的内容，生成的问题应符合这些要求。若涉及时间，必须与当前日期核对，确保时间范围和相关事件精确无误。

     2. 当前模型处于"纯文本处理模式"。不允许使用任何外部知识或模型内置信息，只能从用户提供的文字信息中理解并生成问题。

     3. 所有生成问题的素材必须完全基于用户输入的文本。禁止参考任何外部资料或其他模型的内置知识。

     4. 生成的问题应严格符合用户需求，并旨在通过网络搜索为用户提供更深入的信息理解。无需关注立即解决用户需求，但需在进一步探索中获取更多信息。

     5. 所有问题必须明确、清晰，避免含糊或模糊不清的表述。

    # 输出

     输出格式为一个JSON数组，包含需执行网络搜索的相关问题集合。每个元素必须严格遵循以下格式：
     
        a. id: 问题ID，必须为GUID格式，确保每个问题的ID唯一。
        
        b. question: 需要执行网络搜索的具体问题。
        
        c. time: 搜索时的时间范围，枚举值为: none（不限制），week（最近7天），month（最近30天），semiyear（最近180天），year（最近365天）。

    """
   if history_search:
       system_prompt+=f"""
       #历史搜索
       1.以下是之前通过网络搜索找到的数据，数据是JSON数组格式，字段说明如下:
          a. question 搜索网络的问题

          b. result 搜索网络的结果

       2. 网络资料
         ```json
           {json.dumps(history_search, ensure_ascii=False, indent=2)}
         ```

       """
   
   user_prompt = user_message
    
   answer, reasoning,web_content_list,references =qwen_model.do_call(system_prompt, user_prompt,temperature=0.5,no_search=True)

   answer = answer.strip()
   startIndex= answer.find("[")
   endIndex = answer.rfind("]")

   if startIndex != -1 and endIndex != -1:
        answer = answer[startIndex:endIndex+1]
   else:
        raise ValueError("无法从回答中提取JSON对象")
    
   data=json.loads(answer)
   
   refs=[]
   list=[]
   for item in data:

      question=item['question']
      time=item['time']
      
      print("需要搜索的问题:",question,"\n\n")
      web_content,ref_items=web_search(question,time)

      refs.extend(ref_items)

      list.append({
          'question':question,
          'result':web_content
      })
   
   return list,refs

def search_list(question_list):

    seen = set()

    un_questions = []

    for item in question_list:
        question= item.get("question")
        if not question:
            continue
        time=item.get("time","none")
    
        key = f"Question:`{question}`====Time:`{time}`"
        if key not in seen:          # 首次出现
            seen.add(key)
            un_questions.append({"question":question,"time":time})

    if len(un_questions)==0:
        return []
    
    refs=[]
    t_list=[]
    for item in un_questions:

      question=item.get("question")
      if not question:
          continue
      time=item.get("time","none")
      
      print("需要搜索的问题:",question,"\n\n")
      web_content,refs_items=web_search(question,time)

      t_list.append({
          'question':question,
          'result':web_content
      })

      refs.extend(refs_items)
    
    return t_list,refs


def update_knowledge(qwen_model:QwenModel,now_date,content,history_know,know_list,references):
   
   system_prompt = f"""

   [当前日期]:"{now_date}"

     #用户需求

      ```
        {content}
      ```

    # 输入
      1. 用户提供的 **知识库原始内容**  
      2. 用户提供的 **最新网络搜索结果**

    # 工作流程
      1. **过滤**  
         - 从“最新网络搜索结果”中移除含有**纯粹结论、建议、计划性质的语句**；保留与用户需求高度相关的资料、数据、案例、规范、名词解释等信息。  
         - 严格避免包含没有明确来源的虚构内容。**所有资料必须有明确且可追溯的来源**（如网站、作者、时间等）。  
         
      2. **归纳**  
         - 对过滤后的内容进行**去重、分类、概括**，确保关键信息完整并**标注来源**，避免任何虚构的内容。  
         
      3. **缺失项**  
         - 对于用户需求中提及但在输入中“未找到”的信息，记录于“未找到的信息”章节，并**显式标注“未找到”**，并明确为何无法获取。

         
    # 输出
       
       - **严格仅输出下列章节标题及其内容，不得新增或省略章节，必须严格以符合用户需求，且对于用户需求来说具有高价值为最主要目标进行整理。**

       - 严格注意资料的**信息要素**（如时间、地点、人物、作者、出处等），**禁止遗漏或篡改任何信息要素**，确保每个信息都能够追溯到其来源。

       - **可针对"当前网络搜索获得的信息"进一步执行网络搜索**，如果资料不符合事实逻辑或者不够完整，进行补充和修订，保证**来源明确**，避免使用无法追溯的资料。

         - 若某章节没有内容，**请保留章节标题并写“（无）”**。  
         
         - 章节顺序：  
           ## 名词解释  
           ## 规范  
           ## 相关数据  
           ## 新闻事件  
           ## 论文参考  
           ## 示例  
           ## 其他知识点  
           ## 未找到的信息  
        
        - 每条子项以**序号开头**；如引用外部来源，请附简短出处或链接，确保**所有引用均注明来源**。

        - 不得遗失、篡改任何输入信息；可适当精简信息内容，但**必须保持信息完整性**。

        - 整理的资料必须**明确、准确地说明资料的数据来源**，如网站或网址、作者、时间、地点、人物、机构、单位、论文标题、文章出处、法规或法律条文出处等中的至少两个。如果无法准确提供来源，必须**进行适当标注**，且**不得隐瞒来源的缺失**。

        - **特别强调**：知识库内容若以标注来具体来源，则必须严格保留不得对来源执行任何修改，确保来源可以被追溯；若是多个来源合并，可以追加来源。

        - **特别强调**：来源标注必须是可靠的（例如，有明确的网站或网址、文章标题、作者、时间等中的至少两个），否则视为"来源缺失"。

        - **特别强调**：网络搜索问题本身严格禁止在来源标注中出现，只需聚焦于**网络搜索结果摘要的来源**。

        - **特别强调**：来源中不必特别区分"知识库原始内容"和"最新网络搜索结果"，这不是整理资料的目的。

   """
   
   user_prompt = f"""
     #知识库
     ```
       {history_know}
     ```

     #当前网络搜索获得的信息
     ``` json
       {json.dumps(know_list, ensure_ascii=False, indent=2)}
     ```

     #相关网络引用参考
      ``` json
        {json.dumps(references, ensure_ascii=False, indent=2)}
      ```
   """
    
   answer, reasoning,web_content_list,references = qwen_model.do_call(system_prompt, user_prompt,temperature=0.5,no_search=True,inner_search=True)

   answer = answer.strip()

   return answer
