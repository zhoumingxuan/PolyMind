from api_model import QwenModel,AIStream,create_webquestion_from_user,search_list
from datetime import datetime
import json
import uuid
import re
from role import role_dissucess
import time
from config import ConfigHelper

config = ConfigHelper()

now_date = datetime.today().strftime('%Y年%m月%d日')

MAX_EPCHO=config.get("max_epcho", 5)

ROLE_COUNT=config.get("role_count", 5)

def create_roles(qwen_model:QwenModel,content,knowledges,stream=None):
    """创建研究的角色"""
    
    system_prompt = f"""
当前日期:"{now_date}"

#任务
  1. 你处于深度研究任务中，你需要根据用户需求，生成{ROLE_COUNT}个非常合适的研究员，生成他们的姓名，职业以及性格，以JSON格式输出。
  2. 生成的角色必须与用户需求密切相关，但**不应直接基于已有知识库结论**，必须从用户需求出发，**鼓励生成不同职业特性的角色**，可以多角度去思考用户需求，确保研究的全面性和深度。 
  3. 每个研究员的职业不应单一指向某一领域，而是要从**多个相关领域**考虑，保持**职业多样性**。例如，如果需求偏向技术研究，可以生成科研人员、企业管理人员、市场专家等多种角色，确保多元视角。
  4. 角色之间应具有足够的多样性和互补性，避免所有角色都来自同一行业或领域。通过不同背景、性格和职业的碰撞，推动更加深入的讨论。
  5. 用户需求若无特定说明具体的国家或地域，生成的角色则默认为中国人。
  6. 职业必须在现实中实际存在，**严格禁止虚构职业**；允许有相同职业的角色，但必须有不同的性格特征。
  7. 必须存在至少**两个具备广泛性职业的角色**。这些角色应该具有足够的**广泛性和包容性**，能够代表不同领域的广泛视角。例如，可以选择跨学科领域的专家，既能在某一领域深耕，也能对不同领域保持宽泛的理解。

#研究能力边界#

  * 禁止：任何实验、任何测试、任何非理论性验证、任何代码执行、任何模型调用。
  * 允许：数据对比、简单计算、理论讨论、网络研究。


#输出要求
  1.输出必须严格按照JSON格式，输出类型是JSON格式的数组，有且只有{ROLE_COUNT}个元素，每个元素代表着每个研究员的身份信息。
  2.每个元素中，字段说明如下：
    a. role_name 研究员的姓名。
    b. role_job 研究员的职业，必须适合能够解决当前用户需求的职业，且**不重复或过于单一**，确保角色多样性；必须完全符合研究能力边界的要求。
    c. personality 研究员的性格。

#知识库（如需要）
  1. 如果用户需求中某些背景信息不足，参考以下资料可用于补充角色的背景，帮助理解用户需求。请**仅在有认知缺口时**参考，不要直接从知识库得出结论。
    ```
    {knowledges}
    ``` 
  """
    
    user_prompt = f"""
      #用户需求
      ```
        {content}
      ```
    """
    
    answer, reasoning,web_content_list,references = qwen_model.do_call(system_prompt, user_prompt,no_search=True)

    answer = answer.strip()
    startIndex= answer.find("[")
    endIndex = answer.rfind("]")

    if startIndex != -1 and endIndex != -1:
        answer = answer[startIndex:endIndex+1]
    else:
        raise ValueError("无法从回答中提取JSON对象")
    
    data=json.loads(answer)

    for role in data:
        role['role_id']= str(uuid.uuid4())
    
    if stream:
       des_content=""
       for role in data:
            des_content += f"角色ID:{role['role_id']}\n角色名: {role['role_name']}\n 职业: {role['role_job']}\n 性格: {role['personality']}\n\n"

       stream.process_chunk(f"角色:\n {des_content}\n")
    
    return data

def rrange_knowledge(qwen_model:QwenModel, knowledges,references, user_content):
    """整理资料"""
    
    system_prompt = f"""
    当前日期:"{now_date}"
    
    #需求
    ``` 
    {user_content}
    ```

    #任务

      1. 输入提供的是从网络检索得到的资料信息。你需要基于这些资料和具体需求，整理并提取出能帮助大模型对用户需求进行更深入、专业理解的信息；所有信息必须**为最新的**，且确保来自**可靠来源**，避免使用陈旧或无效的数据（最核心任务）。

      2. 整理的内容必须保留网络资料中的细节，禁止遗漏或篡改任何信息。目标是为后续的讨论和分析提供支持，绝对不能直接解答用户需求。

      3. 整理内容时，需过滤信息，保留与用户需求**紧密相关的背景知识、定义、概念、术语、法律法规、现状描述和相关数据**。特别注意，标记关键信息，如网站或网址、作者、时间、地点、人物、机构、单位、论文标题、文章出处、法规或法律条文出处等中的至少两个，避免遗漏。

      4. 严格避免包含没有明确来源的虚构内容。**所有资料必须有明确且可追溯的来源**（如网站或网址、作者、时间、地点、人物、机构、单位、论文标题、文章出处、法规或法律条文出处等中的至少两个）。  

      5. 严格去除与用户需求无关的内容。确保所有整理的内容为**后续深入讨论提供基础**，并避免在此阶段直接提出解决方案或结论。

      6. 整理的内容必须严格围绕需求描述，确保细节精确且易于理解。

      7. 特别强调，来源标注必须是可靠的（例如，有明确的网站或网址、作者、时间、地点、人物、机构、单位、论文标题、文章出处、法规或法律条文出处等中的至少两个），否则一律视作"来源缺失"。

    #输出

      1. 整理的结果应该按照Markdown格式输出，每个知识点都要有明确的描述。
      
      2. 若现有资料不完整或不符合逻辑或未标注来源，**可以在此阶段进一步执行网络搜索**，用于补充和修正已有内容，但不可直接开始解决需求。

      3. 输出的内容应按列表形式组织，每个知识点可以详细分解，确保便于后续讨论使用。

    #输入
      1. 输入的内容是从网络检索得到的资料信息，包含两部分内容：

         a. 网络搜索结果

         b. 相关网络引用

    """
    user_prompt = f"""
    #网络搜索结果
    ``` json
      {json.dumps(knowledges, ensure_ascii=False, indent=2)}
    ```
    #相关网络引用
    ``` json
      {json.dumps(references, ensure_ascii=False, indent=2)}
    ```
    """
    
    answer, reasoning, web_content_list,references = qwen_model.do_call(system_prompt, user_prompt, no_search=True, inner_search=True)
    
    return answer


def start_meeting(qwen_model:QwenModel,content,stream:AIStream=None):
    """开始会议"""
    print("\n\n用户需求:",content,"\n\n")
    
    knowledges=None
    knowledges,refs=create_webquestion_from_user(qwen_model,content,knowledges,now_date)


    know_data=rrange_knowledge(qwen_model,knowledges,refs,content)

    print("\n\n基础资料:\n\n",know_data)
  

    roles=create_roles(qwen_model,content,know_data,stream=stream)

    epcho=1
    
    his_nodes=[]

    while epcho<=MAX_EPCHO:
       round_record=f"""
       [第{epcho}轮讨论开始]


       """
       for role_index, role in enumerate(roles):   
        if stream:
            stream.process_chunk(f"\n\n角色名: {role['role_name']}\n 职业: {role['role_job']}\n 性格: {role['personality']}\n\n")
        
        new_record,new_know=role_dissucess(qwen_model,content,his_nodes,round_record,know_data,now_date,role,epcho,role_index,MAX_EPCHO,stream=stream)
        #知识库更新
        know_data=new_know

        round_record=new_record

         
       
       print(f"\n\n====第{epcho}轮讨论结束，正在总结，请耐心等待====\n\n")

       msg_content=summary_round(qwen_model,content,now_date,round_record,epcho)

       sugg_content,can_end=summary_sugg(qwen_model,content,now_date,msg_content,his_nodes,epcho,MAX_EPCHO)

       last_conent=f"""
        #第{epcho}轮讨论总结
           
           ## 当前讨论概要
              ```
                {msg_content}
              ```
           
           ## 当前讨论进度和建议
              ```
                 {sugg_content}
              ```
       """

       print("\n\n====当前讨论小结====\n",last_conent)

       his_nodes.append(last_conent)

       if can_end:
          print("\n====讨论中止====\n")
          break

       epcho+=1

    stream.process_chunk(f"\n[研究讨论结束]\n\n")
    
    print("\n====输出最终报告====\n")

    return summary(qwen_model,content,now_date,his_nodes,know_data,stream=stream)


def summary_sugg(qwen_model:QwenModel,content,now_date,round_record_message,his_nodes,epcho,max_epcho,stream:AIStream=None):
    """当前讨论进度和建议"""

    system_prompt = f"""
    [当前日期]:"{now_date}"
    [当前讨论轮次]:"第{epcho}轮讨论"
    [最大讨论轮数]:"{max_epcho}轮"
    [当前模式]:"深度思考"

    #用户需求
    ```
      {content}
    ```
    
    # 输入说明
      用户输入包含两部分内容：
      1. 本轮讨论概要
      2. 历史讨论轮次概要

    # 任务
      当前处于深度研究讨论任务中，你必须严格依据用户输入的上述两部分信息，完成以下任务：

      1. **总结当前会议进展**：
         a. 已明确达成一致且形成结论的内容（**使用有序列表清晰列出**，提供详细说明）。  
         b. 尚未达成结论的内容（**使用有序列表清晰列出**，详细说明每个问题及其待解之处）。
         
      2. **判断讨论是否足够细致，是否具备结束条件**：
         a. 若可以结束，明确指出并说明理由。  
         b. 若无法结束，**提出具体且有深度的下一步讨论建议**，并特别指出未讨论充分的部分。**建议仅限于理论讨论，禁止实验、测试、代码执行等行为**。

    #研究能力边界#

        * 禁止：任何实验、任何测试、任何非理论方式进行验证、任何代码执行、任何模型调用。
        * 允许：数据对比、简单计算、理论讨论、网络研究。


    #输出规范

      1.输出格式应为**JSON对象**，字段要求如下，请严格按照字段要求输出：

        a. **approvedContent**（字符串）：已明确通过的结论内容，使用**有序列表**清晰列出并详细说明。  
        b. **pendingContent**（字符串）：仍未形成结论的内容，使用**有序列表**清晰列出，并详细说明原因。  
        c. **nextStepsContent**（字符串）：对下一步更有深度的理论讨论建议。必须避免历史讨论中的重复内容，提供新的深度观点。  
        d. **canEndMeeting**（布尔值）：若讨论已充分、符合结束条件，返回True；否则返回False。

    # 结束会议条件

      1. 若达到最大讨论轮数（max_epcho），视作会议必须结束，建议提前提供必要建议帮助顺利收尾。
      2. **仅当所有讨论内容已明确得到结论**，**完全满足用户需求并达到目标**，且**没有较大分歧**时方可结束会议。
      3. 任何**存在分歧**且尚未充分讨论的议题，**并不满足用户需求**或**未达到目标**，都不得视作会议结束条件。

    """

    user_prompt = f"""

    #本轮讨论概要
    ```
      {round_record_message}
    ```

    #历史讨论轮次概要
    ```
      {json.dumps(his_nodes, ensure_ascii=False, indent=2)}
    ```

    """

    answer, reasoning,web_content_list,references = qwen_model.do_call(system_prompt, user_prompt,stream=stream,no_search=True)

    answer = re.sub(r'}\s*，\s*{', '},{', answer)

    startIndex= answer.find("{")
    endIndex = answer.rfind("}")

    if startIndex != -1 and endIndex != -1:
      answer = answer[startIndex:endIndex+1]

    json_data=json.loads(answer)
     
    cem=False
    long_content=f"""
    ##讨论已明确通过的结论

       {json_data['approvedContent']}

    ##讨论且尚未形成结论

       {json_data['pendingContent']}
    """

    if json_data['canEndMeeting'] is False:
       
       long_content+=f"""
       ##下一步讨论建议

           {json_data['nextStepsContent']}
       """
    else:
       cem=True
        
    return long_content,cem

def summary_round(qwen_model:QwenModel,content,now_date,round_record,epcho,stream:AIStream=None):
    """一轮讨论小结"""
    
    system_prompt = f"""

    [当前日期]:"{now_date}"
    [当前讨论轮次]:"第{epcho}轮讨论"

    #用户需求

      ```
        {content}
      ```

    
    #输入
     1.用户会输入本轮讨论中所有研究员的发言内容

    #任务

      1你根据本轮讨论中所有研究员的发言内容进行整理，必须严格按照以下要求对发言进行整理：
          
          a.必须围绕着课题信息，严谨、细致的整理每个研究员的发言内容，一一列出，不得出现任何遗漏。

          b.整理时，不得遗漏研究员发表的任何观点，以及表达的"赞同"、"质疑"、"否决"的内容。

    #输出要求

      1.输出的内容为根据用户输入的本轮讨论中所有研究员的发言内容，得到的当前讨论轮次的具体概要信息，严格禁止输出其他不相关的内容。

    
    """

    user_prompt = f"""
    #本轮讨论中所有研究员的发言内容
    ```
      {round_record}
    ```
    """

    answer, reasoning,web_content_list,references = qwen_model.do_call(system_prompt, user_prompt,stream=stream,no_search=True)

    return answer



def summary(qwen_model:QwenModel,dt_content,now_date,his_nodes,knowledge,stream:AIStream=None):
    """总结会议内容"""
    
    system_prompt = f"""
    [当前日期]:"{now_date}"

    #用户需求
    ```
      {dt_content}
    ```

    #任务

      1. 你在一个研究讨论会议中，目前会议已经结束，你需要对讨论的内容进行整理，并为此提供清晰的结构化总结。
      2. 首先，你要从"每一轮讨论的概要信息"中整理出讨论的主要观点和关键信息。
      3. 在整理完会议内容后，你可以搜索相关的网络资料，进一步丰富和补充总结内容。
      4. 最终输出必须包括以下内容：
         - 详细且准确的会议总结。
         - 如果存在创新性观点或名词，单独列出并提供定义和解释。
         - 结合网络搜索结果，确保所有内容全面、准确，并且符合当前讨论的主题。
      
    # 输出格式要求

       1. 输出内容必须严格按照Markdown格式，结构清晰且便于阅读。
       2. 内容必须全面，既要包含讨论的核心要点，也要通过网络搜索确保信息的准确性和全面性。
       3. 必须在单个一个章节中，列出所有创新性概念、名词以及它们的定义和解释（若有）。

    # 强制要求

       1. 使用web_search函数搜索相关的网络资料，返回有用的信息；可以执行多个网络问题的搜索，但该函数仅能被调用一次。 
       2. 网络搜索的结果必须有效，并且可以在总结中被清晰地引用。


    #知识库

      1.以下是整理过后的网络资料，可以直接引用:

        ```
          {knowledge}
        ```

    """
    user_prompt = f"""
    # 会议讨论记录，每一轮讨论的概要信息
    ```
      {json.dumps(his_nodes, ensure_ascii=False, indent=2)}
    ```
    """


    answer, reasoning, web_search_list,references = qwen_model.do_call(system_prompt, user_prompt, stream=stream)


    return answer
    

