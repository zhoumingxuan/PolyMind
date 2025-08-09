import os, requests, json
from datetime import datetime
import time
from config import ConfigHelper

config = ConfigHelper()

cache_data=[]

APPBUILDER_KEY=config.get("baidu_key", None)

url = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
headers = {
    "Authorization": f"Bearer {APPBUILDER_KEY}",
    "Content-Type": "application/json"
}

#当前日期
date_str=datetime.today().strftime('%Y年%m月%d日')

def web_search(search_message,search_recency_filter=None):
   try:
      time.sleep(10)
      return inner_web_search(search_message,search_recency_filter=search_recency_filter)
   except Exception as e:
      time.sleep(180)
      return inner_web_search(search_message,search_recency_filter=search_recency_filter)



def inner_web_search(search_message,search_recency_filter=None):
 
 for record in cache_data:
    if record.get("search_key") == search_message:
        return record.get("content")
 
 key=f"{search_message}\n\n{search_recency_filter}\n\n"

 body = {
    "messages": [
     {"role": "user", "content": search_message}
    ],
    "search_source": "baidu_search_v2",
    "search_mode": "required",
    "temperature":0.5,
    "instruction": """
# 搜索规范和要求（特别严格的强制性要求）

  1. 网络搜索结果必须完全围绕搜索问题展开，不能涉及任何无关内容。

  2. 结果中每个细节必须准确地呈现搜索到的信息，禁止使用概括性描述和泛泛而谈，确保描述精确且完整。

  3. 对搜索结果中的信息严禁篡改、虚构或夸大。所有信息必须忠实于搜索到的网络资料。

  4. 严禁给出任何结论、建议或计划，严格只进行归纳整理，不添加任何分析或推测。

  5. 每条信息后面必须明确标明来源，包括网站名和具体文章的链接。来源格式如下：
     - 来源：网站名称，文章标题，链接（网址）

  6. 不能虚构来源，确保每个来源都是真实可靠的，且能够追溯到具体的网页或文章。

  7. 涉及含有数据、统计信息、行业规范、法律法规等内容，必须明确标注时间信息，具体的时间点或具体的时间范围。

  8. （特别强调）若是搜索结果为空，且模型自带的知识库中有相关信息，则来源必须标注为"大模型生成,存在幻觉请谨慎使用"和引用知识具体的时间。

  9. （特别强调）若是搜索结果为空，且模型自带的知识库中也没有相关信息，则必须明确标注"搜索结果为空"。

# 输出要求
- 输出内容应保持简洁，确保信息无遗漏，并且准确反映网络搜索结果中的关键信息。
- 每一条总结必须附带来源标注，格式统一，并包含准确的网页链接。
- 禁止在输出中加入任何主观分析或不具备明确来源的内容。


# 输出要求
  
  - 输出内容应保持简洁，确保信息无遗漏，并且准确反映网络搜索结果中的关键信息。
  - 禁止在输出中加入任何主观分析或不具备明确来源的内容。

    """,
    "response_format":"text",
    "enable_reasoning":True,
    "enable_corner_markers":False,
    "resource_type_filter": [{"type": "image","top_k": 5},{"type": "web", "top_k": 10}],
    # 可选：指定摘要模型
    "model": "DeepSeek-R1",
    "stream": False          # 流式返回设 True
 }
 
 if search_recency_filter and search_recency_filter in ['year','semiyear','week','month'] and search_recency_filter!="":
    body['search_recency_filter']=search_recency_filter
 
 resp = requests.post(url, headers=headers, json=body, timeout=1200).json()


 while not resp.get("choices"):
    print("Error:",resp)
    time.sleep(180)
    resp = requests.post(url, headers=headers, json=body, timeout=1200).json()

 messages=[]

 references=[]

 references.extend(resp.get("references", []))

 for item in resp["choices"]:
    messages.append(item["message"]["content"])

 data="".join(messages)

 cache_data.append({
    'search_key':key,
    'content':data
 })

 return data,references
