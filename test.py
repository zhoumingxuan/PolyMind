
from api_model import QwenModel,AIStream
from meeting import start_meeting

que1="""
   设计一个网络上没有的,预期用户群体较高的，且具有很高价值的MCP服务。
"""

que2="""
设计一个网络上没有的,预期用户群体较高的，且具有很高价值的智能体。
"""

que3="""
     1.谈谈你对中国A股股市，未来走势的看法，并说明具体的理由；并且需要判断当前股市是否处于牛市，未来是否存在大涨的可能。

     2.推荐一些未来预期可能会热炒的概念板块。

     3.排除当前已涨幅过高且可能存在大跌的概念板块。

     4.推荐一些未来可能会大涨的股票，同时必须注意需要严格判断，推荐股票不得在未来存在大跌可能。

     5.必须严格排除当前涨幅过高且存在回调压力的股票。

     6.只看（上证/深证/创业板），严格禁止看科创版股票（68开头）。

     7.需严格注意大盘股，中盘股，小盘股，牢记小盘股除非概念热炒，否则可能存在人气不足的风险。

"""

stream = AIStream()

qwen_model = QwenModel(model_name="qwen-plus-latest")

result_content = start_meeting(qwen_model,que3,stream)