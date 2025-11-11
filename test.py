
from api_model import QwenModel,AIStream
from meeting import start_meeting

que1="""
   设计一个网络上没有的,预期用户群体较高的，且具有很高价值的MCP服务。
"""

que2="""
不需要看注释，从数学角度评价一下设计的优缺点，并给出适当的建议，输入数据是多个特征的线性数据，例如(1,360,5)，带有正负和幅度。
另外，网络中不可能有此代码，故此，请用相关术语执行网络搜索，然后结合网络资料和代码进行深度研究。
小提示，如下代码，K和V为解耦状态，训练时确实存在V自由度过大，导致容易过拟合，但是此层本质表达是将线性数据转为离散编码，故此如何权衡是需要考虑的一个方向。
还是那句话，要搜索网络资料请用相关专业数据，网络中无此代码，请勿有跟当前代码相关类似的网络搜索问题。
``` python
class ClassEmbeddingAttention(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 class_size: int,
                 topk: int,
                 m_heads: int = 20,
                 eps: float = 1e-6):
        super().__init__()

        assert hidden_size % m_heads == 0, "hidden_size 必须能被 m_heads 整除"
        assert topk >= 1, "topk 必须 ≥ 1"
        assert topk < class_size, "topk 必须小于 class_size"

        self.hidden_size = hidden_size
        self.m_heads = m_heads
        self.head_dim = hidden_size // m_heads
        self.class_size = class_size
        self.topk = topk
        self.eps = eps

        self.cls_type_k = nn.Parameter(torch.empty(class_size, hidden_size))  # [m, H]
        nn.init.orthogonal_(self.cls_type_k)
        
        self.cls_type_v = nn.Parameter(torch.empty(class_size, hidden_size))  # [m, H]
        nn.init.orthogonal_(self.cls_type_v)


        self.q_proj =nn.Linear(input_size, hidden_size,bias=False)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def rms_like(self, x: torch.Tensor) -> torch.Tensor:
       x_f32 = x.float()
       with torch.no_grad():
           rms=torch.square(x_f32).mean(dim=-1,keepdim=True).sqrt().clamp_min(self.eps)

       return x_f32/rms

    def attn(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, Tq, n, _ = Q.size()
        _, Tk, m, _ = K.size()
        out_dtype = Q.dtype  

        Qh = Q.reshape(B, Tq, n, self.m_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B,h,T,n,d]
        Kh = K.reshape(B, Tk, m, self.m_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B,h,T,m,d]
        Vh = V.reshape(B, Tk, m, self.m_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B,h,T,m,d]

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / scale 

        with torch.no_grad():
          top_idx = torch.topk(scores, k=self.topk, dim=-1).indices 


        scores_k = torch.gather(scores, dim=-1, index=top_idx) 
        attn_weights = F.softmax(scores_k.float(), dim=-1)

        
        del scores


        top_idx = top_idx.squeeze(-2)
        top_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)   
        V_top = torch.gather(Vh, dim=-2, index=top_idx).float()                

        attn_output = torch.sum(V_top * attn_weights.squeeze(-2).unsqueeze(-1), dim=-2).to(out_dtype) 

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, Tq, self.hidden_size)
        return attn_output

    def forward(self, input_states: torch.Tensor) -> torch.Tensor:

        B, T, _ = input_states.size()

        Q = self.q_proj(input_states).unsqueeze(-2)

        K = self.cls_type_k.unsqueeze(0).unsqueeze(0).expand(B, T, self.class_size, self.hidden_size)
        V = self.cls_type_v.unsqueeze(0).unsqueeze(0).expand(B, T, self.class_size, self.hidden_size)

        with torch.no_grad():
          K_mean=torch.mean(K,dim=-2,keepdim=True)
          
        K=K-K_mean
        K=self.rms_like(K)


        V=self.rms_like(V)

        with torch.no_grad():
           V_mean=torch.mean(V,dim=-2,keepdim=True)
        
        V=V-V_mean

        attn_output = self.attn(Q, K, V)

        attn_output = self.c_proj(attn_output)
        
        return attn_output
```
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

result_content = start_meeting(qwen_model,que2,stream)