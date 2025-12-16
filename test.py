# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(__file__))

from api_model import AIStream, QwenModel
from meeting import start_meeting

# 预置几个受大众关注的课题，便于快速演示多智能体研究流程。
RESEARCH_TOPICS = {
    "a_share_sector_trend": {
        "title": "A 股未来热门板块与行情趋势研究（含配置建议）",
        "description": "研判未来 3-12 个月可能走强的板块与题材，给出配置比例与风险对冲思路。",
        "content": """
研究未来 3-12 个月可能走强的 A 股板块与题材，并形成配置建议：
- 维度：宏观与政策动向、产业链景气度、估值与资金偏好、外部事件（地缘/供需/技术周期）。
- 约束：给出板块/主题的配置比例建议，明确进场/减仓信号，提示高波动或监管风险；可提供代表性成分股示例。
- 输出：板块排序、配置建议、代表性标的示例、风险对冲思路。
""",
    },
    "two_sessions_science_advice": {
        "title": "两会前基于实时热点的科技政策建议",
        "description": "结合最新热点，面向两会提出科技/产业/科普的建议方向。",
        "content": """
围绕近期社会与产业热点，提出面向两会的科技政策与科普建议：
- 关注硬科技、民生科技、绿色低碳与安全治理等板块，结合国际对标与国内需求。
- 产出建议清单：问题点、潜在政策工具、预期效果与风险提示，可附简要评估指标。
""",
    },
    "ai_safety_theory": {
        "title": "AI 可信安全的理论与可验证框架",
        "description": "聚焦大模型与多智能体的可信与安全，梳理可验证的研究方向。",
        "content": """
面向 AI 可信安全的理论探索与验证思路：
- 关注鲁棒性、对齐、可解释、因果一致性、对抗防护等核心问题。
- 思路包括形式化规格、可验证的约束与评测协议、博弈式红蓝对抗、可复现实验基准与开放问题清单。
"""},
"ai_choose_test": {
    "title": "设计一个基于市场调研与数据分析的自动选品/铺货应用（轻量化）",
    "description": "面向汕头玩具厂的轻量化决策助手：结合网络调研与“郑和数据”，通过 Qwen API + 提示工程输出趋势研判、上架品类建议与费用测算。",
    "content": """
我需要一个 AI 应用设计方案：

- 定位：以轻量化解决为主，不会有多个用户访问问题。
- 场景：
  - 一个中国汕头玩具厂老板，想通过做一些市场调研（例如未来有哪些热门、未来有什么风险影响），以便提前几个月准备。
  - 初步想法：通过市场调研 + 数据分析得到未来趋势，再结合数据判断要上架哪些品类的玩具。
- 交互与调用方式：
  - 基本通过提示工程调用，界面就是平时的 Chat 界面，通过自然语言交互。
  - 市场调研可以来自网络调研；另外老板提供了一个“郑和数据”，我不确定它具体怎么样，若无法使用也不必关注。
- 需要交付的方案内容（请给出详细设计）：
  - 包括但不限于：前端 AI Chat 界面采用什么框架；
  - 大模型采用 Qwen；
  - 提示词调用流程：你不用写具体提示词，只需说明每一块提示词要做什么、每个步骤要调用大模型做哪些事情；
  - 包括可寻找/可接入的一些数据分析接口之类的；
  - 给出整体费用估算：包括研发成本费用和实际报价费用。
- 其他说明：
  - 脑子目前比较模糊，请积极查找网络资料。
  - 特别注意：模型更倾向于调用 API 接口，而不是本地部署;玩具销售是针对海外出口，少部分才是内销。
""",
    },
    "doc_to_react_builder": {
        "title": "从图文需求/设计文档自动生成 React 前端代码",
        "description": "doc/docx 图文需求 + 现有 React 代码，按需求生成可直接应用的 git diff。",
        "content": """
场景：需求文档为图文混合的 doc/docx，已有一套 React 前端代码。
- 提示：解析图文文档可以用大模型解析，参考Qwen API。
- 方式：采用大模型API+提示工程，但不需要具体说明某些环节提示词，只需要说明某些环节的提示词目标是什么，包括生成git diff，也可以利用提示词生成。
- ai应用逻辑：解析需求，定位需改的组件/页面/样式/接口，生成对应的 git diff 代码补丁。
        然后用户通过某些工具可查看更改，然后提交到git。
- 研究目标：输出一套关于以上要求的具体设计方案，至于提示词部分，只需要明确说明具体环节提示词要做哪些东西即可。

""",
    },
    "realtime_risk_arch": {
        "title": "金融级实时风控与事件驱动架构设计",
        "description": "面向交易、信贷、反欺诈等场景的低延迟、高可靠实时风控架构。",
        "content": """
设计兼顾低延迟、韧性与合规的实时风控架构：
- 关注事件驱动总线、特征计算与存储、模型与规则的协同、流批一体数据管道、弹性与隔离。
- 可给出参考架构、性能与合规要点、演进路线与运维监控指标。
""",
    },
}


def list_topics() -> None:
    """打印可选课题及简介。"""
    print("可选课题（键：标题 —— 简介）：")
    for key, meta in RESEARCH_TOPICS.items():
        print(f"- {key}: {meta['title']} —— {meta['description']}")


def pick_topic(args: list[str]):
    """
    根据命令行参数选择课题；支持:
    - `python test.py`           直接运行默认课题
    - `python test.py list`      列出全部课题
    - `python test.py <key>`     运行指定课题
    """
    if len(args) >= 2:
        key = args[1].strip()
        if key.lower() == "list":
            list_topics()
            sys.exit(0)
        if key not in RESEARCH_TOPICS:
            print(f"未找到课题 '{key}'，可选：{', '.join(RESEARCH_TOPICS.keys())}")
            sys.exit(1)
        return key, RESEARCH_TOPICS[key]

    default_key = next(iter(RESEARCH_TOPICS))
    return default_key, RESEARCH_TOPICS[default_key]


def main():
    topic_key, topic_meta = pick_topic(sys.argv)
    print(f"\n即将启动课题：{topic_meta['title']} ({topic_key})\n{topic_meta['description']}\n")

    stream = AIStream()
    qwen_model = QwenModel(model_name="qwen3-max")

    result_content = start_meeting(qwen_model, "自动驾驶世界模型合成navsim数据方案", stream)
    print("\n\n====最终研究报告====\n\n", result_content)


if __name__ == "__main__":
    main()
