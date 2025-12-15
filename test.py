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

    result_content = start_meeting(qwen_model, topic_meta["content"], stream)
    print("\n\n====最终研究报告====\n\n", result_content)


if __name__ == "__main__":
    main()
