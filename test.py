# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(__file__))

from api_model import AIStream, QwenModel
from meeting import start_meeting

# 预置几个受大众关注的课题（A 股选股/板块趋势、系统架构），便于快速演示多智能体研究流程。
RESEARCH_TOPICS = {
    "a_share_stock_pick": {
        "title": "A 股多因子选股与推荐（含投资建议）",
        "description": "结合行业景气、财报质量、量价与事件/舆情信号，形成可解释的股票推荐与配置建议。",
        "content": """
面向中国 A 股（沪深市场）的多因子选股与推荐：
- 目标：基于行业景气度、财报与现金流质量、量价动量、公告/新闻/研报舆情，形成可解释的股票候选清单与仓位建议。
- 约束：覆盖大盘/中盘/小盘与流动性要求，控制单票/行业/风格暴露；需给出逻辑、数据来源、风险提示与止盈止损思路。
- 输出：列出推荐股票及理由，给出仓位建议和风险场景（可包含备用观察名单）。
""",
    },
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
    "ecommerce_arch": {
        "title": "大型电商高并发架构演进方案",
        "description": "面向大促/秒杀场景的高并发、高可用、弹性与降本架构设计。",
        "content": """
为大型电商平台设计高并发架构演进方案：
- 场景：大促/秒杀、库存扣减、订单支付、风控、推荐/搜索的稳定性与延迟控制。
- 约束：异地多活/容灾、降级限流策略、缓存与消息削峰、灰度与回滚；兼顾成本与可观测性。
- 指标：峰值 QPS、尾延迟、可用性 SLA、故障恢复时间、资源成本与容量预测。
""",
    },
    "observability_platform": {
        "title": "全链路可观测性与 SLO 平台方案",
        "description": "构建日志/指标/链路追踪统一的可观测性与 SLO 治理平台。",
        "content": """
设计面向中大型系统的可观测性平台：
- 需求：日志/指标/链路追踪统一采集与关联，SLO/错误预算治理，异常检测与告警降噪。
- 约束：多云/混合部署、数据留存与成本优化、敏感数据脱敏与合规；支持多语言与多运行时。
- 指标：监控覆盖度、告警噪音率、定位耗时、SLO 达成率、存储/计算成本。
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
