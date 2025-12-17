# PolyMind 多智能体协作讨论

> 全仓库文件与输出均使用 UTF-8 编码，请保持编辑器/终端为 UTF-8。  
> 依赖网络访问 DashScope 与百度 AI Search。

## 项目概述
- 多智能体深度研讨流程：拆解需求 → 补齐背景知识 → 多轮角色讨论 → 收敛方案/报告。
- 内置检索工具：调用百度 AI Search（v2/ai_search/web_search），按权威度去重、按需渲染网页正文，聚合为结构化 JSON。
- 文档解析链路：自动识别结果中的 PDF/DOC/DOCX 链接，优先用 `qwen-doc-turbo`，必要时回退本地解析（pdfplumber/python-docx）并由 `qwen-long` 总结。
- 入口示例位于 `test.py`，`meeting.py` 负责多轮讨论编排，`search_service.py`/`search_anylyze.py` 负责检索与文档处理。

## 目录速览
- `test.py`：示例入口（默认模型 `qwen3-max`）。
- `meeting.py`：多轮讨论/收敛主流程。
- `knowledge.py`：根据用户需求生成检索问题、整理基础知识。
- `api_model.py`：DashScope 封装，含工具调用与重试逻辑。
- `search_service.py`：百度搜索、网页抓取与正文清洗（可选 Playwright + BeautifulSoup）。
- `search_anylyze.py`：文档 URL 过滤、下载、解析与大模型总结。
- `role.py`：角色生成与逐轮发言规则。
- `config.py`：读取 `setting.json`。
- `setting.json`：示例配置（UTF-8）。

## 环境要求
- Python 3.10+
- 可访问互联网（DashScope、百度 AI Search）。
- DashScope API Key：`qwen_key`（用于 `qwen3-max` / `qwen-long` / `qwen-doc-turbo`）。
- 百度 AI Search Key：`baidu_key`（/v2/ai_search/web_search）。
- 可选：`antiword` / `catdoc` / `libreoffice`（soffice）用于 `.doc` 回退解析。

## 安装依赖
必装 Python 包（避免编码问题，请确保终端为 UTF-8）：
```bash
pip install -U pip
pip install dashscope requests pdfplumber python-docx
```

推荐/可选增强：
```bash
pip install beautifulsoup4 playwright   # 网页清洗与浏览器渲染
python -m playwright install chromium   # 首次安装 Playwright 需拉取浏览器
```
- 如需更快的 HTML 解析，可额外 `pip install lxml`（BeautifulSoup 会自动使用）。
- `.doc` 回退解析需系统工具：安装 `antiword` 或 `catdoc`，或安装 LibreOffice 并确保 `soffice` 可用。

## 配置
编辑 `setting.json`（保持 UTF-8）：
- `qwen_key`：DashScope Key。
- `baidu_key`：百度 AI Search Key。
- `max_epcho`：讨论最大轮次（默认 5）。
- `role_count`：研讨角色数量（默认 5）。
- `search_provider`：检索提供方，目前支持 `baidu`。
- `search_top_k`：单次搜索返回条数（1-50，默认 10）。
- `search_fetch_timeout`：网页抓取/渲染超时秒数。
- 其他字段：`search_cooldown`、`search_retry_delay`、`search_timeout` 等可按需在配置中追加。

## 示例课题与快速运行
```bash
# 查看课题列表
python test.py list

# 运行默认课题（当前为 a_share_sector_trend）
python test.py

# 指定课题
python test.py <key>
```
- `start_meeting` 的示例需求文案写在 `test.py` 中，可替换为 `RESEARCH_TOPICS[key]["content"]` 或任意自定义文本。
- 文档解析产生的文件会写入 `download/` 目录（自动创建）。

## 当前内置课题
- a_share_sector_trend —— A 股未来热门板块与行情趋势（含配置建议）
- two_sessions_science_advice —— 两会热点驱动的科技政策建议
- ai_safety_theory —— AI 可信安全的理论与可验证框架
- realtime_risk_arch —— 金融级实时风控与事件驱动架构设计

## 手动添加课题
1) 打开 `test.py`，在 `RESEARCH_TOPICS` 字典内新增一条，结构与现有示例一致（`title`、`description`、`content`）。  
2) 保持文件编码 UTF-8，`content` 用多行字符串 `"""..."""` 写明目标、约束、输出要求等。  
3) 若要设为默认运行，将新课题放在字典的第一个位置（默认取 `next(iter(RESEARCH_TOPICS))`）。

## 注意
- 仅内置 Baidu AI Search，请确保网络通畅并配置有效 Token。
- 所有输出为理论研究内容，Prompt 已限制行动导向与来源伪造。
- 当前搜索链路为自研实现，测试未覆盖全部场景，可能出现报错或异常，使用时请留意并反馈。
- 全部文件与输出均使用 UTF-8，避免 ANSI/GBK 等编码导致乱码。
- Playwright 未安装时会回退纯 `requests` 抓取，正文还原度可能下降；`.doc` 回退依赖 `antiword`/`catdoc`/`soffice`，缺失则无法处理该类文档。

### 搜索实现特色（自研管线）
- 默认调用 Baidu AI Search（/v2/ai_search/web_search）获取 top 结果，按权威度/重排得分排序去重。
- 每条结果优先用 Playwright 渲染提取 `body`，失败则回退 `requests` + 编码探测；清理脚本/样式/无关块后保留主体 HTML。
- 自动识别 PDF/DOC/DOCX 链接：先做可达性检查，再尝试 `qwen-doc-turbo` 批量解析；若不支持则下载到本地用 pdfplumber/python-docx 解析，并交给 `qwen-long` 细致总结；仍失败则回退常规页面抓取。
- 最终汇总为包含 title/publish_time/source/snippet/web_content/url 的 JSON，方便大模型整合并保证来源与时间可追溯。
