# PolyMind 多智能体协作讨论

> 全仓库文件与输出均使用 UTF-8 编码，请保持编辑器/终端为 UTF-8。  
> 依赖网络访问 DashScope 与百度 AI Search。

## 项目概述
- 多智能体深度研讨流程：拆解需求 → 补齐背景知识 → 多轮角色讨论 → 收敛方案/报告。
- 内置检索工具：调用百度 AI Search（v2/ai_search/web_search），按权威度去重、按需渲染网页正文，聚合为结构化 JSON。
- 文档解析链路：自动识别结果中的 PDF/DOC/DOCX 链接，优先用 `qwen-doc-turbo`，必要时回退本地解析（pdfplumber/python-docx）并由 `qwen-long` 总结。
- 入口示例位于 `test.py`，`meeting.py` 负责多轮讨论编排，`search_service.py`/`search_anylyze.py` 负责检索与文档处理。

## 适配场景 / 解决痛点
- **热点研判与政策建议**：快速围绕宏观/产业/政策热点生成多视角问题集，结合实时搜索产出可追溯的研判与建议。
- **行业/技术方案对比**：对架构方案、技术路线或行业实践进行多轮对比、证据补强和收敛，避免“一言堂”或信息遗漏。
- **资料快速熟悉**：自动抓取并总结网页和 PDF/DOC/DOCX，按页/块标注来源，解决长文档读不动、引用不可追溯的问题。
- **风险与约束澄清**：在研讨中强制列出时间/地点/主体/前提等边界条件，减少决策中的假设遗漏与风险盲区。
- **多话题批量起步**：通过预置/自定义 `RESEARCH_TOPICS`，快速起步不同主题，统一流程、统一搜索与总结规范。

## 工作流概览（对应代码）
1) **课题选择**：`test.py` 通过 `pick_topic` 从 `RESEARCH_TOPICS` 选题；主流程调用 `start_meeting`。  
2) **问题生成与基础知识构建**：`knowledge.create_webquestion_from_user` 结合当前日期生成检索问题组并调用 `web_search`；结果经 `rrange_knowledge` 清洗、去行动导向、去推荐类内容。  
3) **初始方案**：`meeting.create_initial_solution` 产出首版方案框架（仅理论层面的结构）。  
4) **角色设定**：`knowledge.create_roles` + `meeting` 生成多角色（数量由 `role_count` 控制），并在每轮按序发言。  
5) **多轮讨论与收敛**：`role.role_talk`/`role_dissucess` 在每轮附带搜索工具调用；`meeting` 中按轮次调用 `summarize_and_consolidate_solutions`（收敛为双方案）、`summarize_and_select_final_plan`（确定唯一方案）、`generate_report_from_plan`/`refine_report`（迭代报告），并用 `evaluate_discussion_status` 判断是否提前终止（受 `max_epcho` 控制）。  
6) **搜索与文档解析链路**：`search_service.web_search` 统一入口，具备缓存、冷却与时间范围；`search_anylyze` 识别 PDF/DOC/DOCX，先 `qwen-doc-turbo`，再必要时本地下载 + `pdfplumber`/`python-docx` + `qwen-long` 回退。抓取正文使用 Playwright 优先，失败回退 `requests`，并进行 class/id/敏感词过滤、编码探测。  
7) **输出与存储**：下载文件落在 `download/`；搜索结果结构化为含来源/时间的 JSON 供大模型引用；整体输出均限制为理论研究内容。

## 流程细节与约束
- **研究轮次**：默认 `max_epcho=5`；第 1 轮侧重摸清方向、生成两条候选方案；第 2 轮对比核验并收敛到唯一方案、生成首版报告；后续轮次仅在既定方案内精修报告并判定是否停机。  
- **角色发言规则**：`role.py` 根据轮次切换约束：首轮强调“摸清+对比+检索”，次轮强调“核验+收敛”，3+ 轮仅允许在既定方案内细化/补强报告，禁止引入新方案或行动导向。所有发言强制第一人称。  
- **停机判断**：`evaluate_discussion_status` 同时检查最终方案、当前报告与当轮讨论，若关键维度无缺口且无高影响争议才可停，反之继续；确保不会过早停止。  
- **检索使用原则**：每轮可多次检索，问题需避免语义重叠，尽量附带时间/地域/主体限制；引用必须标注来源/时间，缺乏支撑的点宁缺勿滥。  
- **知识清洗**：`rrange_knowledge` 会删除推荐/行动/方案类表述，仅保留理解需求所需的定义与背景，输出仍为 Markdown 但已做筛减。

## 检索与文档解析细节
- **去重与排序**：`search_service.web_search` 按 `authority_score`、`rerank_score` 排序去重，取前 10 结果。  
- **正文抓取**：优先 Playwright 渲染 `<body>`，失败回退 `requests`；显式探测 charset（含 meta/apparent_encoding），统一中文为 `gb18030`/`utf-8`。清洗阶段会删除脚本/样式/导航/注释/敏感块，保留主体 HTML。  
- **文档识别与回退**：先 HEAD/GET 判断 content-type 或 URL/Content-Disposition 后缀；可达性检查后，优先 `qwen-doc-turbo` 批量解析；若 400/不支持则下载到本地：PDF 用 `pdfplumber` 逐页，DOCX 用 `python-docx` 按块，DOC 依赖 `antiword`/`catdoc`/`soffice` 转换，再由 `qwen-long` 细致总结并标注来源页/块。若仍失败，回退常规页面抓取。  
- **缓存与节流**：内存缓存 + 冷却时间（`search_cooldown`），异常重试（`search_retry_delay`）；抓取超时由 `search_fetch_timeout` 控制。  
- **问题生成**：`knowledge.create_webquestion_from_user` 生成 1-10 条检索问题，避免模糊/重复，支持时间范围（none/week/month/semiyear/year）。

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
- ai_choose_test —— 轻量化自动选品/铺货应用设计
- doc_to_react_builder —— 从图文需求生成 React 代码的方案设计
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
