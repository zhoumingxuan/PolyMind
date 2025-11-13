# PolyMind 多智体协作研究助手

PolyMind 是一个围绕“多智能体辩论 + 检索增强”构建的研究助手。它把复杂问题拆分为若干分目标，派发给具备不同专业背景的虚拟研究员，自动生成检索任务、整合外部资料，并最终产出结构化的会议纪要与后续行动建议。该版本聚焦本地运行，所有流程均由 `meeting.py` 协调。

## 项目亮点
- **多角色辩论驱动**：通过 `role.py` 中的角色模板批量生成研究员，自动注入人格、专业与分工，确保讨论覆盖不同视角。
- **外部检索与知识沉淀**：`search_service.web_search` 负责统一的实时检索，结果经 `meeting.r_range_knowledge` 整理为 Markdown 知识库，再回灌到讨论链路。
- **Prompt Guard 安全护栏**：系统 Prompt 内建多项合规约束，限制泄露敏感指标或主观臆测，适合在企业场景继续扩展。
- **流程可拆可扩**：各阶段被封装为纯函数，能够按需挂接流式输出、中断继续或在编排系统里运行。

## 代码结构
- `meeting.py`：调度多轮会议的主流程，提供 `start_meeting` 等接口。
- `api_model.py`：封装通义千问 Qwen 模型调用、工具调用与流式输出。
- `role.py`：角色模版与质检逻辑。
- `search_service.py`：目前仅包含 Baidu AI Search 提供方，暴露统一的 `web_search` 函数。
- `config.py`：读取 `setting.json` 的轻量封装。
- `setting.json`：运行所需的全部密钥与参数。
- `test.py`：最小可运行示例，展示如何实例化模型并触发一次会议。

## 工作流程
1. **解析任务**：`start_meeting` 接收用户问题，依据 `role_count` 创建若干专长互补的研究员。
2. **拆解检索**：模型通过工具调用生成 `question_list`，交由 `web_search` 实际检索。
3. **知识入库**：`meeting.py` 把检索结果规整成结构化知识，附带引用信息。
4. **多轮讨论**：在 `max_epcho` 轮内，研究员轮流发言、引用资料、修正观点。
5. **总结与建议**：达到终止条件或轮次上限后输出会议纪要、关键证据与后续行动计划。

## 快速开始
1. **准备环境**
   - Python 3.10+。
   - 可选：`python -m venv .venv && .\\.venv\\Scripts\\activate`
2. **安装依赖**
   ```bash
   pip install dashscope requests
   ```
3. **配置 `setting.json`**
   - 填写 Qwen（DashScope）密钥、百度千帆 AI Search 密钥。
   - 详见下文“`setting.json` 字段说明”。
4. **运行示例或集成到业务代码**
   - 直接运行 `python test.py`。
   - 或参考下列片段手动触发会议：
     ```python
     from api_model import QwenModel, AIStream
     from meeting import start_meeting

     stream = AIStream()
     model = QwenModel(model_name="qwen-plus-latest")
     report = start_meeting(model, "请围绕 XXX 制定可落地的研究计划", stream)
     print(report)
     ```

## `setting.json` 字段说明
| 键 | 示例/默认 | 说明 |
| --- | --- | --- |
| `max_epcho` | `5` | 单次会议的最大轮次，控制讨论深度与成本。 |
| `role_count` | `5` | 初始生成的研究员数量。 |
| `qwen_key` | `sk-***` | 通义千问 (DashScope) API Key，供 `api_model.QwenModel` 调用。 |
| `baidu_key` | `bce-v3/...` | 百度千帆 AI Search 的 Bearer Token，用于联网检索。 |
| `search_provider` | `baidu` | 当前版本仅支持 `baidu`，填写其它值会报错。 |
| `dashscope_search_api_key` | 可选 | 预留字段，若以后实现 DashScope Deep Search 或自定义 Provider 可复用。 |
| `dashscope_search_app_id` | 可选 | 同上，保留 AppID。 |
| `dashscope_search_app_version` | `beta` | 预留字段，兼容历史配置。 |
| `dashscope_search_endpoint` | `https://dashscope...` | 预留字段，自定义 Provider 时可复用。 |
| `search_top_k` | `6` | 百度检索返回的 `web` 结果数量上限，越大越耗时。 |
| `search_timeout` | `120`（未配置时默认 1200） | 搜索请求的超时时长（秒），可按网络状况调整。 |

> 额外可选参数：在 `setting.json` 中添加 `search_cooldown`（默认 1 秒）可以控制请求节流；`search_retry_delay`（默认 30 秒）决定失败重试的间隔。

## 搜索服务与 RAG 扩展
- `search_service.py` 现在只保留 `BaiduAISearchProvider`，保证稳定可用。`search_provider` 必须为 `baidu`，否则 `_build_provider` 会抛出 `SearchProviderError`。
- 若需要扩展为 **RAG**、企业私域检索或其它 API，可按以下思路：
  1. 新建一个继承 `SearchProviderBase` 的类，实现 `_search`，在内部调用你的向量数据库、文档库或第三方 API。
  2. 在 `_build_provider` 中注册新类，并在 `setting.json` 中新增对应的配置项。
  3. （可选）在 `meeting.r_range_knowledge` 中追加对自建知识库字段的处理，使会议阶段能引用向量检索的命中记录。
- RAG 方案示例：把查询向量化后，在 Milvus/Faiss 中召回段落，与实时的 Baidu 结果一起写入知识库，实现“现势资讯 + 私域知识”的混合增强。

通过上述方式，本仓库可以平滑演进为更复杂的检索增强生成（RAG）平台，同时保持当前轻量、易调试的特性。
