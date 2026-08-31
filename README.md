# RAG

轻量化企业知识库智能问答系统，基于检索增强生成（RAG）实现本地文档与网页知识的导入、检索和多轮对话问答。系统将混合检索与重排结果交由 OpenAI 兼容的大语言模型生成带来源引用的答案。

## ✨ 特性亮点

- 支持导入 PDF、DOC、DOCX、Markdown、TXT 文件及网页内容
- 采用 BGE 向量检索与中文 BM25 关键词检索
- 通过 RRF（Reciprocal Rank Fusion）融合多路检索结果
- 使用 BGE CrossEncoder 重排候选内容，提升回答相关性
- 自动区分普通文档与技术文档，采用差异化切片策略
- 支持多轮对话、上下文关联与指代性问题检索
- 支持 Ollama、vLLM、LM Studio、DeepSeek 等 OpenAI 兼容模型服务
- 本地持久化 Chroma 向量数据库与 BM25 索引
- 提供 API Key 加密存储、运维密码哈希与敏感日志脱敏能力
- 内置环境检查、模型预热、性能测试和检索评测脚本
- 支持 Docker 容器化部署

## 🛠️ 技术栈

| 分类 | 技术 |
| --- | --- |
| 编程语言 | Python 3.10–3.12 |
| Web 界面 | Streamlit |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 向量模型 | `BAAI/bge-small-zh` |
| 重排模型 | `BAAI/bge-reranker-base` |
| 关键词检索 | BM25、jieba、rank-bm25 |
| 文档解析 | PyPDF、python-docx、Beautiful Soup |
| 模型接入 | OpenAI 兼容 API、LangChain OpenAI |
| 安全能力 | cryptography、Fernet、scrypt |
| 容器化 | Docker、Docker Compose |
| 测试与检查 | pytest、Ruff |

## 🚀 快速开始

### 环境准备

请先安装以下软件：

- Python 3.10–3.12，推荐 Python 3.12
- Git
- 可选：Docker Desktop
- 可选：LibreOffice，用于解析旧版 `.doc` 文件
- 可选：Ollama 或其他 OpenAI 兼容模型服务

首次运行会下载嵌入模型与重排模型，请保持网络可用。

### 克隆项目

```bash
git clone https://github.com/your-username/RAG.git
cd RAG
```

### Windows 启动

在 PowerShell 中执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run.ps1
```

脚本会自动创建虚拟环境、安装依赖并启动应用。

### Linux / macOS 启动

```bash
chmod +x run.sh
./run.sh
```

启动成功后，在浏览器中打开：

```text
http://127.0.0.1:8501
```

### 手动安装与运行

创建虚拟环境：

```bash
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

### 配置大语言模型

复制示例配置文件：

```powershell
Copy-Item config\settings.example.json config\settings.json
```

```bash
cp config/settings.example.json config/settings.json
```

编辑 `config/settings.json` 中的 `llm_base_url` 与 `llm_model`，即可接入 OpenAI 兼容服务。

以 Ollama 为例：

```bash
ollama pull qwen2.5:7b
ollama serve
```

对应配置如下：

```json
{
  "llm_base_url": "http://localhost:11434/v1",
  "llm_model": "qwen2.5:7b"
}
```

### 配置安全密钥

生成主密钥：

```powershell
python scripts/manage_secrets.py generate-key
$env:RAG_MASTER_KEY="生成的主密钥"
```

Linux / macOS：

```bash
python scripts/manage_secrets.py generate-key
export RAG_MASTER_KEY="生成的主密钥"
```

加密保存模型服务 API Key：

```bash
python scripts/manage_secrets.py set-api-key
```

初始化运维密码：

```bash
python scripts/manage_secrets.py init-admin
```

### Docker 部署

```bash
docker compose up --build
```

服务默认仅绑定本机 `127.0.0.1:8501`。

## 📁 项目目录结构

```text
RAG/
├── app.py                          # Streamlit 应用入口
├── pyproject.toml                  # 项目与工具配置
├── requirements.txt                # 生产依赖
├── requirements-dev.txt            # 开发依赖
├── run.ps1                         # Windows 启动脚本
├── run.sh                          # Linux/macOS 启动脚本
├── Dockerfile                      # Docker 镜像定义
├── docker-compose.yml              # Docker Compose 配置
├── config/
│   └── settings.example.json       # 示例配置文件
├── data/
│   ├── chroma/                     # Chroma 向量数据库数据
│   └── logs/                       # 脱敏日志
├── examples/
│   ├── knowledge/                  # 示例知识库文件
│   └── evaluation.jsonl            # 检索评测示例数据
├── scripts/
│   ├── bootstrap_models.py         # 模型下载与预热
│   ├── check_environment.py        # 环境诊断
│   ├── evaluate_retrieval.py       # 检索效果评测
│   ├── benchmark.py                # 性能测试
│   └── manage_secrets.py           # 安全凭据管理
├── src/
│   └── rag_app/
│       ├── chat/                   # 对话、提示词、记忆与 LLM 调用
│       ├── chunking/               # 文档分类与分块
│       ├── indexing/               # Chroma、BM25 索引与入库服务
│       ├── parsing/                # PDF、Word、文本与网页解析
│       ├── retrieval/              # 混合检索、RRF、重排与查询处理
│       ├── security/               # 管理认证与加密存储
│       ├── ui/                     # Streamlit 页面与交互组件
│       ├── config.py               # 配置管理
│       ├── runtime.py              # 运行时依赖初始化
│       └── models.py               # 数据模型
└── tests/
    ├── unit/                       # 单元测试
    ├── integration/                # 集成测试
    └── ui/                         # 界面测试
```

## 📌 主要功能说明

### 知识库导入

支持导入本地 PDF、DOC、DOCX、Markdown、TXT 文件，以及网页 URL 内容。系统会自动清洗文本、识别文档类型并完成切片与索引构建。

- 普通文档默认切片参数：`1000 / 200`
- 技术文档默认切片参数：`500 / 100`
- 支持自动识别文档类型与人工覆盖
- 相同来源重复导入时自动替换，避免生成重复切片
- 单个文件导入失败不会中断整个批次

### 混合检索与重排

系统采用多阶段检索流程：

1. 使用 BGE 向量模型进行语义检索，获取 Top 10 候选内容。
2. 使用 BM25 进行关键词检索，获取 Top 10 候选内容。
3. 使用 RRF 算法融合两路结果，默认参数为 `k = 60`。
4. 使用 `BAAI/bge-reranker-base` 对候选结果重排。
5. 过滤低相关内容后，将 Top 3 参考资料提供给大语言模型生成答案。

### 多轮问答

系统支持会话记忆与上下文关联：

- 默认保留最近 10 轮对话
- 对短问句或指代性问题，自动结合上一轮用户问题优化检索
- 不将模型回答拼接进检索查询，降低错误信息扩散风险
- 会话记忆仅保留在运行时，清空对话或重启服务后自动销毁

### 来源引用

每次回答会附带关联知识来源，便于用户核验答案依据、追溯文档内容并提升问答可信度。

### 安全与运维

- API Key 采用 Fernet 加密存储
- 运维密码使用 scrypt 哈希保存
- 日志自动遮盖 Authorization、API Key、password、secret 等敏感信息
- 支持通过运维面板管理系统配置和知识库数据

### 验证与评测

执行环境诊断：

```bash
python scripts/check_environment.py
```

下载并预热模型：

```bash
python scripts/bootstrap_models.py
```

运行测试与代码检查：

```bash
python -m pytest
python -m ruff check .
```

执行性能测试：

```bash
python scripts/benchmark.py "年假需要提前多久申请？" --repeat 5
```

执行检索评测：

```bash
python scripts/evaluate_retrieval.py examples/evaluation.jsonl
```

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

你可以自由使用、修改和分发本项目，但请保留原始版权与许可声明。

如果这个项目对你有帮助，欢迎点一个 Star ⭐ 支持项目持续完善！
