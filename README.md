# 企业知识库智能问答系统（RAG 增强版）

面向企业内网的轻量化本地 RAG 应用。支持 PDF、DOC/DOCX、Markdown、TXT 和网页导入，采用 BGE 向量检索、中文 BM25、RRF 融合与 BGE CrossEncoder 重排，最终将 Top3 资料交给 OpenAI 兼容模型生成带来源回答。

## 1. 环境要求

- Python 3.10–3.12（推荐 3.12）
- Windows、Linux 或 macOS
- 首次运行需要访问 Hugging Face 下载 `BAAI/bge-small-zh` 和 `BAAI/bge-reranker-base`
- 解析旧版 `.doc` 需要 LibreOffice；`.docx` 不需要
- 推荐 NVIDIA GPU；CPU 可以运行，但不承诺 PRD 的 5 秒检索重排指标

`requirements.txt` 使用 PyPI 通用 PyTorch 构建。在 NVIDIA 主机上如需 CUDA 加速，请按 PyTorch 官方安装器为当前驱动安装对应 CUDA wheel；应用会在 `torch.cuda.is_available()` 为真时自动切换 GPU。

扫描图片型 PDF 不做 OCR；无法提取文字时会返回明确错误。

## 2. 一键启动

Windows PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run.ps1
```

Linux/macOS：

```bash
chmod +x run.sh
./run.sh
```

浏览器访问 `http://127.0.0.1:8501`。启动脚本会创建 `.venv` 并安装精确锁定的依赖。首次模型下载不计入热态性能。

## 3. 安全配置

生成 Fernet 主密钥：

```powershell
python scripts/manage_secrets.py generate-key
$env:RAG_MASTER_KEY="上一步输出"
```

Linux/macOS 使用 `export RAG_MASTER_KEY="..."`。主密钥不得写入项目文件。

加密保存 API key：

```powershell
python scripts/manage_secrets.py set-api-key
```

配置运维口令：

```powershell
python scripts/manage_secrets.py init-admin
```

运维口令仅保存 scrypt 盐化哈希。日志自动遮盖 Authorization、API key、password 和 secret。

## 4. OpenAI 兼容模型

默认地址为 `http://localhost:11434/v1`，默认模型为 `qwen2.5:7b`。可在 `config/settings.json` 或解锁后的运维面板修改。兼容 Ollama、vLLM、LM Studio 及云端 OpenAI 兼容服务。

以 Ollama 为例：

```powershell
ollama pull qwen2.5:7b
ollama serve
```

## 5. 核心行为

- 普通文档切片默认 `1000/200`，技术文档默认 `500/100`，支持自动判断和人工覆盖。
- 向量与 BM25 各取 Top10，RRF `k=60` 融合后保留 Top10。
- `BAAI/bge-reranker-base` 过滤低分结果并输出 Top3。
- 对明确指代或短省略问句，仅拼接最近一轮用户问题；不拼 AI 回答、不调用 LLM 改写，检索输入不超过 512 字符。
- 对话默认保留最近 10 轮，清空或服务重启后销毁，不写入 Chroma。
- 同来源重新导入采用幂等替换，不会产生重复切片；单文件失败不影响批次中的其他文件。

## 6. Docker 部署

```powershell
docker compose up --build
```

端口仅绑定 `127.0.0.1`。部署到内网服务器时仍应通过防火墙或反向代理限制访问，禁止直接暴露公网。

## 7. 验证

环境诊断：

```powershell
python scripts/check_environment.py
```

下载并预热模型：

```powershell
python scripts/bootstrap_models.py
```

执行测试：

```powershell
python -m pytest
python -m ruff check .
```

功能验证流程：

1. 上传 `examples/knowledge/休假制度.txt`，确认显示成功切片数。
2. 提问“正式员工每年有几天年假？”，确认返回答案和来源。
3. 追问“它需要提前多久申请？”，确认检索查询仅使用上一轮用户问题拼接。
4. 清空对话，确认知识库切片数量不变；运维解锁后清库，确认数量归零。
5. 上传损坏文件、空 TXT、超过 20MB 文件或无效 URL，确认给出中文错误且服务不中断。

性能测试（先导入示例并启动 LLM）：

```powershell
python scripts/benchmark.py "年假需要提前多久申请？" --repeat 5
```

Recall@3 对比：

```powershell
python scripts/evaluate_retrieval.py examples/evaluation.jsonl
```

企业真实“提升 23%”必须使用带 `question` 与 `relevant_sources` 的企业标注 JSONL 数据集验证，示例集仅用于验证评测管线。

## 8. 数据位置

- Chroma：`data/chroma/`
- 脱敏轮转日志：`data/logs/rag.log`
- 非敏感配置：`config/settings.json`
- 加密密钥文件：`config/secrets.enc`

知识库数据和模型均在本地持久化；只有配置为远程 OpenAI 兼容地址时，Top3 参考资料与问题才会发送至该服务。
