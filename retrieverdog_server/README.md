# RetrieverDog 后端

基于 LlamaIndex + FastAPI + ChromaDB 的文档检索问答系统后端。

## 功能特性

- 支持多种文档格式：PDF、Word、Excel、PPT、图片（OCR）、纯文本/代码
- PDF 使用 LlamaParse 解析，效果远超本地工具
- 图片使用 Tesseract OCR 识别文字
- 混合检索：向量检索 + BM25 关键词检索，RRF 融合
- 两阶段精排：qwen-rerank 云端重排序
- 流式输出：SSE 逐字推送，打字机效果
- 极简架构：单文档模式，代码简单清晰

## 涵盖的 LlamaIndex 知识点

| 知识点 | 文件 | 代码位置 |
|--------|------|----------|
| Settings 全局配置 | 02.LlamaIndex.md | `config/settings.py` |
| Document / Node 概念 | 02.LlamaIndex.md | `services/document_loader.py` |
| SentenceSplitter 分块 | 02.LlamaIndex.md | `services/index_manager.py` |
| SimpleDirectoryReader | 02.LlamaIndex.md | `services/document_loader.py` |
| VectorStoreIndex 向量索引 | 02.LlamaIndex.md | `services/index_manager.py` |
| as_retriever 检索器 | 02.LlamaIndex.md | `services/query_service.py` |
| as_query_engine 查询引擎 | 02.LlamaIndex.md | `services/query_service.py` |
| StorageContext 持久化 | 02.LlamaIndex.md | `services/index_manager.py` |
| ChromaDB 向量数据库 | 03.LlamaIndex-RAG.md | `services/index_manager.py` |
| LlamaParse PDF 解析 | 03.LlamaIndex-RAG.md | `services/document_loader.py` |
| ChatPromptTemplate 自定义 Prompt | 两份都有 | `services/query_service.py` |
| partial_format 动态赋值 | 03.LlamaIndex-RAG.md | （模板预留，单轮不需要） |
| SimilarityPostprocessor 后处理 | 02.LlamaIndex.md | `services/query_service.py` |
| 流式输出 streaming | 02.LlamaIndex.md | `services/query_service.py` + `main.py` |
| Reranker 重排序 | 03.LlamaIndex-RAG.md | `services/query_service.py` |
| BM25Retriever 关键词检索 | 03.LlamaIndex-RAG.md | `services/query_service.py` |
| QueryFusionRetriever 混合检索 | 03.LlamaIndex-RAG.md | `services/query_service.py` |
| source_nodes 查看来源 | 两份都有 | `services/query_service.py` |
| 三种相似度算法说明 | 02.LlamaIndex.md | `services/index_manager.py` 注释 |

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
cd retrieverdog_server
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，填入你的 API Key：

- `DASHSCOPE_API_KEY`：千问 API Key（必填）
- `LLAMA_CLOUD_API_KEY`：LlamaCloud Key（PDF 解析用，已填好）
- `TESSERACT_CMD`：Tesseract OCR 路径（图片识别用）

### 3. 启动服务

```bash
python main.py
```

或使用 uvicorn 命令：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：http://localhost:8000/docs  （自动生成的 API 文档）

## API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/upload` | 上传文档，构建索引 |
| POST | `/api/query/stream` | 流式问答 |
| DELETE | `/api/document` | 删除当前文档 |
| GET | `/api/health` | 健康检查 |

## 目录结构

```
retrieverdog_server/
├── main.py                  # FastAPI 入口
├── requirements.txt         # 依赖清单
├── .env                     # 环境变量
├── config/
│   └── settings.py          # 全局配置（LLM/Embedding/分块）
├── services/
│   ├── document_loader.py   # 多格式文档加载
│   ├── index_manager.py     # 索引构建 + ChromaDB
│   └── query_service.py     # 混合检索 + 重排序 + 流式生成
├── models/
│   └── schemas.py           # Pydantic 数据模型
├── storage/                 # ChromaDB 数据（运行时生成）
└── uploads/                 # 上传文件（运行时生成）
```

## RAG 流水线

```
用户上传文档
    │
    ▼
document_loader.py（按格式选择解析器）
    │
    ▼
index_manager.py（SentenceSplitter 分块 → ChromaDB 存向量）
    │
    ▼
用户提问
    │
    ▼
query_service.py
    ├─ 阶段1：混合检索（向量 + BM25，RRF 融合）  Top-K=15
    ├─ 阶段2：重排序（qwen-rerank）             Top-N=3
    └─ 阶段3：LLM 生成（自定义 Prompt + 流式输出）
```
