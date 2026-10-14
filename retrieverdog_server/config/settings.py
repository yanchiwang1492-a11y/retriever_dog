"""
全局配置模块
============
负责加载环境变量，并通过 LlamaIndex 的 Settings 对象配置全局默认参数。

知识点对应（来自 02.LlamaIndex.md 第 10 节）：
- Settings.llm: 全局默认 LLM 模型
- Settings.embed_model: 全局默认 Embedding 模型
- Settings.chunk_size / chunk_overlap: 全局默认分块参数
- OpenAILike: 通过 OpenAI 兼容协议接入千问等第三方模型
- DashScopeEmbedding: 阿里云千问文本向量模型
"""

import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量（只在开发环境用，生产环境建议从系统环境变量读取）
load_dotenv()

# ============================================================
# 环境变量读取
# ============================================================

# 千问 API Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
# LlamaCloud API Key（PDF 解析用）
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")

# Tesseract OCR 配置
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "chi_sim+eng")

# 模型名称
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3.7-text-embedding")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen-rerank")

# 文本分块参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# 检索参数
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "15"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.0"))

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# 目录配置（相对项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")    # 上传文件暂存目录
STORAGE_DIR = os.path.join(BASE_DIR, "storage")    # ChromaDB 存储目录

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# ChromaDB collection 名称（单文档模式，固定用这个名字）
COLLECTION_NAME = "retrieverdog_collection"


# ============================================================
# LlamaIndex 全局 Settings 配置
# ============================================================
# 知识点：LlamaIndex 的 Settings 是一个全局单例，
# 所有组件（索引、检索器、查询引擎）默认都会从这里读取配置。
# 也可以在具体组件中局部覆盖（参考 02.LlamaIndex.md 10.3 节）。

from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding


def init_settings():
    """
    初始化 LlamaIndex 全局 Settings。
    在应用启动时调用一次即可。

    包含配置项：
    1. LLM 模型（千问 qwen3.7-flash，通过 OpenAI 兼容协议接入）
    2. Embedding 模型（千问文本向量，DashScope 官方封装）
    3. 文本分块参数（chunk_size / chunk_overlap）
    """

    # ---------- 配置 LLM ----------
    # 使用 OpenAILike 接入千问模型（千问 API 兼容 OpenAI 格式）
    # is_chat_model=True 表示这是一个对话模型，会使用对话接口而非补全接口
    Settings.llm = OpenAILike(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        is_chat_model=True,
        temperature=0.1,  # 越低越确定性，问答场景建议低值
    )

    # ---------- 配置 Embedding 模型 ----------
    # 使用 DashScope 官方封装的千问文本向量模型
    # embed_batch_size：批量 embedding 时每批数量，DashScope 限制最大 25
    Settings.embed_model = DashScopeEmbedding(
        model_name=EMBED_MODEL,
        api_key=DASHSCOPE_API_KEY,
        embed_batch_size=10,
    )

    # ---------- 配置分块参数 ----------
    # 这两个值会被 SentenceSplitter 等分块器作为默认值读取
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    print(f"[Settings] LLM: {LLM_MODEL}")
    print(f"[Settings] Embedding: {EMBED_MODEL}")
    print(f"[Settings] Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
