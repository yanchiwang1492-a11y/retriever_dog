"""
索引管理服务
==============
负责文档的分块、索引构建、以及 ChromaDB 向量数据库的持久化管理。

核心流程：
Documents → SentenceSplitter 分块 → Nodes → VectorStoreIndex → ChromaDB 持久化

知识点对应：
- 02.LlamaIndex.md 第 3 节：Document / Node 概念
- 02.LlamaIndex.md 第 6 节：SentenceSplitter 文本分块
- 02.LlamaIndex.md 第 7 节：VectorStoreIndex + 索引持久化
- 02.LlamaIndex.md 第 3.4 节：三种相似度算法（cosine/dot_product/euclidean）
- 03.LlamaIndex-RAG.md 第 3 节：ChromaDB 外部向量数据库
- 03.LlamaIndex-RAG.md 第 2 节：MarkdownElementNodeParser（PDF 解析后用）
"""

import os
import shutil
from typing import List, Optional

import chromadb
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from config.settings import (
    STORAGE_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class IndexManager:
    """
    索引管理器：封装索引的构建、加载、删除等操作。

    单文档模式：系统中始终只有一个 collection 的索引，
    上传新文档时会删除旧的索引再构建新的。
    """

    def __init__(self):
        """初始化 ChromaDB 客户端"""
        # 使用 PersistentClient，数据持久化到磁盘
        # 知识点：ChromaDB 有两种客户端模式
        # - EphemeralClient: 内存模式，程序退出数据丢失
        # - PersistentClient: 持久化模式，数据存在磁盘目录中
        self.chroma_client = chromadb.PersistentClient(path=STORAGE_DIR)

        # 当前索引缓存（避免每次查询都重新加载）
        self._index: Optional[VectorStoreIndex] = None

    # ========================================================
    # 构建索引
    # ========================================================

    def build_index(self, documents: List[Document], is_pdf: bool = False) -> int:
        """
        从 Document 列表构建向量索引并存入 ChromaDB。

        流程：
        1. 删除旧的 collection（单文档模式）
        2. 使用 SentenceSplitter 将 Document 切分为 Node
           （如果是 PDF 且解析为 Markdown，也用 SentenceSplitter，
             因为 MarkdownElementNodeParser 较重，SentenceSplitter 更通用）
        3. 创建新的 ChromaDB collection
        4. 构建 VectorStoreIndex（自动做 Embedding 并存入向量库）
        5. 缓存索引对象

        参数:
            documents: Document 对象列表
            is_pdf: 是否为 PDF（预留，用于选择不同分块策略）

        返回:
            int: 切分后的节点数量
        """
        # ---------- 1. 删除旧索引 ----------
        self.delete_index()

        # ---------- 2. 文本分块（Document → Node）----------
        # 知识点：SentenceSplitter 是 LlamaIndex 最常用的分块器
        # 它会尽量在句子边界处切分，保证语义完整性
        # 参数说明：
        #   chunk_size:    每个块的最大 token 数
        #   chunk_overlap: 相邻块之间的重叠 token 数，防止语义断裂
        #   separator:     句子分隔符，中文文本建议用空格或标点
        #   paragraph_separator: 段落分隔符，优先按段落切分
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator=" ",
            paragraph_separator="\n\n\n",
            # 二次切分正则：当一个句子超过 chunk_size 时，用此正则进一步切分
            secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",
        )

        # 将所有 Document 切分为 Node 列表
        nodes = splitter.get_nodes_from_documents(documents)
        node_count = len(nodes)

        if node_count == 0:
            raise ValueError("文档切分后节点数为 0，请检查文件内容是否为空")

        print(f"[IndexManager] 文档切分完成，共 {node_count} 个节点")

        # ---------- 3. 创建新的 ChromaDB Collection ----------
        # 知识点：ChromaDB 的 collection 相当于一个表
        # 每个 collection 存储一组向量及其对应的文本和元数据
        collection = self.chroma_client.create_collection(
            name=COLLECTION_NAME,
            # 相似度计算算法（默认 cosine）
            # 三种相似度算法对比（来自 02.LlamaIndex.md 第 3.4 节）：
            # - cosine (余弦相似度): 默认值，只看方向不看长度，适合语义检索
            # - dot_product (点积):   兼顾方向和向量模长
            # - euclidean (欧氏距离):  实际距离，受向量长度影响大
            metadata={"hnsw:space": "cosine"},
        )

        # 将 Chroma collection 包装成 LlamaIndex 的 VectorStore
        vector_store = ChromaVectorStore(chroma_collection=collection)

        # 创建 StorageContext（存储上下文，管理向量存储等）
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # ---------- 4. 构建向量索引 ----------
        # 知识点：VectorStoreIndex 是 LlamaIndex 最核心的索引类型
        # 它会自动：
        #   1. 调用 Embedding 模型将每个 Node 的文本转为向量
        #   2. 将向量存入 VectorStore（这里是 ChromaDB）
        #   3. 维护 Node 与向量的对应关系
        #
        # 构建完成后，可以通过 .as_retriever() 或 .as_query_engine() 使用
        self._index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            # 不使用内存中的默认存储，全部走 ChromaDB
            show_progress=True,
        )

        print(f"[IndexManager] 索引构建完成，已存入 ChromaDB")
        return node_count

    # ========================================================
    # 加载索引
    # ========================================================

    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        从 ChromaDB 加载已有的向量索引。

        知识点（03.LlamaIndex-RAG.md 第 4 节）：
        使用 VectorStoreIndex.from_vector_store() 从已有向量库加载索引，
        不需要重新做 embedding，速度很快。

        返回:
            VectorStoreIndex | None: 如果索引存在则返回，否则返回 None
        """
        # 如果缓存里有，直接返回
        if self._index is not None:
            return self._index

        # 检查 collection 是否存在
        try:
            collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            # collection 不存在，说明还没有索引
            return None

        # 检查 collection 是否为空
        if collection.count() == 0:
            return None

        # 从 VectorStore 构建 Index
        vector_store = ChromaVectorStore(chroma_collection=collection)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

        print(f"[IndexManager] 从 ChromaDB 加载索引，共 {collection.count()} 条向量")
        return self._index

    # ========================================================
    # 删除索引
    # ========================================================

    def delete_index(self) -> bool:
        """
        删除当前的索引（collection）。

        返回:
            bool: 是否成功删除
        """
        try:
            self.chroma_client.delete_collection(name=COLLECTION_NAME)
            self._index = None
            print(f"[IndexManager] 索引已删除")
            return True
        except Exception as e:
            # collection 不存在时也不算错误
            if "does not exist" in str(e).lower():
                return True
            print(f"[IndexManager] 删除索引失败: {e}")
            return False

    # ========================================================
    # 获取所有节点（供 BM25 检索用）
    # ========================================================

    def get_all_nodes(self) -> List[TextNode]:
        """
        从 ChromaDB 中获取所有节点，用于构建 BM25 检索器。

        知识点（03.LlamaIndex-RAG.md 第 9 节）：
        BM25 是基于词频的关键词检索，不需要向量，
        但需要拿到所有文档文本构建倒排索引。
        因此我们从 ChromaDB 读出所有文档，构造成 TextNode 列表。

        返回:
            List[TextNode]: 所有节点列表
        """
        try:
            collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            return []

        # 获取 collection 中所有文档
        all_docs = collection.get()

        nodes = []
        for doc_id, doc_text, doc_meta in zip(
            all_docs["ids"],
            all_docs["documents"],
            all_docs["metadatas"],
        ):
            if doc_text:  # 跳过空文档
                nodes.append(
                    TextNode(
                        id_=doc_id,
                        text=doc_text,
                        metadata=doc_meta if doc_meta else {},
                    )
                )

        return nodes

    # ========================================================
    # 获取 collection 中文档数量
    # ========================================================

    def get_node_count(self) -> int:
        """获取当前索引中的节点数量"""
        try:
            collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            return collection.count()
        except Exception:
            return 0

    # ========================================================
    # 检查索引是否存在
    # ========================================================

    def has_index(self) -> bool:
        """检查是否已有索引"""
        return self.get_node_count() > 0
