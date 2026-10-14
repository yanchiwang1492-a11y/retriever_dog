"""
查询服务
========
负责处理用户提问，执行完整的 RAG 流水线并返回答案。

RAG 流水线（两阶段检索 + 生成）：
┌─────────────────────────────────────────────────────┐
│  阶段 1：混合检索（召回）                              │
│  ┌─────────────┐    ┌─────────────┐                │
│  │ 向量检索     │    │ BM25 关键词 │                │
│  │ (语义匹配)   │    │ (词频匹配)  │                │
│  └──────┬──────┘    └──────┬──────┘                │
│         └─────────┬─────────┘                      │
│            RRF 融合（倒数排名融合）                   │
│                   ↓                                 │
│            Top-K = 15 条候选                        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  阶段 2：重排序（精排）                              │
│         qwen-rerank (DashScope 云端)                 │
│                   ↓                                 │
│            Top-N = 3 条最终片段                       │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  阶段 3：LLM 生成回答                                │
│         自定义 Prompt 模板 + 流式输出                 │
│         (SSE 推送到前端，打字机效果)                  │
└─────────────────────────────────────────────────────┘

知识点对应：
- 02.LlamaIndex.md 第 3.5 节：Retriever 检索器
- 02.LlamaIndex.md 第 8 节：Query Engine 查询引擎
- 02.LlamaIndex.md 第 11.1 节：自定义 Prompt 模板 ChatPromptTemplate
- 02.LlamaIndex.md 第 11.2 节：SimilarityPostprocessor 相似度后处理
- 02.LlamaIndex.md 第 11.3 节：流式输出 streaming=True
- 03.LlamaIndex-RAG.md 第 6 节：自定义提示词 + 检索内容打印
- 03.LlamaIndex-RAG.md 第 7 节：partial_format 动态赋值
- 03.LlamaIndex-RAG.md 第 8 节：FlagEmbeddingReranker 重排序
- 03.LlamaIndex-RAG.md 第 9 节：BM25Retriever 关键词检索
- 03.LlamaIndex-RAG.md 第 10 节：QueryFusionRetriever + RRF 混合检索
- 02.LlamaIndex.md 第 8.2 节：response.source_nodes 查看来源节点
"""

from typing import List, Tuple, AsyncGenerator

from llama_index.core import (
    VectorStoreIndex,
    ChatPromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryType, NodeWithScore
from typing import List as TypingList
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

from config.settings import (
    DASHSCOPE_API_KEY,
    RERANK_MODEL,
    SIMILARITY_TOP_K,
    RERANK_TOP_N,
    SIMILARITY_CUTOFF,
)


# ============================================================
# 自定义 Prompt 模板
# ============================================================
# 知识点（02.LlamaIndex.md 第 11.1 节）：
# 使用 ChatPromptTemplate 自定义问答提示词
# 占位符说明：
#   - {context_str}: 由 LlamaIndex 自动填充，检索到的上下文
#   - {query_str}:   由 LlamaIndex 自动填充，用户的问题
#
# 因为是单轮问答（没有历史记录），所以模板里不需要 {history}

CUSTOM_QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
你是一个专业的文档问答助手，请根据以下上下文信息回答用户的问题。

【上下文信息】
{context_str}

【回答规则】
1. 仅根据上面的上下文信息回答，不要编造上下文中没有的内容
2. 如果上下文信息不足以回答问题，请明确说"根据提供的文档资料无法回答这个问题"
3. 回答要简洁、准确、条理清晰，使用中文
4. 如果答案在上下文中有明确出处，可以简要说明来自哪部分内容
        """.strip(),
    ),
    ("user", "问题：{query_str}"),
])


# ============================================================
# 查询服务类
# ============================================================

class QueryService:
    """
    查询服务：封装完整的 RAG 检索问答流程。

    使用方法：
        service = QueryService(index_manager)
        async for token in service.query_stream("问题"):
            print(token)
    """

    def __init__(self, index_manager):
        """
        初始化查询服务。

        参数:
            index_manager: IndexManager 实例，用于获取索引和节点
        """
        self.index_manager = index_manager

    # ========================================================
    # 构建混合检索器
    # ========================================================

    def _build_hybrid_retriever(self, index: VectorStoreIndex):
        """
        构建混合检索器：向量检索 + BM25 关键词检索，用 RRF 算法融合。

        知识点（03.LlamaIndex-RAG.md 第 10 节）：
        - QueryFusionRetriever: 可以融合多个检索器的结果
        - FUSION_MODES.RECIPROCAL_RANK: RRF（倒数排名融合）算法
          对每个检索器的排名取倒数求和，无需手动调权重
        - 为什么用混合检索？
            向量检索擅长语义匹配（意思相近）
            BM25 擅长关键词精确匹配（字面相同）
            两者结合可以取长补短，提升召回率

        参数:
            index: VectorStoreIndex 索引对象

        返回:
            QueryFusionRetriever: 混合检索器
        """
        # --- 检索器 1：向量检索（语义匹配）---
        # similarity_top_k: 返回最相似的 K 个节点
        vector_retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

        # --- 检索器 2：BM25 关键词检索（词频匹配）---
        # 知识点（03.LlamaIndex-RAG.md 第 9 节）：
        # BM25 是经典的信息检索算法，基于词频和逆文档频率打分
        # 不需要 embedding，纯文本匹配，适合关键词精确查询
        # language="chinese": 使用中文分词
        all_nodes = self.index_manager.get_all_nodes()
        bm25_retriever = BM25Retriever(
            nodes=all_nodes,
            similarity_top_k=SIMILARITY_TOP_K,
            language="chinese",
        )

        # --- 混合检索器：RRF 融合两个检索器的结果 ---
        # RRF (Reciprocal Rank Fusion) 公式：
        #   score(d) = Σ 1 / (k + rank_i(d))
        # 其中 rank_i(d) 是文档 d 在第 i 个检索器中的排名
        # 优点：不需要归一化不同检索器的分数，只看排名
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            mode=FUSION_MODES.RECIPROCAL_RANK,  # RRF 融合算法
            similarity_top_k=SIMILARITY_TOP_K,  # 融合后返回的数量
            num_queries=1,  # 只生成 1 个查询（不做查询改写）
        )

        return hybrid_retriever

    # ========================================================
    # 构建重排序器
    # ========================================================

    def _build_reranker(self) -> DashScopeRerank:
        """
        构建 DashScope 云端重排序器。

        知识点（03.LlamaIndex-RAG.md 第 8 节）：
        重排序（Reranking）是两阶段检索中的第二阶段：
        第一阶段用轻量级的向量/BM25 快速召回大量候选
        第二阶段用更重但更精准的 Cross-Encoder 模型精排序
        这样兼顾了速度和准确性。

        这里使用 DashScope 云端的 qwen-rerank 模型，
        替代本地的 FlagEmbeddingReranker，不需要下载模型文件。

        返回:
            DashScopeRerank: 重排序后处理器
        """
        reranker = DashScopeRerank(
            model=RERANK_MODEL,
            top_n=RERANK_TOP_N,  # 重排序后保留的数量
            api_key=DASHSCOPE_API_KEY,
        )
        return reranker

    # ========================================================
    # 执行检索（两阶段：混合召回 + 重排序）
    # ========================================================

    def _retrieve(self, query_str: str) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
        """
        执行两阶段检索，返回重排序后的节点列表。

        阶段 1：混合检索（向量 + BM25，RRF 融合）召回 15 条
        阶段 2：qwen-rerank 重排序精选 3 条

        参数:
            query_str: 用户查询字符串

        返回:
            (retrieved_nodes, reranked_nodes):
                - retrieved_nodes: 第一阶段混合检索的结果
                - reranked_nodes:  第二阶段重排序后的结果
        """
        # 获取索引
        index = self.index_manager.load_index()
        if index is None:
            raise ValueError("尚未上传文档或索引不存在，请先上传文档")

        # --- 阶段 1：混合检索 ---
        hybrid_retriever = self._build_hybrid_retriever(index)
        retrieved_nodes = hybrid_retriever.retrieve(query_str)

        print(f"[QueryService] 阶段1-混合检索召回 {len(retrieved_nodes)} 条")
        # 打印前 5 条的分数（RRF 融合分数，不是 0~1 范围的相似度）
        for i, node in enumerate(retrieved_nodes[:5], 1):
            print(f"  召回{i}: RRF分数={node.score:.4f}  片段前30字={node.text[:30]}...")

        # 注意：RRF（倒数排名融合）的分数范围不是 0~1 的余弦相似度
        # 它是排名倒数求和的值，通常在 0.01~1.0 之间，具体取决于召回数量
        # 所以相似度阈值过滤放在重排序之后做，用重排序的真实相关度分数更准确
        if not retrieved_nodes:
            return [], []

        # --- 阶段 2：重排序 ---
        reranker = self._build_reranker()
        reranked_nodes = reranker.postprocess_nodes(
            retrieved_nodes,
            query_str=query_str,
        )

        print(f"[QueryService] 阶段2-重排序完成，保留 {len(reranked_nodes)} 条")
        for i, node in enumerate(reranked_nodes, 1):
            print(f"  重排{i}: 相关度={node.score:.4f}  片段前30字={node.text[:30]}...")

        # 相似度阈值过滤（用重排序后的真实相关度分数，范围 0~1）
        # 知识点（02.LlamaIndex.md 第 11.2 节）：
        # SimilarityPostprocessor 可以根据相似度阈值过滤节点
        filtered_nodes = [
            node for node in reranked_nodes
            if node.score >= SIMILARITY_CUTOFF
        ]
        if len(filtered_nodes) < len(reranked_nodes):
            print(f"[QueryService] 相似度过滤（重排序后）：{len(reranked_nodes)} → {len(filtered_nodes)}")
            reranked_nodes = filtered_nodes

        return retrieved_nodes, reranked_nodes

    # ========================================================
    # 构建完整的流式查询引擎
    # ========================================================

    def _build_query_engine(self, index: VectorStoreIndex, use_rerank: bool = True):
        """
        构建完整的查询引擎：混合检索 + 重排序 + 自定义 Prompt + 流式输出。

        使用 RetrieverQueryEngine 组装各个组件：
        - retriever: 混合检索器（向量 + BM25 RRF）
        - node_postprocessors: 重排序后处理器
        - response_synthesizer: 响应合成器（自定义 Prompt + 流式）

        这种方式是 LlamaIndex 的标准组装方式，流式输出最稳定可靠。

        参数:
            index: 向量索引
            use_rerank: 是否启用重排序（默认 True，失败时可设为 False 降级）
        """
        # 1. 混合检索器
        hybrid_retriever = self._build_hybrid_retriever(index)

        # 2. 后处理器列表
        postprocessors = []
        if use_rerank:
            # 重排序后处理器
            reranker = self._build_reranker()
            postprocessors.append(reranker)

        # 3. 响应合成器（自定义 Prompt + 流式输出）
        # 知识点（02.LlamaIndex.md 第 11.1 节）：自定义 Prompt 模板
        # 知识点（02.LlamaIndex.md 第 11.3 节）：streaming=True 流式输出
        response_synthesizer = get_response_synthesizer(
            text_qa_template=CUSTOM_QA_PROMPT,
            streaming=True,
        )

        # 4. 组装成完整的查询引擎
        # 知识点：RetrieverQueryEngine 可以把自定义检索器和响应合成器组装起来
        query_engine = RetrieverQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )

        return query_engine

    # ========================================================
    # 流式问答（主入口）
    # ========================================================

    async def query_stream(self, query_str: str) -> AsyncGenerator[str, None]:
        """
        流式问答：逐 token 返回回答，用于 SSE 推送。

        流程：
        1. 从 ChromaDB 加载索引
        2. 构建查询引擎（混合检索 + 重排序 + 自定义 Prompt + 流式）
        3. 调用 aquery 异步查询，逐 token yield

        知识点（02.LlamaIndex.md 第 11.3 节）：
        - streaming=True 启用流式输出
        - response_gen() 返回生成器，可以逐字迭代

        参数:
            query_str: 用户问题

        Yields:
            str: 每次 yield 一段文本
        """
        # --- 1. 获取索引 ---
        index = self.index_manager.load_index()
        if index is None:
            yield "请先上传文档再提问。"
            return

        # --- 2. 构建查询引擎（先尝试带重排序的）---
        query_engine = self._build_query_engine(index, use_rerank=True)

        # --- 3. 流式查询 ---
        try:
            # aquery 是异步版本，返回 StreamingResponse
            response = await query_engine.aquery(query_str)
        except Exception as e:
            # 如果重排序失败，降级为不带重排序的查询
            print(f"[QueryService] 重排序失败，降级为混合检索模式: {e}")
            query_engine = self._build_query_engine(index, use_rerank=False)
            response = await query_engine.aquery(query_str)

        # 打印来源节点（调试用）
        print(f"[QueryService] 检索到 {len(response.source_nodes)} 个来源节点")
        for i, node in enumerate(response.source_nodes, 1):
            score_str = f"{node.score:.4f}" if node.score else "N/A"
            print(f"  来源{i}: 分数={score_str}  前30字={node.text[:30]}...")

        # 逐 token 生成并返回
        # aquery() 返回的流式响应，response_gen 是异步生成器
        async for token in response.async_response_gen():
            yield token

    # ========================================================
    # 获取检索来源（用于调试/显示）
    # ========================================================

    def get_sources(self, query_str: str) -> List[dict]:
        """
        获取某个问题的检索来源节点信息（不含生成回答）。

        知识点（02.LlamaIndex.md 第 8.1 节）：
        response.source_nodes 可以查看检索到的来源节点，
        包括节点文本、得分、元数据等。

        参数:
            query_str: 用户问题

        返回:
            List[dict]: 来源节点信息列表
        """
        _, reranked_nodes = self._retrieve(query_str)

        sources = []
        for node in reranked_nodes:
            sources.append({
                "text": node.text,
                "score": float(node.score),
                "metadata": node.metadata,
            })

        return sources
