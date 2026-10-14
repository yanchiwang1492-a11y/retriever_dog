"""
Pydantic 数据模型定义
====================
定义 API 请求和响应的数据结构，用于自动校验和文档生成。

知识点对应：
- FastAPI 内置 Pydantic 数据校验
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ============================================================
# 请求模型
# ============================================================

class QueryRequest(BaseModel):
    """
    提问请求体
    - question: 用户的问题
    - document_id: 文档 ID（单文档模式下也传，方便以后扩展）
    """
    question: str = Field(..., description="用户的问题", min_length=1)
    document_id: Optional[str] = Field(None, description="文档 ID，不传则使用当前文档")


# ============================================================
# 响应模型
# ============================================================

class UploadResponse(BaseModel):
    """
    文档上传响应
    - document_id: 文档唯一标识
    - filename: 原始文件名
    - node_count: 切分后的节点数量
    - message: 提示信息
    """
    document_id: str = Field(..., description="文档唯一标识")
    filename: str = Field(..., description="原始文件名")
    node_count: int = Field(..., description="切分后的节点数量")
    message: str = Field(..., description="处理结果提示")


class SourceNodeInfo(BaseModel):
    """
    检索来源节点信息
    - text: 节点文本内容
    - score: 相关性得分
    - metadata: 元数据（页码、文件名等）
    """
    text: str = Field(..., description="节点文本内容")
    score: float = Field(..., description="相关性得分")
    metadata: dict = Field(default_factory=dict, description="元数据")


class QueryResponse(BaseModel):
    """
    问答响应（非流式时用，流式通过 SSE 逐字推送）
    - answer: 最终回答
    - sources: 检索到的来源节点列表
    """
    answer: str = Field(..., description="AI 生成的回答")
    sources: List[SourceNodeInfo] = Field(default_factory=list, description="检索来源")


class DeleteResponse(BaseModel):
    """
    删除文档响应
    """
    success: bool = Field(..., description="是否删除成功")
    message: str = Field(..., description="提示信息")


class HealthResponse(BaseModel):
    """
    健康检查响应
    """
    status: str = Field(..., description="服务状态")
