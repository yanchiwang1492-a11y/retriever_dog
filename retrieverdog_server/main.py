"""
RetrieverDog 后端主入口
========================
FastAPI 应用，提供文档上传、流式问答、删除文档等 API 接口。

接口清单：
- POST /api/upload         上传文档并构建索引
- POST /api/query/stream   流式问答（SSE）
- DELETE /api/document     删除当前文档及索引
- GET  /api/health         健康检查

架构：
┌────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  FastAPI   │ →  │ document_loader  │ →  │  index_manager   │
│  (路由)    │    │  (文档解析)      │    │  (索引/向量库)    │
└─────┬──────┘    └──────────────────┘    └────────┬─────────┘
      │                                            │
      │            ┌──────────────────┐            │
      └──────────→ │  query_service   │ ←──────────┘
                   │ (检索+重排+生成)  │
                   └──────────────────┘

启动命令：
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path

# 导入配置
from config.settings import (
    init_settings,
    HOST,
    PORT,
    UPLOAD_DIR,
)

# 导入服务
from services.document_loader import load_document, is_supported_file
from services.index_manager import IndexManager
from services.query_service import QueryService

# 导入数据模型
from models.schemas import (
    QueryRequest,
    UploadResponse,
    DeleteResponse,
    HealthResponse,
)

# ============================================================
# 初始化
# ============================================================

# 1. 初始化 LlamaIndex 全局 Settings（LLM、Embedding、分块参数等）
init_settings()

# 2. 创建 FastAPI 应用
app = FastAPI(
    title="RetrieverDog API",
    description="极简文档检索问答系统后端 API",
    version="1.0.0",
)

# 3. 配置 CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 开发环境允许所有来源，生产环境建议限制
    allow_credentials=True,
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
)

# 4. 全局服务实例（单例模式，整个应用共享一份）
index_manager = IndexManager()
query_service = QueryService(index_manager)

# 当前文档 ID（单文档模式，全局只有一个）
current_document_id: str | None = None
current_filename: str | None = None


# ============================================================
# 健康检查
# ============================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口，用于确认服务是否正常运行。
    """
    return {"status": "ok"}


# ============================================================
# 文档上传
# ============================================================

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并构建向量索引。

    处理流程：
    1. 校验文件格式是否支持
    2. 保存文件到 uploads 目录
    3. 调用 document_loader 解析文档（根据格式选择解析器）
    4. 调用 index_manager 构建索引（分块 → 向量 → ChromaDB）
    5. 返回文档 ID 和节点数量

    注意：单文档模式，上传新文档会自动删除旧的索引。
    """
    global current_document_id, current_filename

    filename = file.filename or "unknown"

    # --- 1. 校验文件格式 ---
    if not is_supported_file(filename):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {Path(filename).suffix}。支持的格式包括 PDF、Word、Excel、PPT、图片、文本文件等。"
        )

    # --- 2. 保存文件到本地 ---
    # 用 UUID 生成唯一文件名，避免重名冲突
    file_ext = Path(filename).suffix.lower()
    saved_name = f"{uuid.uuid4().hex}{file_ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    # 读取上传的文件内容并写入磁盘
    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    print(f"[Upload] 文件已保存: {filename} → {saved_name}")

    try:
        # --- 3. 解析文档 ---
        documents = load_document(saved_path)
        print(f"[Upload] 文档解析完成，共 {len(documents)} 个 Document")

        # --- 4. 构建索引 ---
        # 判断是否为 PDF（PDF 是 Markdown 格式，分块策略略有不同）
        is_pdf = file_ext == ".pdf"
        node_count = index_manager.build_index(documents, is_pdf=is_pdf)

        # --- 5. 更新当前文档信息 ---
        current_document_id = uuid.uuid4().hex
        current_filename = filename

        return {
            "document_id": current_document_id,
            "filename": filename,
            "node_count": node_count,
            "message": f"文档上传成功，已切分为 {node_count} 个文本块",
        }

    except Exception as e:
        # 出错时清理已保存的文件
        if os.path.exists(saved_path):
            os.remove(saved_path)
        print(f"[Upload] 处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


# ============================================================
# 流式问答（SSE）
# ============================================================

@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式问答接口，使用 SSE (Server-Sent Events) 逐字推送回答。

    为什么用 SSE 而不是 WebSocket？
    - SSE 是单向的（服务端推送给客户端），对于问答场景足够用
    - 实现更简单，基于 HTTP 协议
    - 前端用 EventSource 或直接 fetch 读取流即可

    前端接收方式：
        const response = await fetch('/api/query/stream', {...})
        const reader = response.body.getReader()
        // 逐块读取并显示

    数据流格式：
        每个 chunk 是一段 UTF-8 文本，直接拼接就是完整回答。
    """
    # 校验是否已有索引
    if not index_manager.has_index():
        raise HTTPException(
            status_code=400,
            detail="请先上传文档再提问"
        )

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    print(f"[Query] 用户提问: {question}")

    # 定义异步生成器，用于 StreamingResponse
    async def generate():
        try:
            async for token in query_service.query_stream(question):
                # 直接 yield 文本（SSE 模式下前端按流读取）
                yield token
        except Exception as e:
            import traceback
            print(f"[Query] 生成失败: {e}")
            print("[Query] 错误堆栈:")
            traceback.print_exc()
            yield f"\n\n[错误] 生成回答时出错: {str(e)}"

    # 返回流式响应
    # media_type 设置为 text/event-stream 是 SSE 标准
    # 但这里我们用纯文本流，前端更容易处理
    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，确保流式输出
        },
    )


# ============================================================
# 删除文档
# ============================================================

@app.delete("/api/document", response_model=DeleteResponse)
async def delete_document():
    """
    删除当前文档及其索引。
    单文档模式下不需要传 ID，直接删除当前这一份。
    """
    global current_document_id, current_filename

    if not index_manager.has_index():
        return {
            "success": True,
            "message": "当前没有文档",
        }

    success = index_manager.delete_index()

    if success:
        current_document_id = None
        current_filename = None
        # 清理 uploads 目录中的文件（可选，这里保留文件方便排查）
        return {
            "success": True,
            "message": "文档已删除",
        }
    else:
        raise HTTPException(status_code=500, detail="删除失败")


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,  # 开发环境开启热重载
    )
