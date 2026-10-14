/**
 * API 接口封装
 * ==============
 * 使用 axios 封装后端 API 调用。
 *
 * 接口列表：
 * 1. uploadDocument   上传文档
 * 2. queryStream      流式问答（逐字回调）
 * 3. deleteDocument   删除文档
 * 4. healthCheck      健康检查
 */

import axios from 'axios'

// 创建 axios 实例
// baseURL 指向后端 API 地址，开发环境后端默认 8000 端口
const request = axios.create({
  baseURL: 'http://localhost:8002/api',
  timeout: 300000, // 超时时间 5 分钟（上传+构建索引可能比较久）
})

// ============================================================
// 类型定义
// ============================================================

/** 上传响应 */
export interface UploadResponse {
  document_id: string
  filename: string
  node_count: number
  message: string
}

/** 删除响应 */
export interface DeleteResponse {
  success: boolean
  message: string
}

/** 健康检查响应 */
export interface HealthResponse {
  status: string
}

// ============================================================
// 1. 上传文档
// ============================================================

/**
 * 上传文档并构建索引
 * @param file 上传的文件对象
 * @param onProgress 上传进度回调（可选）
 */
export async function uploadDocument(
  file: File,
  onProgress?: (percent: number) => void
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await request.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    // 上传进度
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percent = Math.round(
          (progressEvent.loaded / progressEvent.total) * 100
        )
        onProgress(percent)
      }
    },
  })

  return res.data
}

// ============================================================
// 2. 流式问答
// ============================================================

/**
 * 流式提问，逐字回调返回答案
 * 使用 fetch + ReadableStream 实现流式读取（axios 对流式支持不好）
 *
 * @param question   用户问题
 * @param documentId 文档 ID（可选）
 * @param onToken    每收到一段文本的回调函数
 * @returns 完整的回答文本
 */
export async function queryStream(
  question: string,
  documentId: string | null = null,
  onToken: (text: string) => void
): Promise<string> {
  const response = await fetch('http://localhost:8002/api/query/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      document_id: documentId,
    }),
  })

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}))
    throw new Error(errData.detail || '请求失败')
  }

  // 获取 reader，逐块读取流
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('浏览器不支持流式读取')
  }

  const decoder = new TextDecoder('utf-8')
  let fullText = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    // 将 Uint8Array 解码为文本
    const chunk = decoder.decode(value, { stream: true })
    fullText += chunk

    // 调用回调，把新的文本片段传出去
    onToken(chunk)
  }

  return fullText
}

// ============================================================
// 3. 删除文档
// ============================================================

/**
 * 删除当前文档及索引
 */
export async function deleteDocument(): Promise<DeleteResponse> {
  const res = await request.delete('/document')
  return res.data
}

// ============================================================
// 4. 健康检查
// ============================================================

/**
 * 检查后端服务是否正常运行
 */
export async function healthCheck(): Promise<HealthResponse> {
  const res = await request.get('/health')
  return res.data
}

export default request
