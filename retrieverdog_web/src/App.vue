<!--
  RetrieverDog 主页面
  =====================
  极简风格的单页面应用，包含：
  1. 标题区：RetrieverDog + 一句话介绍
  2. 上传区：文件上传组件
  3. 问答区：聊天框组件

  设计风格：黑白灰 + 居中单列布局，除了必要功能不添加任何多余设计。
-->

<script setup lang="ts">
import { ref } from 'vue'
import UploadArea from './components/UploadArea.vue'
import ChatBox from './components/ChatBox.vue'
import { uploadDocument, queryStream } from './api'

// ---------- 状态 ----------

// 当前已上传的文件名（显示在上传区）
const currentFile = ref('')
// 当前文档 ID
const documentId = ref<string | null>(null)
// 是否正在上传
const uploading = ref(false)
// 上传状态文本
const uploadStatus = ref('')
// 错误提示
const errorMsg = ref('')

// ChatBox 组件引用（调用暴露的方法）
const chatBox = ref<InstanceType<typeof ChatBox> | null>(null)

// 计算属性：是否可以提问（有文档且不在上传中）
const canAsk = () => !!currentFile.value && !uploading.value

// ---------- 方法 ----------

/**
 * 处理文件上传
 */
async function handleUpload(file: File) {
  if (uploading.value) return

  uploading.value = true
  uploadStatus.value = '上传中...'
  errorMsg.value = ''
  // 清空聊天区
  chatBox.value?.clear()

  try {
    uploadStatus.value = '正在解析文档并构建索引，请稍候...'

    // 调用上传接口
    const result = await uploadDocument(file, (percent) => {
      uploadStatus.value = `上传中... ${percent}%`
    })

    // 上传成功
    currentFile.value = result.filename
    documentId.value = result.document_id
    uploadStatus.value = ''

    // 简单提示
    console.log(`文档上传成功，节点数：${result.node_count}`)
  } catch (err: any) {
    errorMsg.value = err?.response?.data?.detail || err.message || '上传失败'
    currentFile.value = ''
    documentId.value = null
  } finally {
    uploading.value = false
    uploadStatus.value = ''
  }
}

/**
 * 处理提问
 */
async function handleAsk(question: string) {
  if (!canAsk() || !chatBox.value) return

  // 设置加载状态
  chatBox.value.setLoading(true)
  errorMsg.value = ''

  try {
    // 流式获取回答，逐字追加
    await queryStream(question, documentId.value, (token) => {
      chatBox.value?.appendAnswer(token)
    })
  } catch (err: any) {
    const errText = err?.response?.data?.detail || err.message || '提问失败'
    chatBox.value?.appendAnswer(`\n\n[错误] ${errText}`)
  } finally {
    chatBox.value?.setLoading(false)
  }
}
</script>

<template>
  <div class="app">
    <!-- 标题区 -->
    <header class="header">
      <h1 class="title">RetrieverDog</h1>
      <p class="subtitle">上传文档，即刻问答你的专属资料</p>
    </header>

    <!-- 主体内容区 -->
    <main class="main">
      <!-- 上传区 -->
      <UploadArea
        :uploading="uploading"
        :upload-status="uploadStatus"
        :current-file="currentFile"
        @upload="handleUpload"
      />

      <!-- 错误提示 -->
      <div v-if="errorMsg" class="error-msg">
        {{ errorMsg }}
      </div>

      <!-- 问答区 -->
      <ChatBox
        ref="chatBox"
        :disabled="!canAsk()"
        @ask="handleAsk"
      />
    </main>

    <!-- 页脚（极简，一行文字） -->
    <footer class="footer">
      Powered by LlamaIndex · RetrieverDog
    </footer>
  </div>
</template>

<style scoped>
/* 整体布局：居中单列 */
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #fafafa;
  padding: 48px 20px;
  box-sizing: border-box;
}

/* 标题区 */
.header {
  text-align: center;
  margin-bottom: 40px;
}

.title {
  font-size: 32px;
  font-weight: 600;
  color: #222;
  margin: 0 0 8px 0;
  letter-spacing: 1px;
}

.subtitle {
  font-size: 14px;
  color: #888;
  margin: 0;
}

/* 主体区 */
.main {
  width: 100%;
  max-width: 640px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 错误提示 */
.error-msg {
  padding: 12px 16px;
  background: #fff2f0;
  border: 1px solid #ffccc7;
  border-radius: 6px;
  color: #cf1322;
  font-size: 14px;
  text-align: center;
}

/* 页脚 */
.footer {
  margin-top: auto;
  padding-top: 48px;
  font-size: 12px;
  color: #ccc;
}
</style>
