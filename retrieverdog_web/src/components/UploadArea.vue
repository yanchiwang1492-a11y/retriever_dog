<!--
  文件上传组件
  ==============
  极简风格的文件上传区域，支持点击选择和拖拽上传。
  只允许上传一个文件，上传后显示文件名和状态。

  Props:
    - uploading:    是否正在上传中
    - uploadStatus: 上传状态文本（显示给用户看的提示）
    - currentFile:  当前已上传的文件名

  Emits:
    - @upload: 选择文件后触发，携带 File 对象
-->

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  uploading: boolean
  uploadStatus: string
  currentFile: string
}>()

const emit = defineEmits<{
  upload: [file: File]
}>()

// 是否处于拖拽悬浮状态（控制虚线框高亮）
const isDragging = ref(false)

// 隐藏的 file input 引用
const fileInput = ref<HTMLInputElement | null>(null)

// 支持的文件类型（用于 accept 属性提示，实际以后端校验为准）
const acceptTypes =
  '.pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt,.md,.csv,.json,.xml,.html,.css,.js,.py,.png,.jpg,.jpeg,.bmp,.tiff'

/**
 * 点击上传区域 → 触发隐藏的 file input 点击
 */
function handleClick() {
  fileInput.value?.click()
}

/**
 * 选择文件后的处理
 */
function handleFileChange(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    emit('upload', file)
  }
  // 清空 input，这样选同一个文件也能重新触发
  target.value = ''
}

// ---- 拖拽相关 ----

function handleDragOver(event: DragEvent) {
  event.preventDefault()
  isDragging.value = true
}

function handleDragLeave() {
  isDragging.value = false
}

function handleDrop(event: DragEvent) {
  event.preventDefault()
  isDragging.value = false

  const file = event.dataTransfer?.files?.[0]
  if (file) {
    emit('upload', file)
  }
}
</script>

<template>
  <div class="upload-area">
    <!-- 隐藏的文件选择框 -->
    <input
      ref="fileInput"
      type="file"
      :accept="acceptTypes"
      style="display: none"
      @change="handleFileChange"
    />

    <!-- 上传区域（虚线框） -->
    <div
      class="upload-box"
      :class="{ 'is-dragging': isDragging, 'is-uploading': uploading }"
      @click="handleClick"
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
      @drop="handleDrop"
    >
      <div class="upload-icon">📎</div>
      <div class="upload-text">
        <template v-if="uploading">
          {{ uploadStatus }}
        </template>
        <template v-else-if="currentFile">
          <span class="filename">{{ currentFile }}</span>
          <span class="hint">点击重新上传</span>
        </template>
        <template v-else>
          <span class="title">点击或拖拽文件到此处上传</span>
          <span class="hint">支持 PDF / Word / Excel / PPT / 图片 / 文本 等</span>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.upload-area {
  width: 100%;
}

.upload-box {
  /* 虚线边框 */
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  background: #fff;
}

/* 拖拽悬浮时的样式 */
.upload-box.is-dragging {
  border-color: #333;
  background: #f8f8f8;
}

/* 上传中禁用点击 */
.upload-box.is-uploading {
  cursor: not-allowed;
  opacity: 0.7;
}

.upload-icon {
  font-size: 32px;
  margin-bottom: 12px;
}

.upload-text {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.title {
  font-size: 16px;
  color: #333;
}

.hint {
  font-size: 13px;
  color: #999;
}

.filename {
  font-size: 15px;
  color: #333;
  font-weight: 500;
  word-break: break-all;
}
</style>
