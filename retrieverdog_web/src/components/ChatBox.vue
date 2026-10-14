<!--
  问答聊天框组件
  ================
  极简风格的问答框，包含回答展示区和输入框。
  支持流式输出（打字机效果）。

  Props:
    - disabled: 是否禁用（没有上传文档时禁用输入）

  Emits:
    - @ask: 点击发送或回车时触发，携带问题文本
-->

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'

const props = defineProps<{
  disabled: boolean
}>()

const emit = defineEmits<{
  ask: [question: string]
}>()

// 回答内容（流式输出时逐步追加）
const answer = ref('')
// 是否正在生成回答
const isLoading = ref(false)
// 输入框的问题
const question = ref('')
// 回答区 DOM 引用，用于自动滚动到底部
const answerBox = ref<HTMLDivElement | null>(null)

/**
 * 暴露给父组件的方法
 * appendAnswer: 流式追加回答文本
 * setLoading:  设置加载状态
 * clear:       清空回答
 */
defineExpose({
  appendAnswer(text: string) {
    answer.value += text
    // 自动滚动到底部
    nextTick(() => {
      if (answerBox.value) {
        answerBox.value.scrollTop = answerBox.value.scrollHeight
      }
    })
  },
  setLoading(loading: boolean) {
    isLoading.value = loading
    if (loading) {
      answer.value = '' // 新问题开始时清空旧回答
    }
  },
  clear() {
    answer.value = ''
    isLoading.value = false
  },
})

/**
 * 发送问题
 */
function sendQuestion() {
  const q = question.value.trim()
  if (!q || props.disabled || isLoading.value) return

  emit('ask', q)
  question.value = '' // 发送后清空输入框
}

/**
 * 回车发送（Shift+Enter 换行）
 */
function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendQuestion()
  }
}
</script>

<template>
  <div class="chat-box">
    <!-- 回答展示区 -->
    <div ref="answerBox" class="answer-box">
      <template v-if="answer">
        <!-- 回答内容，保留换行和空格 -->
        <p class="answer-text">{{ answer }}</p>
      </template>
      <template v-else-if="isLoading">
        <p class="placeholder">正在思考中<span class="dots">...</span></p>
      </template>
      <template v-else>
        <p class="placeholder">上传文档后，在这里提问</p>
      </template>
    </div>

    <!-- 输入区 -->
    <div class="input-area">
      <input
        v-model="question"
        type="text"
        class="question-input"
        :placeholder="disabled ? '请先上传文档' : '输入你的问题，按回车发送...'"
        :disabled="disabled || isLoading"
        @keydown="handleKeydown"
      />
      <button
        class="send-btn"
        :disabled="disabled || isLoading || !question.trim()"
        @click="sendQuestion"
      >
        发送
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-box {
  display: flex;
  flex-direction: column;
  width: 100%;
  gap: 12px;
}

/* 回答展示区 */
.answer-box {
  min-height: 200px;
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #eee;
  border-radius: 8px;
  padding: 20px;
  background: #fff;
  line-height: 1.7;
}

.answer-text {
  margin: 0;
  color: #333;
  font-size: 15px;
  white-space: pre-wrap;
  word-break: break-word;
}

.placeholder {
  margin: 0;
  color: #bbb;
  font-size: 14px;
  text-align: center;
  padding-top: 60px;
}

.dots {
  animation: blink 1.4s infinite;
}

@keyframes blink {
  0%, 20% { opacity: 0.2; }
  50% { opacity: 1; }
  100%, 80% { opacity: 0.2; }
}

/* 输入区 */
.input-area {
  display: flex;
  gap: 8px;
}

.question-input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 15px;
  outline: none;
  transition: border-color 0.2s;
}

.question-input:focus {
  border-color: #666;
}

.question-input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.send-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  background: #333;
  color: #fff;
  font-size: 15px;
  cursor: pointer;
  transition: background 0.2s;
}

.send-btn:hover:not(:disabled) {
  background: #000;
}

.send-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}
</style>
