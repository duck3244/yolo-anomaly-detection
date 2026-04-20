<template>
  <div>
    <h2>학습</h2>
    <el-alert v-if="error" :title="error" type="error" show-icon class="mb" />

    <el-card>
      <h3>새 학습 작업</h3>
      <div class="form">
        <input type="file" accept="video/*" @change="onFile" />
        <el-input-number v-model="maxFrames" :min="100" :max="10000" :step="100" /> max_frames
        <el-input v-model="modelSaveName" placeholder="trained_model.pkl" />
        <el-button type="primary" @click="start" :disabled="!file || !session.current || running">훈련 시작</el-button>
      </div>
    </el-card>

    <el-card class="mt" v-if="job">
      <h3>진행 중인 Job: {{ job.job_id.slice(0, 8) }}</h3>
      <div>상태: <el-tag :type="stateTag(job.state)">{{ job.state }}</el-tag></div>
      <el-progress :percentage="Math.round((job.progress ?? 0) * 100)" />
      <div class="muted">{{ job.message }}</div>
      <pre v-if="job.result" class="result">{{ JSON.stringify(job.result, null, 2) }}</pre>
      <div v-if="job.error" class="err">{{ job.error }}</div>
    </el-card>

    <el-card class="mt">
      <h3>저장된 모델</h3>
      <el-table :data="models" size="small">
        <el-table-column prop="name" label="이름" />
        <el-table-column label="크기">
          <template #default="{ row }">{{ formatSize(row.size_bytes) }}</template>
        </el-table-column>
        <el-table-column label="수정 시각">
          <template #default="{ row }">{{ new Date(row.modified_at * 1000).toLocaleString() }}</template>
        </el-table-column>
        <el-table-column label="작업">
          <template #default="{ row }">
            <el-button size="small" type="primary" @click="loadModel(row.name)" :disabled="!session.current">로드</el-button>
            <el-button size="small" type="danger" @click="deleteModel(row.name)">삭제</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-button size="small" @click="refreshModels">새로고침</el-button>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { modelsApi, trainApi } from '@/api'
import type { JobStatus, ModelMeta } from '@/types/api'
import { useSessionStore } from '@/stores/session'
import { useWebSocket } from '@/composables/useWebSocket'

const session = useSessionStore()
const file = ref<File | null>(null)
const maxFrames = ref(1000)
const modelSaveName = ref('trained_model.pkl')
const running = ref(false)
const job = ref<JobStatus | null>(null)
const models = ref<ModelMeta[]>([])
const error = ref<string | null>(null)
let wsCtrl: ReturnType<typeof useWebSocket> | null = null

function onFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  file.value = input.files?.[0] ?? null
}

function stateTag(s: string) {
  return s === 'completed' ? 'success' : s === 'failed' ? 'danger' : 'warning'
}

function formatSize(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1048576).toFixed(1)} MB`
}

async function start() {
  if (!file.value || !session.current) return
  error.value = null
  running.value = true
  try {
    const j = await trainApi.start(session.current.session_id, file.value, maxFrames.value, modelSaveName.value)
    job.value = j
    subscribe(j.job_id)
  } catch (e: any) {
    error.value = e.message
  } finally {
    running.value = false
  }
}

function subscribe(jobId: string) {
  wsCtrl?.close()
  wsCtrl = useWebSocket(`/ws/training?job_id=${jobId}`, {
    onMessage(d) {
      if (d?.error) { error.value = d.error; return }
      job.value = d as JobStatus
      if (job.value.state === 'completed') {
        ElMessage.success('훈련 완료')
        refreshModels()
        session.refresh().catch(() => {})
      } else if (job.value.state === 'failed') {
        ElMessage.error(`훈련 실패: ${job.value.error ?? ''}`)
      }
    },
  })
  wsCtrl.connect()
}

async function refreshModels() {
  try { models.value = await modelsApi.list() } catch (e: any) { error.value = e.message }
}

async function loadModel(name: string) {
  if (!session.current) return
  try {
    await modelsApi.load(session.current.session_id, name)
    ElMessage.success('모델 로드 완료')
    await session.refresh()
  } catch (e: any) { ElMessage.error(e.message) }
}

async function deleteModel(name: string) {
  try {
    await modelsApi.remove(name)
    await refreshModels()
  } catch (e: any) { ElMessage.error(e.message) }
}

onMounted(refreshModels)
onBeforeUnmount(() => wsCtrl?.close())
</script>

<style scoped>
.form { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.mt { margin-top: 16px; }
.mb { margin-bottom: 12px; }
.muted { color: #6b7280; margin-top: 4px; }
.result { background: #0f172a; color: #e2e8f0; padding: 8px; border-radius: 4px; overflow: auto; }
.err { color: #ef4444; }
</style>
