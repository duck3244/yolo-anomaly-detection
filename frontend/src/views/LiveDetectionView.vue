<template>
  <div>
    <h2>실시간 검출 (웹캠)</h2>
    <el-alert v-if="error" :title="error" type="error" show-icon class="mb" />
    <div class="toolbar">
      <el-button type="primary" @click="start" :disabled="streaming || !session.current">시작</el-button>
      <el-button @click="stop" :disabled="!streaming">정지</el-button>
      <el-tag :type="streaming ? 'success' : 'info'">{{ streaming ? 'streaming' : 'idle' }}</el-tag>
      <span class="muted">{{ lastLatencyMs.toFixed(0) }} ms · {{ fps.toFixed(1) }} fps</span>
    </div>
    <div class="stage">
      <video ref="videoEl" autoplay playsinline muted class="offscreen-video" />
      <VideoCanvas :source="videoEl" :result="lastResult" :width="640" :height="480" />
      <el-card class="side">
        <h4>검출 결과</h4>
        <div class="muted">비디오 상태: {{ videoState }}</div>
        <div v-if="!lastResult" class="muted">대기 중</div>
        <ul v-else>
          <li v-for="d in lastResult.detections" :key="d.person_id">
            <strong>#{{ d.person_id }}</strong>
            score {{ d.anomaly_score.toFixed(2) }}
            <el-tag v-if="d.is_anomaly" type="danger" size="small">ANOMALY</el-tag>
          </li>
        </ul>
      </el-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'
import VideoCanvas from '@/components/VideoCanvas.vue'
import { useSessionStore } from '@/stores/session'
import { useWebSocket } from '@/composables/useWebSocket'
import type { FrameResult } from '@/types/api'

const session = useSessionStore()
const videoEl = ref<HTMLVideoElement | null>(null)
const streaming = ref(false)
const error = ref<string | null>(null)
const lastResult = ref<FrameResult | null>(null)
const lastLatencyMs = ref(0)
const fps = ref(0)
const videoState = ref('idle')

let stream: MediaStream | null = null
let captureTimer: number | null = null
let wsCtrl: ReturnType<typeof useWebSocket> | null = null
let sentAt = 0
let frameCount = 0
let fpsTimer: number | null = null

async function waitForVideo(v: HTMLVideoElement, timeoutMs = 5000) {
  if (v.readyState >= 2) return
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      v.removeEventListener('loadedmetadata', ok)
      v.removeEventListener('loadeddata', ok)
      v.removeEventListener('error', bad)
      clearTimeout(t)
    }
    const ok = () => { cleanup(); resolve() }
    const bad = () => { cleanup(); reject(new Error('video element error')) }
    const t = setTimeout(() => { cleanup(); reject(new Error('timeout waiting for video')) }, timeoutMs)
    v.addEventListener('loadedmetadata', ok, { once: true })
    v.addEventListener('loadeddata', ok, { once: true })
    v.addEventListener('error', bad, { once: true })
  })
}

async function start() {
  error.value = null
  videoState.value = 'requesting camera…'
  if (!session.current) { await session.ensure() }
  if (!session.current) { error.value = '세션 생성 실패'; return }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    })
  } catch (e: any) {
    error.value = `카메라 접근 실패: ${e?.name ?? ''} ${e?.message ?? e}`
    videoState.value = 'camera error'
    return
  }

  if (!videoEl.value) { error.value = 'video element not mounted'; return }
  videoEl.value.srcObject = stream
  videoState.value = 'waiting for metadata…'
  try {
    await waitForVideo(videoEl.value)
    await videoEl.value.play()
    videoState.value = `playing ${videoEl.value.videoWidth}x${videoEl.value.videoHeight}`
  } catch (e: any) {
    error.value = `비디오 초기화 실패: ${e?.message ?? e}`
    videoState.value = 'video error'
    return
  }

  const sid = session.current.session_id
  wsCtrl = useWebSocket(`/ws/stream?session_id=${sid}`, {
    onMessage(data) {
      if (data?.error) { error.value = data.error; return }
      lastResult.value = data as FrameResult
      lastLatencyMs.value = performance.now() - sentAt
      frameCount++
    },
    onClose() { streaming.value = false },
    reconnect: false,
  })
  wsCtrl.connect()
  streaming.value = true
  scheduleCapture()
  fpsTimer = window.setInterval(() => { fps.value = frameCount; frameCount = 0 }, 1000)
}

function scheduleCapture() {
  const loop = async () => {
    if (!streaming.value) return
    await captureAndSend()
    captureTimer = window.setTimeout(loop, 100) // ~10 fps
  }
  loop()
}

async function captureAndSend() {
  if (!videoEl.value || !wsCtrl) return
  const v = videoEl.value
  if (v.readyState < 2) return
  const ws = wsCtrl.socket.value
  if (ws && ws.bufferedAmount > 256 * 1024) return // 백프레셔: 직전 프레임이 아직 송신 중
  const canvas = document.createElement('canvas')
  canvas.width = v.videoWidth || 640
  canvas.height = v.videoHeight || 480
  canvas.getContext('2d')!.drawImage(v, 0, 0, canvas.width, canvas.height)
  const blob = await new Promise<Blob | null>((res) => canvas.toBlob(res, 'image/jpeg', 0.7))
  if (!blob) return
  const buf = await blob.arrayBuffer()
  sentAt = performance.now()
  wsCtrl.send(buf)
}

function stop() {
  streaming.value = false
  if (captureTimer !== null) { clearTimeout(captureTimer); captureTimer = null }
  if (fpsTimer !== null) { clearInterval(fpsTimer); fpsTimer = null }
  wsCtrl?.close()
  wsCtrl = null
  stream?.getTracks().forEach((t) => t.stop())
  stream = null
  if (videoEl.value) videoEl.value.srcObject = null
}

onBeforeUnmount(stop)
</script>

<style scoped>
.toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
.muted { color: #6b7280; }
.stage { display: flex; gap: 16px; }
.side { flex: 1; }
.mb { margin-bottom: 12px; }
ul { margin: 0; padding-left: 18px; }
.offscreen-video { position: absolute; left: -9999px; top: -9999px; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
</style>
