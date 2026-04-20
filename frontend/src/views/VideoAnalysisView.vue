<template>
  <div>
    <h2>비디오 분석</h2>
    <el-alert v-if="error" :title="error" type="error" show-icon class="mb" />
    <div class="toolbar">
      <input type="file" accept="video/*" @change="onFile" />
      <el-button type="primary" @click="start" :disabled="!file || streaming || !session.current">분석 시작</el-button>
      <el-button @click="stop" :disabled="!streaming">정지</el-button>
      <el-tag :type="streaming ? 'success' : 'info'">{{ streaming ? 'streaming' : 'idle' }}</el-tag>
      <span class="muted">{{ videoState }}{{ frameCount ? ` · sent ${frameCount}` : '' }}</span>
    </div>
    <div class="stage">
      <video ref="videoEl" controls muted playsinline :src="videoUrl" style="max-width: 640px" />
      <VideoCanvas :source="videoEl" :result="lastResult" :width="640" :height="480" />
    </div>
    <el-card class="mt">
      <div>총 검출: {{ lastResult?.total_detections ?? 0 }} · 추적: {{ lastResult?.total_tracked ?? 0 }}</div>
      <div>처리시간: {{ (lastResult?.processing_time_ms ?? 0).toFixed(1) }} ms</div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, nextTick, ref } from 'vue'
import VideoCanvas from '@/components/VideoCanvas.vue'
import { useSessionStore } from '@/stores/session'
import { useWebSocket } from '@/composables/useWebSocket'
import type { FrameResult } from '@/types/api'

const session = useSessionStore()
const file = ref<File | null>(null)
const videoUrl = ref('')
const videoEl = ref<HTMLVideoElement | null>(null)
const streaming = ref(false)
const lastResult = ref<FrameResult | null>(null)
const error = ref<string | null>(null)
const videoState = ref('idle')
const frameCount = ref(0)
let wsCtrl: ReturnType<typeof useWebSocket> | null = null
let captureTimer: number | null = null
let sentAt = 0

async function onFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const f = input.files?.[0]
  if (!f) return
  // 이전 blob URL 해제
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  file.value = f
  videoUrl.value = URL.createObjectURL(f)
  videoState.value = `loading ${f.name}…`
  frameCount.value = 0
  lastResult.value = null
  error.value = null
  // src가 Vue로 갱신된 뒤 명시적으로 load() 호출
  await nextTick()
  if (videoEl.value) {
    try { videoEl.value.pause() } catch {}
    videoEl.value.currentTime = 0
    videoEl.value.load()
  }
}

async function waitForVideo(v: HTMLVideoElement, timeoutMs = 10000) {
  if (v.readyState >= 2 && v.videoWidth > 0) return
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      v.removeEventListener('loadedmetadata', ok)
      v.removeEventListener('loadeddata', ok)
      v.removeEventListener('canplay', ok)
      v.removeEventListener('error', bad)
      clearTimeout(t)
    }
    const ok = () => { if (v.videoWidth > 0 && v.readyState >= 2) { cleanup(); resolve() } }
    const bad = () => { cleanup(); reject(new Error(`video element error (code=${v.error?.code ?? '?'})`)) }
    const t = setTimeout(() => { cleanup(); reject(new Error('timeout waiting for video')) }, timeoutMs)
    v.addEventListener('loadedmetadata', ok)
    v.addEventListener('loadeddata', ok)
    v.addEventListener('canplay', ok)
    v.addEventListener('error', bad, { once: true })
  })
}

async function start() {
  error.value = null
  frameCount.value = 0
  if (!session.current) await session.ensure()
  if (!session.current || !videoEl.value) { error.value = '세션 또는 비디오 엘리먼트 없음'; return }

  const v = videoEl.value
  // 재생 종료 상태면 처음으로 되돌림
  if (v.ended || (v.duration && v.currentTime >= v.duration - 0.05)) {
    try { v.currentTime = 0 } catch {}
  }
  // readyState가 유효하지 않으면 강제 재로드
  if (v.readyState < 2 || v.videoWidth === 0) {
    try { v.load() } catch {}
  }

  videoState.value = 'loading video…'
  try {
    await waitForVideo(v)
    await v.play()
    videoState.value = `playing ${v.videoWidth}x${v.videoHeight}`
  } catch (e: any) {
    error.value = `비디오 재생 실패: ${e?.message ?? e}`
    videoState.value = 'video error'
    return
  }

  wsCtrl = useWebSocket(`/ws/stream?session_id=${session.current.session_id}`, {
    onMessage(d) {
      if (d?.error) { error.value = d.error; return }
      lastResult.value = d as FrameResult
    },
    onClose() { streaming.value = false },
    reconnect: false,
  })
  wsCtrl.connect()
  streaming.value = true
  loop()
}

async function loop() {
  if (!streaming.value || !videoEl.value || !wsCtrl) return
  const v = videoEl.value
  if (v.readyState < 2 || v.videoWidth === 0) {
    captureTimer = window.setTimeout(loop, 200)
    return
  }
  if (v.ended) {
    videoState.value = 'ended'
    stop()
    return
  }
  if (!v.paused) {
    // 백프레셔: WS 버퍼가 쌓이면 이번 프레임 스킵
    const ws = wsCtrl.socket.value
    if (ws && ws.bufferedAmount < 256 * 1024) {
      const canvas = document.createElement('canvas')
      canvas.width = v.videoWidth
      canvas.height = v.videoHeight
      canvas.getContext('2d')!.drawImage(v, 0, 0, canvas.width, canvas.height)
      const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, 'image/jpeg', 0.7))
      if (blob) {
        sentAt = performance.now()
        wsCtrl.send(await blob.arrayBuffer())
        frameCount.value++
      }
    }
  }
  captureTimer = window.setTimeout(loop, 66) // ~15 fps
}

function stop() {
  streaming.value = false
  if (captureTimer !== null) { clearTimeout(captureTimer); captureTimer = null }
  wsCtrl?.close(); wsCtrl = null
  videoEl.value?.pause()
  videoState.value = 'stopped'
}

onBeforeUnmount(stop)
</script>

<style scoped>
.toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
.stage { display: flex; gap: 16px; }
.mt { margin-top: 12px; }
.mb { margin-bottom: 12px; }
.muted { color: #6b7280; font-size: 12px; }
</style>
