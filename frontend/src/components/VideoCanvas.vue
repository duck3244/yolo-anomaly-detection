<template>
  <div class="canvas-wrap">
    <canvas ref="canvasRef" :width="width" :height="height" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import type { Detection, FrameResult } from '@/types/api'

const props = defineProps<{
  source: HTMLVideoElement | null
  result: FrameResult | null
  width?: number
  height?: number
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const width = props.width ?? 640
const height = props.height ?? 480
let raf = 0

function drawBoxes(ctx: CanvasRenderingContext2D, dets: Detection[], sx: number, sy: number) {
  ctx.lineWidth = 2
  ctx.font = '14px system-ui'
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.bbox
    const color = d.is_anomaly ? '#ef4444' : '#10b981'
    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
    const label = `ID:${d.person_id} ${(d.anomaly_score * 100).toFixed(0)}%${d.is_anomaly ? ' ⚠' : ''}`
    const padY = 4
    ctx.fillRect(x1 * sx, y1 * sy - 18, ctx.measureText(label).width + 8, 18)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, x1 * sx + 4, y1 * sy - padY)
  }
}

function render() {
  const canvas = canvasRef.value
  const video = props.source
  if (canvas && video && video.readyState >= 2) {
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    if (props.result) {
      const sx = canvas.width / (props.result.frame_width || canvas.width)
      const sy = canvas.height / (props.result.frame_height || canvas.height)
      drawBoxes(ctx, props.result.detections, sx, sy)
    }
  }
  raf = requestAnimationFrame(render)
}

onMounted(() => { raf = requestAnimationFrame(render) })
onBeforeUnmount(() => cancelAnimationFrame(raf))
watch(() => props.source, () => {})
</script>

<style scoped>
.canvas-wrap { display: inline-block; background: #000; }
canvas { display: block; }
</style>
