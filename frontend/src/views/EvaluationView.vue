<template>
  <div>
    <h2>평가</h2>
    <el-alert v-if="error" :title="error" type="error" show-icon class="mb" />
    <el-card>
      <h4>CSV 업로드</h4>
      <p class="muted">형식: 각 행에 `f1,f2,...,fN,label` — label은 0(정상)/1(이상)</p>
      <input type="file" accept=".csv,text/csv" @change="onFile" />
      <el-button type="primary" @click="run" :disabled="!session.current || !features.length">평가 실행</el-button>
      <div class="muted">rows: {{ features.length }}, features/row: {{ features[0]?.length ?? 0 }}</div>
    </el-card>

    <el-card class="mt" v-if="result">
      <h3>결과</h3>
      <el-row :gutter="12">
        <el-col :span="6"><div class="k">Accuracy</div><div class="v">{{ result.accuracy.toFixed(3) }}</div></el-col>
        <el-col :span="6"><div class="k">Precision</div><div class="v">{{ result.precision.toFixed(3) }}</div></el-col>
        <el-col :span="6"><div class="k">Recall</div><div class="v">{{ result.recall.toFixed(3) }}</div></el-col>
        <el-col :span="6"><div class="k">F1</div><div class="v">{{ result.f1.toFixed(3) }}</div></el-col>
      </el-row>
      <div class="mt">TP {{ result.true_positives }} · FP {{ result.false_positives }} · TN {{ result.true_negatives }} · FN {{ result.false_negatives }} (N={{ result.support }})</div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { evaluateApi } from '@/api'
import type { EvaluateResponse } from '@/types/api'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const features = ref<number[][]>([])
const labels = ref<number[]>([])
const result = ref<EvaluateResponse | null>(null)
const error = ref<string | null>(null)

function parseCsv(text: string) {
  const rows = text.split(/\r?\n/).filter((l) => l.trim().length)
  const f: number[][] = []
  const y: number[] = []
  for (const line of rows) {
    const parts = line.split(',').map((s) => s.trim())
    if (parts.length < 2) continue
    const label = parseInt(parts[parts.length - 1], 10)
    if (Number.isNaN(label)) continue
    const nums = parts.slice(0, -1).map(Number)
    if (nums.some((n) => Number.isNaN(n))) continue
    f.push(nums)
    y.push(label)
  }
  return { f, y }
}

async function onFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  const text = await file.text()
  const { f, y } = parseCsv(text)
  features.value = f
  labels.value = y
  result.value = null
}

async function run() {
  if (!session.current) return
  error.value = null
  try {
    result.value = await evaluateApi.run(session.current.session_id, features.value, labels.value)
  } catch (e: any) { error.value = e.message }
}
</script>

<style scoped>
.k { color: #6b7280; font-size: 12px; }
.v { font-size: 22px; font-weight: 700; }
.mt { margin-top: 12px; }
.mb { margin-bottom: 12px; }
.muted { color: #6b7280; }
</style>
