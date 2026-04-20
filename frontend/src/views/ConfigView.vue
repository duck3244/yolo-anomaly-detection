<template>
  <div>
    <h2>설정 (config.json)</h2>
    <el-alert v-if="error" :title="error" type="error" show-icon class="mb" />
    <el-alert v-if="notice" :title="notice" type="success" show-icon class="mb" />
    <el-input v-model="text" type="textarea" :rows="22" />
    <div class="toolbar">
      <el-button @click="load">다시 불러오기</el-button>
      <el-button type="primary" @click="save" :disabled="!valid">저장</el-button>
      <span class="muted" v-if="!valid">JSON 파싱 오류</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { configApi } from '@/api'

const text = ref('{}')
const error = ref<string | null>(null)
const notice = ref<string | null>(null)

const valid = computed(() => {
  try { JSON.parse(text.value); return true } catch { return false }
})

async function load() {
  error.value = null; notice.value = null
  try { text.value = JSON.stringify(await configApi.get(), null, 2) }
  catch (e: any) { error.value = e.message }
}

async function save() {
  error.value = null; notice.value = null
  try {
    await configApi.put(JSON.parse(text.value))
    notice.value = '저장됨. 재시작해야 새 세션에 반영됩니다.'
  } catch (e: any) { error.value = e.message }
}

onMounted(load)
</script>

<style scoped>
.toolbar { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
.muted { color: #6b7280; }
.mb { margin-bottom: 10px; }
</style>
