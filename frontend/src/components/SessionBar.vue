<template>
  <div class="bar">
    <span v-if="session.current" class="sid">
      Session: <code>{{ session.current.session_id.slice(0, 8) }}</code>
      <el-tag :type="session.current.is_trained ? 'success' : 'info'" size="small" class="ml">
        {{ session.current.is_trained ? 'trained' : 'untrained' }}
      </el-tag>
    </span>
    <span v-else class="muted">No session</span>
    <div class="spacer" />
    <el-select v-model="device" size="small" style="width: 100px" @change="onChange">
      <el-option label="CPU" value="cpu" />
      <el-option label="CUDA" value="cuda" />
    </el-select>
    <el-button size="small" @click="onEnsure" :loading="session.loading">Ensure</el-button>
    <el-button size="small" type="warning" @click="onRecreate">Recreate</el-button>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useSessionStore } from '@/stores/session'

const session = useSessionStore()
const device = ref<'cpu' | 'cuda'>('cpu')

async function onEnsure() { await session.ensure(device.value) }
async function onRecreate() { await session.recreate(device.value) }
function onChange() { /* next ensure uses new device */ }

onMounted(() => { session.ensure(device.value).catch(() => {}) })
</script>

<style scoped>
.bar { display: flex; width: 100%; align-items: center; gap: 8px; padding: 0 16px; }
.spacer { flex: 1; }
.muted { color: #9ca3af; }
.ml { margin-left: 6px; }
code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
</style>
