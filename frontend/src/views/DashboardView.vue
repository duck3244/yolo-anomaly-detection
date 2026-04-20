<template>
  <div>
    <h2>대시보드</h2>
    <el-row :gutter="16">
      <el-col :span="6"><el-card><div class="k">처리 프레임</div><div class="v">{{ stats.data?.total_frames ?? 0 }}</div></el-card></el-col>
      <el-col :span="6"><el-card><div class="k">검출</div><div class="v">{{ stats.data?.total_detections ?? 0 }}</div></el-card></el-col>
      <el-col :span="6"><el-card><div class="k">이상행동</div><div class="v danger">{{ stats.data?.total_anomalies ?? 0 }}</div></el-card></el-col>
      <el-col :span="6"><el-card><div class="k">평균 FPS</div><div class="v">{{ (stats.data?.avg_fps ?? 0).toFixed(1) }}</div></el-card></el-col>
    </el-row>
    <el-card class="mt">
      <div class="row">
        <strong>런타임</strong><span>{{ (stats.data?.total_runtime ?? 0).toFixed(1) }} s</span>
      </div>
      <div class="row">
        <strong>검출율 / 이상비율</strong>
        <span>{{ ((stats.data?.detection_rate ?? 0) * 100).toFixed(1) }}% / {{ ((stats.data?.anomaly_rate ?? 0) * 100).toFixed(1) }}%</span>
      </div>
      <div class="row"><el-button size="small" @click="onReset" :disabled="!session.current">통계 리셋</el-button></div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, watch } from 'vue'
import { useSessionStore } from '@/stores/session'
import { useStatsStore } from '@/stores/stats'

const session = useSessionStore()
const stats = useStatsStore()

function onReset() {
  if (session.current) stats.reset(session.current.session_id)
}

watch(
  () => session.current?.session_id,
  (sid) => { if (sid) stats.startPolling(sid, 2000) },
  { immediate: true },
)

onBeforeUnmount(() => stats.stopPolling())
</script>

<style scoped>
.k { color: #6b7280; font-size: 12px; }
.v { font-size: 28px; font-weight: 700; }
.v.danger { color: #ef4444; }
.row { display: flex; justify-content: space-between; padding: 6px 0; }
.mt { margin-top: 16px; }
</style>
