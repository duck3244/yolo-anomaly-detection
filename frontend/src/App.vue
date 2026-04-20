<template>
  <el-container class="app-shell">
    <el-aside width="220px" class="app-aside">
      <div class="brand">YOLO Anomaly</div>
      <el-menu :router="true" :default-active="$route.path">
        <el-menu-item index="/">대시보드</el-menu-item>
        <el-menu-item index="/live">실시간 검출</el-menu-item>
        <el-menu-item index="/video">비디오 분석</el-menu-item>
        <el-menu-item index="/train">학습</el-menu-item>
        <el-menu-item index="/evaluate">평가</el-menu-item>
        <el-menu-item index="/config">설정</el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="app-header">
        <SessionBar />
      </el-header>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import SessionBar from '@/components/SessionBar.vue'
</script>

<style scoped>
.app-shell { height: 100vh; }
.app-aside { background: #1f2937; color: #fff; }
.brand { padding: 16px; font-weight: 700; font-size: 18px; }
.app-header { background: #fff; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; }
</style>
