import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'dashboard', component: () => import('@/views/DashboardView.vue') },
  { path: '/live', name: 'live', component: () => import('@/views/LiveDetectionView.vue') },
  { path: '/video', name: 'video', component: () => import('@/views/VideoAnalysisView.vue') },
  { path: '/train', name: 'train', component: () => import('@/views/TrainingView.vue') },
  { path: '/evaluate', name: 'evaluate', component: () => import('@/views/EvaluationView.vue') },
  { path: '/config', name: 'config', component: () => import('@/views/ConfigView.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
