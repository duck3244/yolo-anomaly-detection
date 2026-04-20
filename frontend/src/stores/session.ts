import { defineStore } from 'pinia'
import { ref } from 'vue'
import { sessionsApi } from '@/api'
import type { SessionInfo } from '@/types/api'

const STORAGE_KEY = 'yolo_anomaly.session_id'

export const useSessionStore = defineStore('session', () => {
  const current = ref<SessionInfo | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function ensure(device = 'cpu', modelPath = 'yolov8n.pt') {
    loading.value = true
    error.value = null
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        try {
          current.value = await sessionsApi.get(saved)
          return current.value
        } catch {
          localStorage.removeItem(STORAGE_KEY)
        }
      }
      current.value = await sessionsApi.create(device, modelPath)
      localStorage.setItem(STORAGE_KEY, current.value.session_id)
      return current.value
    } catch (e: any) {
      error.value = e?.message ?? String(e)
      throw e
    } finally {
      loading.value = false
    }
  }

  async function recreate(device = 'cpu', modelPath = 'yolov8n.pt') {
    if (current.value) {
      try { await sessionsApi.remove(current.value.session_id) } catch {}
    }
    localStorage.removeItem(STORAGE_KEY)
    current.value = null
    return ensure(device, modelPath)
  }

  async function refresh() {
    if (!current.value) return null
    current.value = await sessionsApi.get(current.value.session_id)
    return current.value
  }

  return { current, loading, error, ensure, recreate, refresh }
})
