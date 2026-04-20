import { defineStore } from 'pinia'
import { ref } from 'vue'
import { statsApi } from '@/api'
import type { StatsResponse } from '@/types/api'

export const useStatsStore = defineStore('stats', () => {
  const data = ref<StatsResponse | null>(null)
  const polling = ref<number | null>(null)
  const lastFetchedAt = ref<number | null>(null)

  let currentSessionId: string | null = null
  let currentInterval = 2000
  let visibilityHandler: (() => void) | null = null
  let inFlight = false

  async function load(sessionId: string) {
    if (inFlight) return data.value
    inFlight = true
    try {
      data.value = await statsApi.get(sessionId)
      lastFetchedAt.value = Date.now()
      return data.value
    } finally {
      inFlight = false
    }
  }

  function _tick() {
    if (!currentSessionId) return
    if (document.hidden) return
    load(currentSessionId).catch(() => { /* ignore transient errors */ })
  }

  function _install() {
    if (polling.value !== null) return
    polling.value = window.setInterval(_tick, currentInterval)
  }

  function _remove() {
    if (polling.value !== null) {
      clearInterval(polling.value)
      polling.value = null
    }
  }

  function startPolling(sessionId: string, intervalMs = 2000) {
    if (
      polling.value !== null &&
      currentSessionId === sessionId &&
      currentInterval === intervalMs
    ) return  // already polling this session at this rate
    stopPolling()
    currentSessionId = sessionId
    currentInterval = intervalMs
    _tick()  // fire once immediately so UI isn't blank
    _install()
    if (!visibilityHandler) {
      visibilityHandler = () => { if (!document.hidden) _tick() }
      document.addEventListener('visibilitychange', visibilityHandler)
    }
  }

  function stopPolling() {
    _remove()
    currentSessionId = null
    if (visibilityHandler) {
      document.removeEventListener('visibilitychange', visibilityHandler)
      visibilityHandler = null
    }
  }

  async function reset(sessionId: string) {
    await statsApi.reset(sessionId)
    await load(sessionId)
  }

  return { data, lastFetchedAt, load, startPolling, stopPolling, reset }
})
