import { ref, onBeforeUnmount } from 'vue'

export interface WebSocketOptions {
  onMessage?: (data: any, raw: MessageEvent) => void
  onOpen?: () => void
  onClose?: (ev: CloseEvent) => void
  onError?: (ev: Event) => void
  reconnect?: boolean
  reconnectDelayMs?: number
}

export function useWebSocket(path: string, opts: WebSocketOptions = {}) {
  const socket = ref<WebSocket | null>(null)
  const connected = ref(false)
  const lastError = ref<string | null>(null)
  let reconnectTimer: number | null = null
  let manuallyClosed = false

  function url() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${location.host}${path}`
  }

  function connect() {
    manuallyClosed = false
    const ws = new WebSocket(url())
    socket.value = ws

    ws.onopen = () => {
      connected.value = true
      opts.onOpen?.()
    }
    ws.onmessage = (ev) => {
      let data: any = ev.data
      if (typeof data === 'string') {
        try { data = JSON.parse(data) } catch { /* keep string */ }
      }
      opts.onMessage?.(data, ev)
    }
    ws.onclose = (ev) => {
      connected.value = false
      opts.onClose?.(ev)
      if (!manuallyClosed && opts.reconnect) {
        reconnectTimer = window.setTimeout(connect, opts.reconnectDelayMs ?? 1500)
      }
    }
    ws.onerror = (ev) => {
      lastError.value = 'websocket error'
      opts.onError?.(ev)
    }
  }

  function send(data: string | ArrayBufferLike | Blob | ArrayBufferView) {
    if (socket.value && socket.value.readyState === WebSocket.OPEN) {
      socket.value.send(data)
    }
  }

  function sendJSON(obj: unknown) {
    send(JSON.stringify(obj))
  }

  function close() {
    manuallyClosed = true
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    socket.value?.close()
    socket.value = null
  }

  onBeforeUnmount(close)

  return { socket, connected, lastError, connect, send, sendJSON, close }
}
