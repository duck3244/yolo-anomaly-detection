import { http } from './client'
import type { SessionInfo } from '@/types/api'

export const sessionsApi = {
  create(device = 'cpu', model_path = 'yolov8n.pt'): Promise<SessionInfo> {
    return http.post('/sessions', { device, model_path }).then((r) => r.data)
  },
  list(): Promise<SessionInfo[]> {
    return http.get('/sessions').then((r) => r.data)
  },
  get(id: string): Promise<SessionInfo> {
    return http.get(`/sessions/${id}`).then((r) => r.data)
  },
  remove(id: string): Promise<{ deleted: string }> {
    return http.delete(`/sessions/${id}`).then((r) => r.data)
  },
}
