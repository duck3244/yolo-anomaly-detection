import { http } from './client'
import type {
  AppConfig, EvaluateResponse, FrameResult, JobStatus, ModelMeta, StatsResponse,
} from '@/types/api'

export { sessionsApi } from './sessions'

export const configApi = {
  get(): Promise<AppConfig> { return http.get('/config').then((r) => r.data) },
  put(cfg: AppConfig): Promise<{ saved: boolean }> { return http.put('/config', cfg).then((r) => r.data) },
  patch(partial: Partial<AppConfig>): Promise<AppConfig> { return http.patch('/config', partial).then((r) => r.data) },
}

export const detectApi = {
  frame(sessionId: string, frameNumber: number, jpegBlob: Blob): Promise<FrameResult> {
    const form = new FormData()
    form.append('session_id', sessionId)
    form.append('frame_number', String(frameNumber))
    form.append('file', jpegBlob, 'frame.jpg')
    return http.post('/detect/frame', form, { headers: { 'Content-Type': 'multipart/form-data' } }).then((r) => r.data)
  },
  frameBase64(sessionId: string, frameNumber: number, base64: string): Promise<FrameResult> {
    return http.post('/detect/frame_base64', {
      session_id: sessionId, frame_number: frameNumber, jpeg_base64: base64,
    }).then((r) => r.data)
  },
}

export const trainApi = {
  start(sessionId: string, file: File, maxFrames = 1000, modelSaveName = 'trained_model.pkl'): Promise<JobStatus> {
    const form = new FormData()
    form.append('session_id', sessionId)
    form.append('max_frames', String(maxFrames))
    form.append('model_save_name', modelSaveName)
    form.append('file', file)
    return http.post('/train', form, { headers: { 'Content-Type': 'multipart/form-data' } }).then((r) => r.data)
  },
  job(id: string): Promise<JobStatus> { return http.get(`/train/jobs/${id}`).then((r) => r.data) },
  jobs(): Promise<JobStatus[]> { return http.get('/train/jobs').then((r) => r.data) },
}

export const modelsApi = {
  list(): Promise<ModelMeta[]> { return http.get('/models').then((r) => r.data) },
  load(sessionId: string, name: string): Promise<{ loaded: boolean; path: string }> {
    return http.post('/models/load', { session_id: sessionId, name }).then((r) => r.data)
  },
  remove(name: string): Promise<{ deleted: string }> {
    return http.delete(`/models/${encodeURIComponent(name)}`).then((r) => r.data)
  },
}

export const statsApi = {
  get(sessionId: string): Promise<StatsResponse> { return http.get(`/stats/${sessionId}`).then((r) => r.data) },
  reset(sessionId: string): Promise<{ reset: boolean }> { return http.post(`/stats/${sessionId}/reset`).then((r) => r.data) },
}

export const evaluateApi = {
  run(sessionId: string, features: number[][], labels: number[]): Promise<EvaluateResponse> {
    return http.post(`/evaluate/${sessionId}`, { features, labels }).then((r) => r.data)
  },
}
