export interface SessionInfo {
  session_id: string
  created_at: number
  last_active: number
  is_trained: boolean
  model_path?: string | null
}

export interface Detection {
  person_id: number
  bbox: [number, number, number, number]
  center: [number, number]
  anomaly_score: number
  is_anomaly: boolean
  confidence: number
}

export interface FrameResult {
  session_id: string
  frame_number: number
  frame_width: number
  frame_height: number
  processing_time_ms: number
  total_detections: number
  total_tracked: number
  detections: Detection[]
}

export interface StatsResponse {
  total_frames: number
  total_detections: number
  total_anomalies: number
  total_runtime: number
  avg_fps: number
  avg_processing_time: number
  detection_rate: number
  anomaly_rate: number
  detector_stats?: Record<string, unknown>
  tracker_stats?: Record<string, unknown>
  anomaly_detector_stats?: Record<string, unknown>
}

export interface JobStatus {
  job_id: string
  state: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  message?: string | null
  result?: Record<string, unknown> | null
  error?: string | null
  started_at?: number | null
  finished_at?: number | null
}

export interface ModelMeta {
  name: string
  path: string
  size_bytes: number
  modified_at: number
}

export interface EvaluateResponse {
  accuracy: number
  precision: number
  recall: number
  f1: number
  true_positives: number
  false_positives: number
  true_negatives: number
  false_negatives: number
  support: number
}

export type AppConfig = Record<string, any>
