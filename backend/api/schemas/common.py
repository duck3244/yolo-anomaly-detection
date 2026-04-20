"""API 공통 Pydantic 스키마"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    last_active: float
    is_trained: bool
    model_path: Optional[str] = None


class SessionCreateRequest(BaseModel):
    device: str = Field(default="cpu", description="cpu | cuda")
    model_path: str = Field(default="yolov8n.pt")


class Detection(BaseModel):
    person_id: int
    bbox: List[float] = Field(..., description="[x1,y1,x2,y2] in frame coords")
    center: List[float]
    anomaly_score: float
    is_anomaly: bool
    confidence: float


class FrameResult(BaseModel):
    session_id: str
    frame_number: int
    frame_width: int
    frame_height: int
    processing_time_ms: float
    total_detections: int
    total_tracked: int
    detections: List[Detection]


class StatsResponse(BaseModel):
    total_frames: int
    total_detections: int
    total_anomalies: int
    total_runtime: float
    avg_fps: float
    avg_processing_time: float
    detection_rate: float
    anomaly_rate: float
    detector_stats: Optional[Dict[str, Any]] = None
    tracker_stats: Optional[Dict[str, Any]] = None
    anomaly_detector_stats: Optional[Dict[str, Any]] = None


class TrainRequest(BaseModel):
    max_frames: int = 1000
    model_save_name: str = Field(default="trained_model.pkl")


class JobStatus(BaseModel):
    job_id: str
    state: str = Field(..., description="pending | running | completed | failed")
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class ModelMeta(BaseModel):
    name: str
    path: str
    size_bytes: int
    modified_at: float


class EvaluateRequest(BaseModel):
    features: List[List[float]]
    labels: List[int]


class EvaluateResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    support: int
