"""세션 실시간 통계 API"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.common import StatsResponse
from api.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/{session_id}", response_model=StatsResponse)
async def get_stats(session_id: str):
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    stats = session.system.get_statistics()
    return StatsResponse(
        total_frames=int(stats.get("total_frames", 0)),
        total_detections=int(stats.get("total_detections", 0)),
        total_anomalies=int(stats.get("total_anomalies", 0)),
        total_runtime=float(stats.get("total_runtime", 0.0)),
        avg_fps=float(stats.get("avg_fps", 0.0)),
        avg_processing_time=float(stats.get("avg_processing_time", 0.0)),
        detection_rate=float(stats.get("detection_rate", 0.0)),
        anomaly_rate=float(stats.get("anomaly_rate", 0.0)),
        detector_stats=stats.get("detector_stats"),
        tracker_stats=stats.get("tracker_stats"),
        anomaly_detector_stats=stats.get("anomaly_detector_stats"),
    )


@router.post("/{session_id}/reset")
async def reset_stats(session_id: str):
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    session.system.reset_statistics()
    return {"reset": True}
