"""프레임 디코딩/결과 변환 헬퍼"""
from __future__ import annotations

import base64
from typing import Any, Dict, List

import cv2
import numpy as np


def decode_jpeg_base64(b64: str) -> np.ndarray:
    """base64 JPEG 문자열을 BGR numpy 배열로 디코딩"""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid JPEG payload")
    return frame


def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid JPEG payload")
    return frame


def result_to_response(
    session_id: str,
    frame_number: int,
    frame_shape: tuple,
    process_result: Dict[str, Any],
) -> Dict[str, Any]:
    """YOLOAnomalyDetectionSystem.process_frame() 결과를 API 응답으로 변환"""
    h, w = frame_shape[:2]
    detections: List[Dict[str, Any]] = []
    for person_id, result in process_result.get("anomaly_results", {}).items():
        center = result.get("center")
        if hasattr(center, "__iter__"):
            center = [float(center[0]), float(center[1])]
        else:
            center = [0.0, 0.0]
        detections.append({
            "person_id": int(person_id),
            "bbox": [float(v) for v in result["bbox"]],
            "center": center,
            "anomaly_score": float(result["anomaly_score"]),
            "is_anomaly": bool(result["is_anomaly"]),
            "confidence": float(result["confidence"]),
        })

    return {
        "session_id": session_id,
        "frame_number": int(frame_number),
        "frame_width": int(w),
        "frame_height": int(h),
        "processing_time_ms": float(process_result.get("processing_time", 0.0) * 1000),
        "total_detections": int(process_result.get("total_detections", 0)),
        "total_tracked": int(process_result.get("total_tracked", 0)),
        "detections": detections,
    }
