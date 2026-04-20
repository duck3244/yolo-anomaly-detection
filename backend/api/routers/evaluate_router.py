"""모델 평가 API (라벨 데이터 → P/R/F1)"""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas.common import EvaluateRequest, EvaluateResponse
from api.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])
logger = logging.getLogger(__name__)


@router.post("/{session_id}", response_model=EvaluateResponse)
async def evaluate(session_id: str, req: EvaluateRequest):
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    ad = session.system.anomaly_detector
    if not getattr(ad, "is_trained", False):
        raise HTTPException(status_code=400, detail="anomaly detector is not trained")

    features = np.array(req.features, dtype=np.float32)
    labels = np.array(req.labels, dtype=np.int32)
    if features.ndim != 2 or len(features) != len(labels):
        raise HTTPException(status_code=400, detail="shape mismatch of features/labels")

    # AnomalyDetector.evaluate 가 있으면 사용
    if hasattr(ad, "evaluate"):
        try:
            metrics = ad.evaluate(features, labels)
        except (ValueError, RuntimeError) as e:
            logger.exception("evaluate failed")
            raise HTTPException(status_code=500, detail=str(e))
        if not metrics:
            raise HTTPException(
                status_code=422,
                detail="evaluate returned empty result; check feature dimension "
                       "matches the trained model",
            )
        return EvaluateResponse(**metrics)

    # 폴백: SimpleAnomalyDetector 에 대해 Z-score 기준 계산
    if ad.normal_mean is None or ad.normal_std is None:
        raise HTTPException(status_code=400, detail="normal statistics unavailable")
    z = np.abs((features - ad.normal_mean) / (ad.normal_std + 1e-6))
    scores = np.minimum(1.0, np.max(z, axis=1) / float(getattr(ad, "z_score_divisor", 3.0)))
    preds = (scores > float(getattr(ad, "z_score_threshold", 0.7))).astype(np.int32)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    support = int(len(labels))
    accuracy = (tp + tn) / support if support else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvaluateResponse(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        true_positives=tp, false_positives=fp,
        true_negatives=tn, false_negatives=fn, support=support,
    )
