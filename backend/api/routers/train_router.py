"""훈련 API (업로드 비디오 기반, 백그라운드 Job)"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas.common import JobStatus
from api.services.job_manager import Job, get_job_manager
from api.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/train", tags=["train"])
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models")


def _job_to_status(job: Job) -> JobStatus:
    return JobStatus(
        job_id=job.job_id,
        state=job.state,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post("", response_model=JobStatus)
async def start_training(
    session_id: str = Form(...),
    max_frames: int = Form(1000),
    model_save_name: str = Form("trained_model.pkl"),
    file: UploadFile = File(...),
):
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    # 업로드 파일을 임시 경로에 보관
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
    except Exception as e:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail=f"upload failed: {e}")

    video_path = tmp.name
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, model_save_name)

    jm = get_job_manager()

    def _run(job: Job):
        jm.update_progress(job, 0.05, "훈련 시작")
        try:
            with session.lock:
                ok = session.system.train_on_video(video_path, max_frames=max_frames)
                if not ok:
                    raise RuntimeError("훈련 실패 (데이터 부족 또는 처리 오류)")
                jm.update_progress(job, 0.85, "모델 저장 중")
                saved = session.system.save_model(model_path)
                if not saved:
                    raise RuntimeError("모델 저장 실패")
                metrics = getattr(session.system.anomaly_detector,
                                  "training_metrics", {}) or {}
            jm.update_progress(job, 1.0, "훈련 완료")
            return {
                "model_path": model_path,
                "training_metrics": metrics,
            }
        finally:
            try:
                os.unlink(video_path)
            except OSError:
                pass

    job = jm.submit(_run)
    return _job_to_status(job)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    jm = get_job_manager()
    job = jm.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_to_status(job)


@router.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    jm = get_job_manager()
    return [_job_to_status(j) for j in jm.list()]
