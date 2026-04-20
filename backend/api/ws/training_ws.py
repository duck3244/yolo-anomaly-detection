"""훈련 Job 진행률 WebSocket"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.services.job_manager import get_job_manager

router = APIRouter()
logger = logging.getLogger(__name__)


def _snapshot(job):
    return {
        "job_id": job.job_id,
        "state": job.state,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "result": job.result,
    }


@router.websocket("/ws/training")
async def training_ws(websocket: WebSocket, job_id: str = Query(...)):
    await websocket.accept()
    jm = get_job_manager()
    job = jm.get(job_id)
    if job is None:
        await websocket.send_json({"error": "job not found"})
        await websocket.close(code=4404)
        return

    event = jm.subscribe(job_id)
    try:
        await websocket.send_json(_snapshot(job))
        while True:
            await asyncio.get_running_loop().run_in_executor(None, event.wait, 30.0)
            event.clear()
            await websocket.send_json(_snapshot(job))
            if job.state in ("completed", "failed"):
                break
    except WebSocketDisconnect:
        logger.info(f"training ws disconnected: {job_id}")
    finally:
        jm.unsubscribe(job_id, event)
