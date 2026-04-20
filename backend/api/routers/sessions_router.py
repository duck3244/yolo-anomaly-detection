"""세션 라이프사이클 API"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from api.schemas.common import SessionCreateRequest, SessionInfo
from api.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _to_info(session) -> SessionInfo:
    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        last_active=session.last_active,
        is_trained=bool(getattr(session.system.anomaly_detector, "is_trained", False)),
        model_path=session.model_path,
    )


@router.post("", response_model=SessionInfo)
async def create_session(req: SessionCreateRequest):
    sm = get_session_manager()
    session = sm.create(device=req.device, model_path=req.model_path)
    session.model_path = req.model_path
    return _to_info(session)


@router.get("", response_model=List[SessionInfo])
async def list_sessions():
    sm = get_session_manager()
    return [_to_info(s) for s in sm.list()]


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return _to_info(session)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    sm = get_session_manager()
    ok = sm.delete(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="session not found")
    return {"deleted": session_id}
