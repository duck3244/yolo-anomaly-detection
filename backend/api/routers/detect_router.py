"""단일 프레임 검출 API (멀티파트 또는 base64 JSON)"""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from api.schemas.common import FrameResult
from api.services.frame_processor import (
    decode_jpeg_base64,
    decode_jpeg_bytes,
    result_to_response,
)
from api.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/detect", tags=["detect"])
logger = logging.getLogger(__name__)


class FrameBase64Request(BaseModel):
    session_id: str
    frame_number: int = 0
    jpeg_base64: str


@router.post("/frame", response_model=FrameResult)
async def detect_frame(
    session_id: str = Form(...),
    frame_number: int = Form(0),
    file: UploadFile = File(...),
):
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    data = await file.read()
    try:
        frame = decode_jpeg_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with session.lock:
        result = session.system.process_frame(frame, frame_number=frame_number)

    return result_to_response(session_id, frame_number, frame.shape, result)


@router.post("/frame_base64", response_model=FrameResult)
async def detect_frame_base64(req: FrameBase64Request):
    sm = get_session_manager()
    try:
        session = sm.get(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    try:
        frame = decode_jpeg_base64(req.jpeg_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with session.lock:
        result = session.system.process_frame(frame, frame_number=req.frame_number)

    return result_to_response(req.session_id, req.frame_number, frame.shape, result)
