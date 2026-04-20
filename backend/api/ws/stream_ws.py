"""실시간 프레임 처리 WebSocket

클라이언트 → 서버: binary JPEG frame 또는 JSON {"jpeg_base64": "..."}
서버 → 클라이언트: JSON FrameResult
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.services.frame_processor import (
    decode_jpeg_base64,
    decode_jpeg_bytes,
    result_to_response,
)
from api.services.session_manager import get_session_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/stream")
async def stream_ws(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        await websocket.send_json({"error": "session not found"})
        await websocket.close(code=4404)
        return

    frame_number = 0
    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                try:
                    frame = decode_jpeg_bytes(msg["bytes"])
                except ValueError as e:
                    await websocket.send_json({"error": str(e)})
                    continue
            elif "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                    if payload.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    if "frame_number" in payload:
                        frame_number = int(payload["frame_number"])
                    frame = decode_jpeg_base64(payload["jpeg_base64"])
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    await websocket.send_json({"error": f"bad payload: {e}"})
                    continue
            else:
                continue

            with session.lock:
                result = session.system.process_frame(frame, frame_number=frame_number)
            response = result_to_response(session_id, frame_number, frame.shape, result)
            await websocket.send_json(response)
            frame_number += 1
    except WebSocketDisconnect:
        logger.info(f"stream ws disconnected: {session_id}")
    except RuntimeError as e:
        # Starlette raises when receive() is called after a disconnect frame
        if "disconnect" in str(e).lower():
            logger.info(f"stream ws closed: {session_id}")
        else:
            logger.exception(f"stream ws error: {e}")
    except Exception as e:
        logger.exception(f"stream ws error: {e}")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass
