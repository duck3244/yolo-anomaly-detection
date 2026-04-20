"""이상행동 알림 WebSocket (세션의 alert_callback 에 훅)"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from api.services.session_manager import get_session_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()
    sm = get_session_manager()
    try:
        session = sm.get(session_id)
    except KeyError:
        await websocket.send_json({"error": "session not found"})
        await websocket.close(code=4404)
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    def _callback(alert_type, alert_data):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, {
                "type": alert_type,
                "data": alert_data,
            })
        except asyncio.QueueFull:
            logger.warning("alerts queue full; dropping alert")

    session.system.add_alert_callback(_callback)

    try:
        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(evt)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        logger.info(f"alerts ws disconnected: {session_id}")
    finally:
        try:
            if _callback in session.system.alert_callbacks:
                session.system.alert_callbacks.remove(_callback)
        except (ValueError, AttributeError):
            pass
