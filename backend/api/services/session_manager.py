"""세션별 YOLOAnomalyDetectionSystem 인스턴스 관리 (TTL 기반 정리)"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from main_system import YOLOAnomalyDetectionSystem

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    system: YOLOAnomalyDetectionSystem
    created_at: float
    last_active: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    model_path: Optional[str] = None


class SessionManager:
    """세션 생성/조회/만료 관리"""

    def __init__(self, ttl_seconds: int = 1800, config_path: str = "config.json"):
        self._sessions: Dict[str, Session] = {}
        self._global_lock = threading.Lock()
        self._ttl = ttl_seconds
        self._config_path = config_path
        self._cleanup_task: Optional[asyncio.Task] = None

    def create(self, device: str = "cpu", model_path: str = "yolov8n.pt") -> Session:
        session_id = uuid.uuid4().hex
        system = YOLOAnomalyDetectionSystem(
            config_path=self._config_path,
            model_path=model_path,
            device=device,
        )
        now = time.time()
        session = Session(
            session_id=session_id,
            system=system,
            created_at=now,
            last_active=now,
        )
        with self._global_lock:
            self._sessions[session_id] = session
        logger.info(f"session created: {session_id}")
        return session

    def get(self, session_id: str) -> Session:
        with self._global_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        session.last_active = time.time()
        return session

    def list(self):
        with self._global_lock:
            return list(self._sessions.values())

    def delete(self, session_id: str) -> bool:
        with self._global_lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        self._release(session)
        logger.info(f"session deleted: {session_id}")
        return True

    def _release(self, session: Session):
        try:
            detector = session.system.detector
            if hasattr(detector, "release"):
                detector.release()
        except Exception as e:
            logger.warning(f"session release failed: {e}")

    async def start_cleanup(self):
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self):
        if self._cleanup_task is None:
            return
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        self._cleanup_task = None

    async def _cleanup_loop(self):
        try:
            while True:
                await asyncio.sleep(60)
                self._expire_idle()
        except asyncio.CancelledError:
            raise

    def _expire_idle(self):
        now = time.time()
        expired = []
        with self._global_lock:
            for sid, session in list(self._sessions.items()):
                if now - session.last_active > self._ttl:
                    expired.append(sid)
                    self._sessions.pop(sid, None)
        for sid in expired:
            logger.info(f"session expired (TTL): {sid}")


_singleton: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _singleton
    if _singleton is None:
        _singleton = SessionManager()
    return _singleton
