"""단순 in-memory 백그라운드 작업 매니저 (로컬 단일 사용자용)"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    state: str = "pending"
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    log_buffer: List[str] = field(default_factory=list)


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._jobs: Dict[str, Job] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._subscribers: Dict[str, List[threading.Event]] = {}

    def submit(self, fn: Callable[[Job], Any]) -> Job:
        """fn(job) 형태의 callable 실행. job.progress/message를 내부에서 업데이트."""
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job

        def _runner():
            job.state = "running"
            job.started_at = time.time()
            self._notify(job_id)
            try:
                result = fn(job)
                job.result = result if isinstance(result, dict) else {"result": result}
                job.state = "completed"
                job.progress = 1.0
            except Exception as e:
                logger.exception(f"job {job_id} failed: {e}")
                job.error = str(e)
                job.state = "failed"
            finally:
                job.finished_at = time.time()
                self._notify(job_id)

        future = self._executor.submit(_runner)
        with self._lock:
            self._futures[job_id] = future
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def update_progress(self, job: Job, progress: float, message: Optional[str] = None):
        job.progress = max(0.0, min(1.0, progress))
        if message:
            job.message = message
            job.log_buffer.append(message)
        self._notify(job.job_id)

    def subscribe(self, job_id: str) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(event)
        return event

    def unsubscribe(self, job_id: str, event: threading.Event):
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if event in subs:
                subs.remove(event)

    def _notify(self, job_id: str):
        with self._lock:
            events = list(self._subscribers.get(job_id, []))
        for e in events:
            e.set()

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


_singleton: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    global _singleton
    if _singleton is None:
        _singleton = JobManager()
    return _singleton
