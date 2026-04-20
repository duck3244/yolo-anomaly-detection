"""FastAPI 애플리케이션 엔트리 (로컬 단일 호스트용)"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    config_router,
    detect_router,
    evaluate_router,
    models_router,
    sessions_router,
    stats_router,
    train_router,
)
from api.services.session_manager import get_session_manager
from api.ws import alerts_ws, stream_ws, training_ws

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sm = get_session_manager()
    await sm.start_cleanup()
    logger.info("API startup complete")
    try:
        yield
    finally:
        await sm.stop_cleanup()
        logger.info("API shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="YOLO Anomaly Detection API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sessions_router.router)
    app.include_router(config_router.router)
    app.include_router(detect_router.router)
    app.include_router(train_router.router)
    app.include_router(models_router.router)
    app.include_router(stats_router.router)
    app.include_router(evaluate_router.router)

    app.include_router(stream_ws.router)
    app.include_router(alerts_ws.router)
    app.include_router(training_ws.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": app.version}

    return app


app = create_app()
