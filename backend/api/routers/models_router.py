"""학습된 모델 파일 목록/로드/삭제 API"""
from __future__ import annotations

import logging
import os
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.schemas.common import ModelMeta
from api.services.session_manager import get_session_manager

router = APIRouter(prefix="/api/models", tags=["models"])
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models")


class LoadModelRequest(BaseModel):
    session_id: str
    name: str


def _safe_path(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid model name")
    return os.path.join(MODELS_DIR, name)


@router.get("", response_model=List[ModelMeta])
async def list_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    items: List[ModelMeta] = []
    for name in sorted(os.listdir(MODELS_DIR)):
        if not name.endswith(".pkl"):
            continue
        full = os.path.join(MODELS_DIR, name)
        if not os.path.isfile(full):
            continue
        st = os.stat(full)
        items.append(ModelMeta(
            name=name,
            path=full,
            size_bytes=st.st_size,
            modified_at=st.st_mtime,
        ))
    return items


@router.post("/load")
async def load_model(req: LoadModelRequest):
    sm = get_session_manager()
    try:
        session = sm.get(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    path = _safe_path(req.name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="model file not found")

    with session.lock:
        ok = session.system.load_model(path)
    if not ok:
        raise HTTPException(status_code=500, detail="model load failed")

    session.model_path = path
    return {"loaded": True, "path": path}


@router.delete("/{name}")
async def delete_model(name: str):
    path = _safe_path(name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="model file not found")
    try:
        os.unlink(path)
        info = path.replace(".pkl", "_info.json")
        if os.path.exists(info):
            os.unlink(info)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": name}
