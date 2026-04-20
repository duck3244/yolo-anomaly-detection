"""config.json 조회/수정 API"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/config", tags=["config"])
logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config.json")

_ALLOWED_TOP_KEYS = {
    "system", "tracking", "feature_extraction", "anomaly_detection",
    "display", "output", "performance",
}


def _load() -> Dict[str, Any]:
    if not os.path.exists(_CONFIG_PATH):
        raise HTTPException(status_code=404, detail="config.json not found")
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(cfg: Dict[str, Any]):
    # atomic write via temp + rename, keep .bak of previous content
    dir_ = os.path.dirname(_CONFIG_PATH)
    tmp = _CONFIG_PATH + ".tmp"
    bak = _CONFIG_PATH + ".bak"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    if os.path.exists(_CONFIG_PATH):
        try:
            os.replace(_CONFIG_PATH, bak)
        except OSError:
            pass
    os.replace(tmp, _CONFIG_PATH)


def _validate(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    if not cfg:
        raise HTTPException(status_code=400, detail="body is empty")
    unknown = [k for k in cfg if k not in _ALLOWED_TOP_KEYS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown top-level key(s): {unknown}; "
                   f"allowed: {sorted(_ALLOWED_TOP_KEYS)}",
        )
    for k, v in cfg.items():
        if not isinstance(v, dict):
            raise HTTPException(
                status_code=422,
                detail=f"value of '{k}' must be an object",
            )


@router.get("")
async def get_config():
    return _load()


@router.put("")
async def put_config(cfg: Dict[str, Any]):
    _validate(cfg)
    try:
        _save(cfg)
    except OSError as e:
        logger.exception("config save failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"saved": True, "path": _CONFIG_PATH}


@router.patch("")
async def patch_config(partial: Dict[str, Any]):
    _validate(partial)
    cfg = _load()
    for k, v in partial.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    _save(cfg)
    return cfg
