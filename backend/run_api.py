"""로컬 개발용 uvicorn 엔트리포인트"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# backend 루트를 sys.path에 포함 (api.* import 용)
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="YOLO Anomaly Detection API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        app_dir=_BACKEND_DIR,
    )


if __name__ == "__main__":
    main()
