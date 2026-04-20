# YOLO 이상행동 검출 시스템

YOLOv8 기반 실시간 이상행동 검출 시스템. **FastAPI 백엔드 + Vue 3 프론트엔드** 구조로 웹캠/비디오 분석, 모델 학습, 설정 편집, 평가까지 브라우저에서 수행한다. CLI 스크립트도 함께 제공한다.

- **입력**: 웹캠 스트림 · 업로드된 비디오 · CLI 파일 경로
- **파이프라인**: YOLOv8 → Hungarian 추적 → 156-dim 특징 추출 → IsolationForest 기반 이상치 검출
- **출력**: 프레임별 bbox · 추적 ID · anomaly score (WebSocket JSON)

자세한 설계는 [`docs/architecture.md`](docs/architecture.md) 와 [`docs/uml.md`](docs/uml.md) 참고.

---

## 1. 요구사항

- Python 3.9+ (3.10 권장)
- Node.js 18+ (프론트엔드 개발용)
- 4GB+ RAM, 선택적 CUDA GPU
- ffmpeg (업로드 비디오의 H.264 트랜스코딩에 권장)

## 2. 빠른 시작

### 2.1 백엔드

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_api.py --port 8000 --reload
```

- REST: `http://127.0.0.1:8000/api/*`
- WebSocket: `ws://127.0.0.1:8000/ws/*`
- 최초 실행 시 `yolov8n.pt` 가 자동 다운로드된다.

### 2.2 프론트엔드

```bash
cd frontend
npm install
npm run dev
```

- 개발 서버: `http://127.0.0.1:5173`
- `vite.config.ts` 가 `/api`, `/ws` 를 백엔드(8000)로 프록시한다.

### 2.3 브라우저에서 사용

1. **Dashboard** — 세션/잡 상태, 최근 통계 확인.
2. **Live Detection** — 웹캠을 WebSocket 으로 스트리밍, 실시간 bbox 오버레이.
3. **Video Analysis** — mp4 업로드 후 프레임 단위 분석. (H.264 코덱 권장)
4. **Training** — 학습 데이터 경로 지정 → 진행률 스트리밍 → 완료 시 모델 저장.
5. **Evaluation** — 저장된 모델 + 데이터셋으로 precision/recall/f1 계산.
6. **Config** — `config.json` 조회/편집 (top-level 키 화이트리스트 검증).

## 3. 프로젝트 구조

```
yolo-anomaly-detection/
├── backend/
│   ├── run_api.py              # uvicorn 엔트리포인트
│   ├── main_system.py          # YOLOAnomalyDetectionSystem (ML 오케스트레이션)
│   ├── yolo_detector.py        # YOLOv8 + SimpleDetector fallback
│   ├── person_tracker.py       # Hungarian 추적기
│   ├── feature_extractor.py    # 156-dim 특징
│   ├── anomaly_detector.py     # IsolationForest 앙상블
│   ├── config.json             # 런타임 설정
│   ├── create_test_video.py    # 테스트 비디오 생성 (H.264 fallback)
│   ├── full_mp4_test.py        # CLI 통합 시나리오
│   └── api/
│       ├── app.py              # FastAPI 앱 + lifespan
│       ├── routers/            # sessions·detect·train·models·stats·config·evaluate
│       ├── ws/                 # stream·training·alerts
│       ├── services/           # SessionManager, JobManager
│       └── schemas/common.py   # Pydantic 모델
├── frontend/
│   └── src/
│       ├── views/              # Dashboard/Live/Video/Training/Evaluation/Config
│       ├── components/VideoCanvas.vue
│       ├── composables/useWebSocket.ts
│       ├── stores/             # Pinia (session, stats)
│       └── api/                # Axios 클라이언트
└── docs/
    ├── architecture.md
    └── uml.md
```

## 4. 주요 API

| Method | Path | 용도 |
|--------|------|------|
| POST   | `/api/sessions` | 세션 생성 |
| GET    | `/api/sessions` | 세션 목록 |
| DELETE | `/api/sessions/{id}` | 세션 종료 |
| POST   | `/api/detect/frame` | 단일 프레임 검출 (multipart) |
| POST   | `/api/train` | 학습 잡 제출 |
| GET    | `/api/train/jobs/{id}` | 잡 상태 조회 |
| GET    | `/api/models` | 저장된 모델 목록 |
| POST   | `/api/evaluate` | 평가 실행 |
| GET/PUT/PATCH | `/api/config` | 설정 조회/저장 |
| GET    | `/api/stats/summary` | 통계 요약 |
| WS     | `/ws/stream?session_id=` | 프레임 스트리밍 |
| WS     | `/ws/training?job_id=` | 학습 진행률 구독 |

FastAPI 자동 문서: `http://127.0.0.1:8000/docs`.

## 5. 설정 (`backend/config.json`)

허용되는 top-level 키만 PUT/PATCH 로 저장된다: `system`, `tracking`, `feature_extraction`, `anomaly_detection`, `display`, `output`, `performance`.

```jsonc
{
  "system": {
    "device": "cpu",              // cpu | cuda | mps
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.5
  },
  "anomaly_detection": {
    "algorithm": "isolation_forest",
    "contamination": 0.1,
    "use_ensemble": false
  },
  "performance": {
    "frame_skip": 1,
    "roi_enabled": false
  }
}
```

저장은 tmp → `os.replace` atomic write 이며 이전 내용은 `config.json.bak` 으로 보관된다.

## 6. CLI 사용

브라우저 없이 스크립트만으로도 동일한 파이프라인을 구동할 수 있다.

```bash
# 테스트용 비디오 생성 (H.264 우선 → mp4v fallback)
python backend/create_test_video.py --type normal --duration 30

# 정상 영상으로 학습
python backend/main_system.py --mode train \
    --train_video test_videos/normal_behavior.mp4 \
    --model_save backend/models/my_model.pkl

# 분석 실행
python backend/main_system.py --mode video \
    --input test_videos/anomaly_behavior.mp4 \
    --model_load backend/models/my_model.pkl \
    --output result.mp4

# 실제 CCTV 데이터 통합 테스트
python backend/test_with_real_data.py

# 전체 시나리오 일괄 실행
python backend/full_mp4_test.py
```

## 7. 동작 특성

- **세션 격리** — `SessionManager` 가 요청별로 독립 파이프라인 인스턴스를 유지하고 TTL(기본 1800s) 이후 자동 정리한다.
- **비동기 학습** — `JobManager` 는 `ThreadPoolExecutor(max_workers=2)` 로 잡을 실행하고 `threading.Event` 기반 subscribe/notify 로 WS 에 진행률을 밀어낸다.
- **WS 백프레셔** — 프론트는 `ws.bufferedAmount > 256KB` 이면 프레임을 드롭하여 레이턴시 누적을 방지한다.
- **브라우저 호환** — 업로드 비디오는 H.264 필요. 테스트 생성기는 `avc1 → H264 → mp4v` 순으로 fallback.

## 8. 확장 포인트

| 레이어 | 대체 구현 |
|--------|-----------|
| 검출기 | `MultiScaleYOLODetector`, `OptimizedYOLODetector`, `YOLOEnsembleDetector` |
| 추적기 | `AdvancedPersonTracker` (trajectory smoothing) |
| 특징  | `AdvancedFeatureExtractor` (interaction/context) |
| 이상치 | `EnsembleAnomalyDetector`, `AdaptiveAnomalyDetector` |

동일 인터페이스를 유지하므로 `YOLOAnomalyDetectionSystem.__init__` 에서 인스턴스만 교체하면 된다.

## 9. 문제 해결

| 증상 | 조치 |
|------|------|
| `MediaError code=4` (브라우저 비디오 로드 실패) | `ffmpeg -i in.mp4 -c:v libx264 -c:a aac out.mp4` 로 재인코딩 |
| `timeout waiting for video` (두 번째 파일) | 재현 시 페이지 새로고침 — 정상 처리되면 무시 가능 |
| YOLO 최초 실행 실패 (가중치 없음) | 인터넷 연결 확인, `yolov8n.pt` 자동 다운로드 재시도 |
| `CUDA out of memory` | `config.json` 에서 `system.device="cpu"` 또는 `performance.frame_skip` 상향 |
| 설정 저장 거부 (422) | top-level 키가 화이트리스트에 있는지 확인 |

## 10. 라이선스

MIT. `LICENSE` 파일 참조.
