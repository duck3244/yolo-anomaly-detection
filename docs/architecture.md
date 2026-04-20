# Architecture

YOLO 기반 이상행동 검출 시스템의 전체 아키텍처 문서입니다. 단일 호스트에서 동작하는 개발/검증용 구조이며, Python 백엔드(FastAPI)와 Vue 3 프론트엔드로 분리되어 있습니다.

## 1. System Overview

- **목적**: 웹캠 또는 업로드된 비디오에서 YOLO로 사람을 검출하고, 추적·특징 추출·이상치 모델을 결합해 이상행동을 실시간 시각화한다.
- **배포 형태**: 로컬 단일 호스트. 백엔드는 uvicorn, 프론트엔드는 Vite dev server. 프록시를 통해 `/api`, `/ws` 를 백엔드로 포워딩한다.
- **주요 특성**:
  - 세션 단위 격리: 사용자/탭마다 독립적인 검출기·추적기·이상치 모델 인스턴스를 보유한다.
  - 스트리밍 추론: WebSocket 바이너리로 JPEG 프레임을 받아 결과 JSON을 반환한다.
  - In-memory Job Manager: 학습 작업은 `ThreadPoolExecutor` 기반 비동기 잡으로 실행, WebSocket으로 진행률을 구독한다.

## 2. Technology Stack

| Layer | Stack |
|-------|-------|
| Frontend | Vue 3 (Composition API), TypeScript, Vite, Pinia, Vue Router 4, Element Plus, Axios |
| Backend API | FastAPI, uvicorn, Pydantic v2, Starlette WebSocket |
| ML Pipeline | ultralytics YOLOv8, OpenCV, NumPy, scipy (Hungarian), scikit-learn (IsolationForest) |
| Storage | 파일 시스템 (모델 pkl, config.json, logs/, output/) |
| Async Primitives | `asyncio.Lock`, `threading.Lock`, `ThreadPoolExecutor`, `threading.Event` |

## 3. Top-Level Layout

```
yolo-anomaly-detection/
├── backend/
│   ├── run_api.py                # uvicorn entrypoint + lifespan
│   ├── main_system.py            # YOLOAnomalyDetectionSystem (ML 파이프라인 오케스트레이션)
│   ├── yolo_detector.py          # YOLO detector + Simple fallback
│   ├── person_tracker.py         # 헝가리안 기반 추적기
│   ├── feature_extractor.py      # 156-dim 특징 추출
│   ├── anomaly_detector.py       # IsolationForest 앙상블
│   ├── config.json               # 런타임 설정 (config_router로 편집)
│   └── api/
│       ├── routers/              # REST 엔드포인트
│       │   ├── sessions_router.py
│       │   ├── detect_router.py
│       │   ├── train_router.py
│       │   ├── models_router.py
│       │   ├── stats_router.py
│       │   ├── config_router.py
│       │   └── evaluate_router.py
│       ├── ws/                   # WebSocket 엔드포인트
│       │   ├── stream_ws.py      # /ws/stream (프레임 스트리밍)
│       │   ├── training_ws.py    # /ws/training (진행률 구독)
│       │   └── alerts_ws.py
│       ├── services/
│       │   ├── session_manager.py
│       │   └── job_manager.py
│       └── schemas/common.py     # Pydantic 모델
└── frontend/
    └── src/
        ├── main.ts, App.vue, router/
        ├── views/                # 라우트별 화면
        │   ├── DashboardView.vue
        │   ├── LiveDetectionView.vue
        │   ├── VideoAnalysisView.vue
        │   ├── TrainingView.vue
        │   ├── EvaluationView.vue
        │   └── ConfigView.vue
        ├── components/VideoCanvas.vue
        ├── composables/useWebSocket.ts
        ├── stores/               # Pinia (session, stats)
        ├── api/                  # Axios 클라이언트
        └── types/api.ts          # 백엔드 스키마 미러링
```

## 4. Layered Architecture

```
 ┌────────────────────────────────────────────────────────────┐
 │ Presentation  (Vue views / Element Plus / VideoCanvas)     │
 ├────────────────────────────────────────────────────────────┤
 │ Client State  (Pinia: session, stats · composables)        │
 ├────────────────────────────────────────────────────────────┤
 │ Transport     (Axios REST /api · WebSocket /ws)            │
 ├────────────────────────────────────────────────────────────┤
 │ API Layer     (FastAPI routers + WS endpoints + schemas)   │
 ├────────────────────────────────────────────────────────────┤
 │ Service Layer (SessionManager · JobManager)                │
 ├────────────────────────────────────────────────────────────┤
 │ ML Pipeline   (YOLOAnomalyDetectionSystem)                 │
 │   Detector → Tracker → FeatureExtractor → AnomalyDetector  │
 ├────────────────────────────────────────────────────────────┤
 │ Infra         (ultralytics · OpenCV · scikit-learn · FS)   │
 └────────────────────────────────────────────────────────────┘
```

## 5. Key Design Patterns

### 5.1 Session-per-Request
- `SessionManager.create()` 가 호출될 때마다 `YOLOAnomalyDetectionSystem` 인스턴스를 새로 만들고 `threading.Lock` 으로 감싼 `Session` 을 반환한다.
- `/ws/stream` 과 `/api/detect/frame` 은 `session_id` 를 필수로 받아 같은 세션의 검출기/추적기 상태를 공유한다.
- TTL (`session.ttl_sec`, 기본 1800s) 동안 접근이 없으면 `SessionManager.cleanup_loop` 가 자동 제거한다.

### 5.2 In-memory Job Manager with Subscribe/Notify
- 학습은 `JobManager.submit(target_fn)` 으로 `ThreadPoolExecutor` (max_workers=2) 에 등록한다.
- 각 잡은 `Job` 데이터클래스로 `state`, `progress`, `message`, `result/error` 를 추적한다.
- `/ws/training` 은 `subscribe(job_id)` 로 `threading.Event` 를 받고, 상태 변화 시 `_notify()` 가 이벤트를 셋하여 전파한다.
- 연결 유지를 위해 30초 `event.wait` 타임아웃 + 재진입 패턴을 사용한다.

### 5.3 Config Hot-Reload
- `config.json` 을 `config_router` 가 GET/PUT/PATCH 한다.
- PUT/PATCH는 허용 top-level 키 화이트리스트 (`system`, `tracking`, `feature_extraction`, `anomaly_detection`, `display`, `output`, `performance`) 로 검증한다.
- 저장 시 tmp → `os.replace` atomic write + `.bak` 백업.
- `YOLOAnomalyDetectionSystem.__init__` 이 `config.json` 을 다시 읽어 세션을 재생성하면 적용된다.

### 5.4 Backpressure on WebSocket
- 프론트가 JPEG 을 보낼 때 `ws.bufferedAmount > 256KB` 이면 해당 프레임을 스킵한다. 서버 처리보다 송신이 빠르면 드롭되도록 하여 레이턴시 누적을 방지한다.

## 6. Data Flows

### 6.1 Frame Streaming (Live / Video Analysis)
```
[Browser]  getUserMedia or <video>
   │ canvas.toBlob(jpeg, 0.7)
   ▼
[WS /ws/stream?session_id=…]
   │ binary JPEG
   ▼
[stream_ws]  session.lock 획득 → system.process_frame(ndarray)
   ▼
[main_system]  YOLO → tracker → features → anomaly scores
   │ FrameResult(JSON)
   ▼
[Browser]  VideoCanvas 에 bbox/score 오버레이
```

### 6.2 Training
```
POST /api/train {data_path, model_type}
   │  JobManager.submit(train_fn)
   ▼
[worker thread]  system.train(...) → progress_cb(%) → job_manager._notify()
                                                            │
[WS /ws/training?job_id=…]  event.wait → send JSON status  ◀┘
   ▼
[TrainingView.vue]  진행률 바 업데이트 + 완료 시 모델 메타 표시
```

### 6.3 Config Edit
```
ConfigView.vue → GET /api/config → 로컬 편집 → PUT /api/config
   │ _validate → atomic save (+ .bak)
   ▼
다음 세션 생성 시 새 설정이 로딩됨
```

### 6.4 Evaluation
```
POST /api/evaluate {dataset, model_id} → 블로킹 실행 (데이터셋 크기 작을 때)
   → precision/recall/f1 반환
```

## 7. Module Responsibilities

### Backend
- `run_api.py` — FastAPI 앱 생성, 라우터/WS 등록, `lifespan` 으로 SessionManager 정리 태스크 시작.
- `main_system.py:YOLOAnomalyDetectionSystem` — 검출기/추적기/특징추출기/이상치 모델 조립, `process_frame`, `train`, `save_model`, `load_model` 제공.
- `yolo_detector.py` — ultralytics YOLOv8 래퍼 (+ `SimpleDetector` 프레임차분 fallback). 모델 미존재 시 자동 다운로드.
- `person_tracker.py` — Hungarian assignment 기반 ID 매칭, 트랙 수명/스무딩 관리.
- `feature_extractor.py` — motion/spatial/shape/temporal/velocity 156-d 특징.
- `anomaly_detector.py` — IsolationForest 앙상블, 학습/예측/저장/로드.
- `api/services/session_manager.py` — dict 기반 세션 레지스트리 + TTL cleanup + singleton.
- `api/services/job_manager.py` — Job 레지스트리 + executor + subscribe/notify.
- `api/schemas/common.py` — 공통 Pydantic 응답/요청 모델.

### Frontend
- `views/*.vue` — 라우트별 화면, 각자 세션 + WS 라이프사이클 관리.
- `components/VideoCanvas.vue` — 비디오 요소 위에 bbox/텍스트 오버레이 (requestAnimationFrame).
- `composables/useWebSocket.ts` — 재연결/송수신/상태를 캡슐화한 WS 헬퍼.
- `stores/session.ts` — 현재 `SessionInfo` 와 `ensure()` (없으면 생성) 유틸.
- `stores/stats.ts` — 통계 폴링.
- `api/` — Axios 인스턴스 + REST 래퍼.

## 8. Operational Concerns

- **로그**: `backend/logs/` 아래 파일 로깅 + stdout. 모듈별 `logging.getLogger(__name__)`.
- **모델 아티팩트**: `backend/models/` 하위에 pkl/pt 저장. `.gitignore` 에 포함.
- **런타임 산출물**: `backend/output/`, `test_videos/` 도 git 제외.
- **브라우저 호환성**: 업로드 비디오는 H.264 권장 (테스트용 생성기는 avc1 → H264 → mp4v 순으로 fallback).
- **리소스 한계**: JobManager workers=2, uvicorn single process. 수평 확장/GPU 스케줄링은 범위 밖.

## 9. Extension Points

- YOLO 대체: `YOLODetector` 와 동일 인터페이스(`detect(frame) -> List[Detection]`)로 `MultiScaleYOLODetector`, `OptimizedYOLODetector`, `YOLOEnsembleDetector` 를 drop-in 교체.
- 추적기 고도화: `AdvancedPersonTracker` (trajectory smoothing).
- 특징 추가: `AdvancedFeatureExtractor` (interaction/context).
- 이상치 모델: `EnsembleAnomalyDetector`, `AdaptiveAnomalyDetector`.
- 새 UI: `views/` 에 추가 후 `router/index.ts` 등록.
