# UML Diagrams

프로젝트의 주요 구조와 상호작용을 Mermaid 로 표현한 UML 문서입니다. GitHub/IntelliJ/대부분의 Markdown 뷰어에서 바로 렌더링됩니다.

## 1. Component Diagram (전체 구성)

```mermaid
flowchart LR
    subgraph Browser
        V[Vue Views]
        ST[Pinia Stores]
        WS1[useWebSocket]
        AX[Axios client]
    end

    subgraph Backend[FastAPI / uvicorn]
        direction TB
        R[Routers<br/>sessions · detect · train · models<br/>stats · config · evaluate]
        W[WS Endpoints<br/>stream · training · alerts]
        SM[SessionManager]
        JM[JobManager]
        MS[YOLOAnomalyDetectionSystem]
    end

    subgraph ML[ML Pipeline]
        YD[YOLODetector]
        PT[PersonTracker]
        FE[FeatureExtractor]
        AD[AnomalyDetector]
    end

    V --> ST
    V --> WS1
    V --> AX
    AX -- REST /api --> R
    WS1 -- WS /ws --> W
    R --> SM
    R --> JM
    W --> SM
    W --> JM
    SM --> MS
    JM --> MS
    MS --> YD
    MS --> PT
    MS --> FE
    MS --> AD
```

## 2. Backend Class Diagram

```mermaid
classDiagram
    class YOLOAnomalyDetectionSystem {
        +config: dict
        +detector: YOLODetector
        +tracker: PersonTracker
        +feature_extractor: FeatureExtractor
        +anomaly_detector: AnomalyDetector
        +is_trained: bool
        +process_frame(frame) FrameResult
        +train(data_path, progress_cb)
        +save_model(path)
        +load_model(path)
    }

    class YOLODetector {
        +model_path: str
        +device: str
        +confidence: float
        +logger: Logger
        +detect(frame) List~Detection~
        +_load_model()
    }
    class SimpleDetector {
        +prev_frame: ndarray
        +detect(frame) List~Detection~
    }
    class MultiScaleYOLODetector
    class OptimizedYOLODetector
    class YOLOEnsembleDetector

    class PersonTracker {
        +max_disappeared: int
        +update(detections) List~Track~
        -_hungarian_assign(...)
    }
    class AdvancedPersonTracker {
        +smooth_trajectory()
    }

    class FeatureExtractor {
        +extract(track_history) ndarray
        -_motion_features()
        -_spatial_features()
        -_shape_features()
        -_temporal_features()
        -_velocity_features()
    }
    class AdvancedFeatureExtractor {
        +extract_interaction()
        +extract_context()
    }

    class AnomalyDetector {
        +model: IsolationForest
        +threshold: float
        +fit(X)
        +predict(X) (score, is_anom)
        +save(path)
        +load(path)
    }
    class EnsembleAnomalyDetector
    class AdaptiveAnomalyDetector

    YOLOAnomalyDetectionSystem o-- YOLODetector
    YOLOAnomalyDetectionSystem o-- PersonTracker
    YOLOAnomalyDetectionSystem o-- FeatureExtractor
    YOLOAnomalyDetectionSystem o-- AnomalyDetector
    YOLODetector <|-- MultiScaleYOLODetector
    YOLODetector <|-- OptimizedYOLODetector
    YOLODetector <|-- YOLOEnsembleDetector
    PersonTracker <|-- AdvancedPersonTracker
    FeatureExtractor <|-- AdvancedFeatureExtractor
    AnomalyDetector <|-- EnsembleAnomalyDetector
    AnomalyDetector <|-- AdaptiveAnomalyDetector
```

## 3. Service Layer Class Diagram

```mermaid
classDiagram
    class Session {
        +session_id: str
        +created_at: float
        +last_access: float
        +ttl_sec: int
        +system: YOLOAnomalyDetectionSystem
        +lock: threading.Lock
        +touch()
    }

    class SessionManager {
        -sessions: dict~str,Session~
        -lock: asyncio.Lock
        +create() Session
        +get(sid) Session
        +list() List~Session~
        +delete(sid) bool
        +cleanup_loop()
    }

    class Job {
        +job_id: str
        +state: str  pending/running/completed/failed
        +progress: float
        +message: str
        +result: Any
        +error: str
        +created_at: float
    }

    class JobManager {
        -jobs: dict~str,Job~
        -executor: ThreadPoolExecutor
        -subscribers: dict~str,List~Event~~
        +submit(target_fn) Job
        +get(jid) Job
        +list() List~Job~
        +update_progress(jid, pct, msg)
        +subscribe(jid) Event
        +unsubscribe(jid, ev)
        -_notify(jid)
    }

    SessionManager "1" *-- "*" Session
    JobManager "1" *-- "*" Job
    Session --> YOLOAnomalyDetectionSystem
```

## 4. Frontend Class/Module Diagram

```mermaid
classDiagram
    class SessionStore {
        +current: SessionInfo
        +ensure()
        +reset()
    }
    class StatsStore {
        +summary: StatsResponse
        +refresh()
    }
    class useWebSocket {
        +socket: Ref~WebSocket~
        +connect()
        +send(data)
        +close()
    }
    class apiClient {
        +get(url) Promise
        +post(url, body) Promise
        +put(url, body) Promise
    }

    class DashboardView
    class LiveDetectionView
    class VideoAnalysisView
    class TrainingView
    class EvaluationView
    class ConfigView
    class VideoCanvas

    DashboardView --> StatsStore
    DashboardView --> apiClient
    LiveDetectionView --> SessionStore
    LiveDetectionView --> useWebSocket
    LiveDetectionView --> VideoCanvas
    VideoAnalysisView --> SessionStore
    VideoAnalysisView --> useWebSocket
    VideoAnalysisView --> VideoCanvas
    TrainingView --> apiClient
    TrainingView --> useWebSocket
    EvaluationView --> apiClient
    ConfigView --> apiClient
```

## 5. Sequence Diagram — Live Frame Streaming

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant V as LiveDetectionView
    participant S as SessionStore
    participant AX as Axios
    participant SR as sessions_router
    participant SM as SessionManager
    participant WS as /ws/stream
    participant MS as YOLOAnomalyDetectionSystem

    U->>V: "시작" 클릭
    V->>S: ensure()
    alt 세션 없음
        S->>AX: POST /api/sessions
        AX->>SR: create()
        SR->>SM: create()
        SM-->>SR: Session
        SR-->>AX: SessionInfo
        AX-->>S: SessionInfo
    end
    V->>WS: WS connect(?session_id)
    WS->>SM: get(sid)
    SM-->>WS: Session
    loop every ~100ms
        V->>V: canvas.toBlob(jpeg)
        V->>WS: send(binary)
        WS->>MS: process_frame(ndarray)
        MS-->>WS: FrameResult
        WS-->>V: JSON
        V->>V: VideoCanvas overlay bbox
    end
    U->>V: "정지"
    V->>WS: close()
```

## 6. Sequence Diagram — Training Job

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant T as TrainingView
    participant AX as Axios
    participant TR as train_router
    participant JM as JobManager
    participant W as worker thread
    participant MS as YOLOAnomalyDetectionSystem
    participant WT as /ws/training

    U->>T: 파라미터 입력 → 시작
    T->>AX: POST /api/train
    AX->>TR: start()
    TR->>JM: submit(train_fn)
    JM-->>TR: Job{id, pending}
    TR-->>AX: {job_id}
    AX-->>T: {job_id}
    T->>WT: WS connect(?job_id)
    WT->>JM: subscribe(job_id)
    JM-->>WT: Event

    par background
        JM->>W: executor.submit(train_fn)
        W->>MS: train(data_path, progress_cb)
        loop every batch
            MS-->>W: progress%
            W->>JM: update_progress()
            JM->>JM: _notify(job_id) → event.set
        end
        W->>JM: state=completed, result=meta
        JM->>JM: _notify(job_id)
    and streaming
        loop
            WT->>JM: event.wait(30s)
            JM-->>WT: set / timeout
            WT->>JM: get(job_id)
            JM-->>WT: Job snapshot
            WT-->>T: JSON status
            T->>T: 진행률 바 업데이트
        end
    end

    WT-->>T: state=completed
    T->>AX: GET /api/models
```

## 7. Sequence Diagram — Config Edit with Hot-Reload

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant C as ConfigView
    participant AX as Axios
    participant CR as config_router
    participant FS as Filesystem

    C->>AX: GET /api/config
    AX->>CR: get_config()
    CR->>FS: read config.json
    FS-->>CR: dict
    CR-->>AX: dict
    AX-->>C: dict

    U->>C: 편집 → 저장
    C->>AX: PUT /api/config {cfg}
    AX->>CR: put_config()
    CR->>CR: _validate(cfg) ▸ top-key whitelist
    alt invalid
        CR-->>AX: 400/422
    else valid
        CR->>FS: write tmp + os.replace(.bak, new)
        CR-->>AX: {saved:true}
    end
    Note over C,CR: 다음 세션 생성 시 새 config 적용
```

## 8. Sequence Diagram — Session TTL Cleanup

```mermaid
sequenceDiagram
    autonumber
    participant L as lifespan
    participant SM as SessionManager
    participant T as asyncio task

    L->>SM: start cleanup_loop()
    SM->>T: create_task(loop)
    loop every N sec
        T->>SM: scan sessions
        alt now - last_access > ttl
            SM->>SM: delete(sid)
        end
    end
    L->>SM: stop → cancel task
```

## 9. State Diagram — Job Lifecycle

```mermaid
stateDiagram-v2
    [*] --> pending: submit()
    pending --> running: executor picks up
    running --> running: update_progress()
    running --> completed: fn returns
    running --> failed: raises
    completed --> [*]
    failed --> [*]
```

## 10. State Diagram — Video Analysis UI

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> loading: 파일 선택 (onFile)
    loading --> ready: loadedmetadata/canplay
    loading --> error: video error(code=4)
    ready --> streaming: 시작 클릭 + WS open
    streaming --> streaming: 프레임 송수신
    streaming --> stopped: 정지 / video.ended
    streaming --> error: WS error
    stopped --> streaming: 재시작
    error --> idle: 새 파일 선택
```
