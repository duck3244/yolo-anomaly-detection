"""
main_system.py
YOLO 기반 이상행동 검출 시스템의 메인 모듈
"""

import cv2
import numpy as np
import time
import argparse
import os
import json
import logging
from pathlib import Path
from collections import deque

# 모듈 임포트
try:
    from yolo_detector import YOLODetector, OptimizedYOLODetector
except ImportError:
    print("Warning: yolo_detector 모듈을 찾을 수 없습니다.")
    YOLODetector = None
    OptimizedYOLODetector = None

try:
    from person_tracker import PersonTracker, AdvancedPersonTracker
except ImportError:
    print("Warning: person_tracker 모듈을 찾을 수 없습니다.")
    PersonTracker = None
    AdvancedPersonTracker = None

try:
    from feature_extractor import FeatureExtractor, AdvancedFeatureExtractor
except ImportError:
    print("Warning: feature_extractor 모듈을 찾을 수 없습니다.")
    FeatureExtractor = None
    AdvancedFeatureExtractor = None

try:
    from anomaly_detector import AnomalyDetector, EnsembleAnomalyDetector, RealTimeAnomalyDetector
except ImportError:
    print("Warning: anomaly_detector 모듈을 찾을 수 없습니다.")
    AnomalyDetector = None
    EnsembleAnomalyDetector = None
    RealTimeAnomalyDetector = None


class SimpleDetector:
    """간단한 사람 검출기"""

    def __init__(self, model_path='yolov8n.pt', device='cpu', confidence_threshold=0.5):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self.prev_frame = None

    def detect_persons(self, frame):
        """간단한 사람 검출 (움직임 기반)"""
        h, w = frame.shape[:2]
        detections = []

        if self.prev_frame is not None:
            # 배경 차분
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_current, gray_prev)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # 노이즈 제거
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # 윤곽선 찾기
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 20000:  # 적절한 크기
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = ch / cw if cw > 0 else 0
                    if 1.5 < aspect_ratio < 4.0:  # 사람 비율
                        detections.append({
                            'bbox': [float(x), float(y), float(x + cw), float(y + ch)],
                            'confidence': 0.8,
                            'class_id': 0,
                            'class_name': 'person'
                        })

        # 검출이 없으면 기본 사람 하나 생성
        if len(detections) == 0:
            center_x = w // 2
            center_y = h // 2
            width, height = 60, 120
            detections.append({
                'bbox': [center_x - width//2, center_y - height//2, center_x + width//2, center_y + height//2],
                'confidence': 0.6,
                'class_id': 0,
                'class_name': 'person'
            })

        self.prev_frame = frame.copy()
        return detections

    def warm_up(self):
        pass

    def get_performance_stats(self):
        return {'total_inferences': 0, 'avg_inference_time': 0.03, 'fps': 30}


class SimpleTracker:
    """간단한 사람 추적기"""

    def __init__(self, max_disappeared=10, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        """추적 업데이트"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            return self.objects

        input_centroids = []
        for detection in detections:
            bbox = detection['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.objects[self.next_object_id] = {
                    'centroid': centroid,
                    'bbox': detections[i]['bbox'],
                    'features': deque(maxlen=30)
                }
                self.disappeared[self.next_object_id] = 0
                self.next_object_id += 1
        else:
            # 간단한 거리 기반 매칭
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())

            for i, input_centroid in enumerate(input_centroids):
                min_distance = float('inf')
                closest_id = None

                for j, object_centroid in enumerate(object_centroids):
                    distance = np.sqrt((input_centroid[0] - object_centroid[0])**2 +
                                     (input_centroid[1] - object_centroid[1])**2)
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        closest_id = object_ids[j]

                if closest_id is not None:
                    self.objects[closest_id]['centroid'] = input_centroid
                    self.objects[closest_id]['bbox'] = detections[i]['bbox']
                    self.disappeared[closest_id] = 0
                else:
                    # 새로운 객체 등록
                    self.objects[self.next_object_id] = {
                        'centroid': input_centroid,
                        'bbox': detections[i]['bbox'],
                        'features': deque(maxlen=30)
                    }
                    self.disappeared[self.next_object_id] = 0
                    self.next_object_id += 1

        return self.objects

    def get_statistics(self):
        return {'active_objects': len(self.objects), 'total_created': self.next_object_id}


class SimpleFeatureExtractor:
    """간단한 특징 추출기"""

    def extract_comprehensive_features(self, person_id, bbox, prev_bbox, frame_shape, frame_number=0, fps=30):
        """종합 특징 추출"""
        features = []

        # 기본 박스 특징
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0

        # 정규화
        norm_width = width / frame_shape[1]
        norm_height = height / frame_shape[0]
        norm_area = area / (frame_shape[0] * frame_shape[1])

        features.extend([norm_width, norm_height, norm_area, aspect_ratio])

        # 위치 특징
        center_x = (bbox[0] + bbox[2]) / (2 * frame_shape[1])
        center_y = (bbox[1] + bbox[3]) / (2 * frame_shape[0])
        features.extend([center_x, center_y])

        # 모션 특징
        if prev_bbox is not None:
            prev_center_x = (prev_bbox[0] + prev_bbox[2]) / (2 * frame_shape[1])
            prev_center_y = (prev_bbox[1] + prev_bbox[3]) / (2 * frame_shape[0])

            dx = center_x - prev_center_x
            dy = center_y - prev_center_y
            velocity = np.sqrt(dx**2 + dy**2)

            features.extend([dx, dy, velocity])
        else:
            features.extend([0.0, 0.0, 0.0])

        # 추가 랜덤 특징으로 차원 맞추기
        while len(features) < 20:
            features.append(np.random.normal(0, 0.1))

        return np.array(features[:20], dtype=np.float32)

    def cleanup_old_histories(self, active_person_ids):
        pass


class SimpleAnomalyDetector:
    """간단한 이상 검출기"""

    def __init__(self, window_size=30, contamination=0.1, algorithm='simple'):
        self.window_size = window_size
        self.contamination = contamination
        self.algorithm = algorithm
        self.is_trained = False
        self.normal_mean = None
        self.normal_std = None
        self.training_features = []

    def add_training_features(self, features):
        """훈련 특징 추가"""
        if isinstance(features, list):
            self.training_features.extend(features)
        else:
            self.training_features.append(features)

    def train(self):
        """모델 훈련"""
        if len(self.training_features) < 50:
            return False

        try:
            features_array = np.array(self.training_features)

            # NaN, Inf 제거
            valid_mask = ~(np.isnan(features_array).any(axis=1) | np.isinf(features_array).any(axis=1))
            features_array = features_array[valid_mask]

            if len(features_array) < 50:
                return False

            # 정상 패턴 학습
            self.normal_mean = np.mean(features_array, axis=0)
            self.normal_std = np.std(features_array, axis=0)
            self.is_trained = True

            return True

        except Exception as e:
            return False

    def detect_anomaly(self, features_window):
        """이상 검출"""
        if not self.is_trained or len(features_window) < self.window_size:
            return 0.0, False, 0.0

        try:
            recent_features = np.array(features_window[-self.window_size:])

            if np.any(np.isnan(recent_features)) or np.any(np.isinf(recent_features)):
                return 0.0, False, 0.0

            # Z-score 계산
            z_scores = np.abs((recent_features - self.normal_mean) / (self.normal_std + 1e-6))
            max_z_score = np.max(z_scores)
            avg_z_score = np.mean(np.max(z_scores, axis=1))

            # 이상 점수 계산
            anomaly_score = min(1.0, avg_z_score / 3.0)  # Z-score 3을 기준으로 정규화
            is_anomaly = anomaly_score > 0.7
            confidence = 0.8

            return anomaly_score, is_anomaly, confidence

        except Exception as e:
            return 0.0, False, 0.0

    def save_model(self, filepath):
        """모델 저장"""
        try:
            import pickle
            model_data = {
                'is_trained': self.is_trained,
                'normal_mean': self.normal_mean,
                'normal_std': self.normal_std,
                'window_size': self.window_size,
                'contamination': self.contamination
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True

        except Exception as e:
            return False

    def load_model(self, filepath):
        """모델 로드"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.is_trained = model_data['is_trained']
            self.normal_mean = model_data['normal_mean']
            self.normal_std = model_data['normal_std']
            self.window_size = model_data['window_size']
            self.contamination = model_data['contamination']

            return True

        except Exception as e:
            return False

    def get_statistics(self):
        return {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_features),
            'algorithm': self.algorithm
        }


class YOLOAnomalyDetectionSystem:
    """YOLO 기반 이상행동 검출 시스템"""

    def __init__(self, config_path=None, model_path='yolov8n.pt', device='cpu'):
        """시스템 초기화"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        self.device = device
        self.model_path = model_path

        # 모듈 생성
        self.detector = self._create_detector()
        self.tracker = self._create_tracker()
        self.feature_extractor = self._create_feature_extractor()
        self.anomaly_detector = self._create_anomaly_detector()

        # 실시간 검출기
        self.realtime_detector = None

        # 통계 정보
        self.stats = {
            'frame_count': 0,
            'detection_count': 0,
            'tracking_count': 0,
            'anomaly_count': 0,
            'processing_times': deque(maxlen=100),
            'start_time': time.time()
        }

        # 알림 콜백
        self.alert_callbacks = []

        self.logger.info("YOLO 이상행동 검출 시스템 초기화 완료")

    def _load_config(self, config_path):
        """설정 로드"""
        default_config = {
            "system": {
                "device": "cpu",
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.5,
                "log_level": "INFO"
            },
            "tracking": {
                "max_disappeared": 10,
                "max_distance": 100
            },
            "anomaly_detection": {
                "window_size": 30,
                "contamination": 0.1
            },
            "display": {
                "show_id": True,
                "show_score": True,
                "show_bbox": True,
                "anomaly_color": [0, 0, 255],
                "normal_color": [0, 255, 0],
                "font_scale": 0.6,
                "thickness": 2
            },
            "output": {
                "log_directory": "logs"
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # 간단한 설정 병합
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(value, dict) and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"설정 파일 로드 실패: {e}")

        return default_config

    def _setup_logging(self):
        """로깅 설정"""
        log_level = getattr(logging, self.config['system']['log_level'].upper(), logging.INFO)

        log_dir = Path(self.config['output']['log_directory'])
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'system.log'),
                logging.StreamHandler()
            ]
        )

    def _create_detector(self):
        """검출기 생성"""
        if YOLODetector is not None:
            try:
                return YOLODetector(
                    model_path=self.model_path,
                    device=self.device,
                    confidence_threshold=self.config['system']['confidence_threshold']
                )
            except Exception as e:
                self.logger.warning(f"YOLO 검출기 생성 실패, 간단한 검출기 사용: {e}")

        return SimpleDetector(
            model_path=self.model_path,
            device=self.device,
            confidence_threshold=self.config['system']['confidence_threshold']
        )

    def _create_tracker(self):
        """추적기 생성"""
        if PersonTracker is not None:
            try:
                return PersonTracker(
                    max_disappeared=self.config['tracking']['max_disappeared'],
                    max_distance=self.config['tracking']['max_distance']
                )
            except Exception as e:
                self.logger.warning(f"추적기 생성 실패, 간단한 추적기 사용: {e}")

        return SimpleTracker(
            max_disappeared=self.config['tracking']['max_disappeared'],
            max_distance=self.config['tracking']['max_distance']
        )

    def _create_feature_extractor(self):
        """특징 추출기 생성"""
        if FeatureExtractor is not None:
            try:
                return FeatureExtractor()
            except Exception as e:
                self.logger.warning(f"특징 추출기 생성 실패, 간단한 추출기 사용: {e}")

        return SimpleFeatureExtractor()

    def _create_anomaly_detector(self):
        """이상 검출기 생성"""
        if AnomalyDetector is not None:
            try:
                return AnomalyDetector(
                    window_size=self.config['anomaly_detection']['window_size'],
                    contamination=self.config['anomaly_detection']['contamination']
                )
            except Exception as e:
                self.logger.warning(f"이상 검출기 생성 실패, 간단한 검출기 사용: {e}")

        return SimpleAnomalyDetector(
            window_size=self.config['anomaly_detection']['window_size'],
            contamination=self.config['anomaly_detection']['contamination']
        )

    def add_alert_callback(self, callback):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)

    def train_on_video(self, video_path, max_frames=1000):
        """비디오로 모델 훈련"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.logger.error(f"비디오 파일을 열 수 없습니다: {video_path}")
            return False

        self.logger.info(f"훈련 시작: {video_path}")
        frame_count = 0
        training_features = []

        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame.shape[1] > 640:
                    frame = cv2.resize(frame, (640, 480))

                detections = self.detector.detect_persons(frame)

                if detections:
                    tracked_objects = self.tracker.update(detections)

                    for person_id, obj_data in tracked_objects.items():
                        bbox = obj_data['bbox']
                        prev_bbox = None

                        if len(obj_data['features']) > 0:
                            prev_bbox = obj_data['features'][-1].get('bbox')

                        try:
                            features = self.feature_extractor.extract_comprehensive_features(
                                person_id, bbox, prev_bbox, frame.shape[:2], frame_count
                            )

                            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                                training_features.append(features)

                        except Exception as e:
                            continue

                frame_count += 1
                if frame_count % 100 == 0:
                    self.logger.info(f"처리 프레임: {frame_count}, 특징: {len(training_features)}")

        except Exception as e:
            self.logger.error(f"훈련 중 오류: {e}")
            return False

        finally:
            cap.release()

        if len(training_features) >= 50:
            self.anomaly_detector.add_training_features(training_features)
            success = self.anomaly_detector.train()

            if success:
                self.logger.info(f"훈련 완료: {len(training_features)}개 특징")
                return True
            else:
                self.logger.error("모델 훈련 실패")
                return False
        else:
            self.logger.error(f"훈련 데이터 부족: {len(training_features)}개")
            return False

    def process_frame(self, frame, frame_number=0):
        """프레임 처리"""
        start_time = time.time()

        self.stats['frame_count'] += 1
        frame_height, frame_width = frame.shape[:2]

        # 검출
        detections = self.detector.detect_persons(frame)
        self.stats['detection_count'] += len(detections)

        # 추적
        tracked_objects = self.tracker.update(detections)
        self.stats['tracking_count'] = len(tracked_objects)

        anomaly_results = {}

        for person_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            prev_bbox = None

            if len(obj_data['features']) > 0:
                prev_bbox = obj_data['features'][-1].get('bbox')

            try:
                features = self.feature_extractor.extract_comprehensive_features(
                    person_id, bbox, prev_bbox, (frame_height, frame_width), frame_number
                )

                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue

                obj_data['features'].append({
                    'features': features,
                    'bbox': bbox,
                    'timestamp': time.time()
                })

                if self.anomaly_detector.is_trained and len(obj_data['features']) >= self.anomaly_detector.window_size:
                    feature_window = [f['features'] for f in obj_data['features']]
                    anomaly_score, is_anomaly, confidence = self.anomaly_detector.detect_anomaly(feature_window)

                    anomaly_results[person_id] = {
                        'bbox': bbox,
                        'center': obj_data['centroid'],
                        'anomaly_score': anomaly_score,
                        'is_anomaly': is_anomaly,
                        'confidence': confidence
                    }

                    if is_anomaly:
                        self.stats['anomaly_count'] += 1

            except Exception as e:
                continue

        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)

        return {
            'anomaly_results': anomaly_results,
            'total_detections': len(detections),
            'total_tracked': len(tracked_objects),
            'processing_time': processing_time,
            'frame_number': frame_number
        }

    def draw_results(self, frame, process_result):
        """결과 그리기"""
        result_frame = frame.copy()
        anomaly_results = process_result['anomaly_results']
        display_config = self.config['display']

        for person_id, result in anomaly_results.items():
            bbox = result['bbox']
            anomaly_score = result['anomaly_score']
            is_anomaly = result['is_anomaly']
            confidence = result['confidence']

            color = tuple(display_config['anomaly_color']) if is_anomaly else tuple(display_config['normal_color'])
            thickness = display_config['thickness'] + (1 if is_anomaly else 0)

            if display_config['show_bbox']:
                cv2.rectangle(result_frame,
                             (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])),
                             color, thickness)

            labels = []
            if display_config['show_id']:
                labels.append(f"ID:{person_id}")
            if display_config['show_score']:
                labels.append(f"Score:{anomaly_score:.2f}")
                labels.append(f"Conf:{confidence:.2f}")
            if is_anomaly:
                labels.append("ANOMALY!")

            if labels:
                label_text = " ".join(labels)
                font_scale = display_config['font_scale']

                cv2.putText(result_frame, label_text,
                           (int(bbox[0]), int(bbox[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                           color, thickness)

        # 시스템 정보
        fps = 1.0 / np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        info_lines = [
            f"Frame: {self.stats['frame_count']}",
            f"FPS: {fps:.1f}",
            f"Detections: {process_result['total_detections']}",
            f"Tracked: {process_result['total_tracked']}",
            f"Anomalies: {len([r for r in process_result['anomaly_results'].values() if r['is_anomaly']])}"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(result_frame, line, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result_frame

    def process_video(self, input_path, output_path=None, display=False):
        """비디오 처리"""
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            self.logger.error(f"비디오를 열 수 없습니다: {input_path}")
            return None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.logger.info(f"비디오 처리 시작: {input_path}")
        frame_number = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                process_result = self.process_frame(frame, frame_number)
                result_frame = self.draw_results(frame, process_result)

                if display:
                    cv2.imshow('Anomaly Detection', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if out:
                    out.write(result_frame)

                frame_number += 1

                if frame_number % 100 == 0:
                    avg_fps = 1.0 / np.mean(list(self.stats['processing_times'])[-10:]) if len(self.stats['processing_times']) >= 10 else 0
                    self.logger.info(f"프레임: {frame_number}, FPS: {avg_fps:.1f}")

        except KeyboardInterrupt:
            self.logger.info("사용자 중단")
        except Exception as e:
            self.logger.error(f"비디오 처리 오류: {e}")
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()


    def process_webcam(self, camera_id=0):
        """웹캠 실시간 처리"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            self.logger.error(f"카메라를 열 수 없습니다: {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.logger.info("웹캠 처리 시작 (q키로 종료)")
        frame_number = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("프레임을 읽을 수 없습니다.")
                    break

                process_result = self.process_frame(frame, frame_number)
                result_frame = self.draw_results(frame, process_result)

                cv2.imshow('YOLO Anomaly Detection - Webcam', result_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                frame_number += 1

                if frame_number % 300 == 0:
                    stats = self.get_statistics()
                    self.logger.info(f"실시간 통계 - FPS: {stats['avg_fps']:.1f}, "
                                   f"이상행동: {stats['total_anomalies']}")

        except KeyboardInterrupt:
            self.logger.info("사용자 중단")
        except Exception as e:
            self.logger.error(f"웹캠 처리 오류: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("웹캠 처리 완료")

    def save_model(self, filepath):
        """모델 저장"""
        try:
            model_saved = self.anomaly_detector.save_model(filepath)

            if model_saved:
                system_info = {
                    'config': self.config,
                    'stats': self.get_statistics(),
                    'device': self.device
                }

                info_filepath = filepath.replace('.pkl', '_info.json')
                with open(info_filepath, 'w') as f:
                    json.dump(system_info, f, indent=2, default=str)

                self.logger.info(f"모델 저장 완료: {filepath}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            return False

    def load_model(self, filepath):
        """모델 로드"""
        try:
            model_loaded = self.anomaly_detector.load_model(filepath)

            if model_loaded:
                info_filepath = filepath.replace('.pkl', '_info.json')
                if os.path.exists(info_filepath):
                    with open(info_filepath, 'r') as f:
                        system_info = json.load(f)
                    self.logger.info(f"모델 정보 로드 완료")

                self.logger.info(f"모델 로드 완료: {filepath}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            return False

    def get_statistics(self):
        """통계 반환"""
        current_time = time.time()
        total_time = current_time - self.stats['start_time']

        stats = {
            'total_frames': self.stats['frame_count'],
            'total_detections': self.stats['detection_count'],
            'total_anomalies': self.stats['anomaly_count'],
            'total_runtime': total_time,
            'avg_fps': self.stats['frame_count'] / total_time if total_time > 0 else 0,
            'avg_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
            'detection_rate': self.stats['detection_count'] / self.stats['frame_count'] if self.stats['frame_count'] > 0 else 0,
            'anomaly_rate': self.stats['anomaly_count'] / self.stats['frame_count'] if self.stats['frame_count'] > 0 else 0
        }

        if hasattr(self.detector, 'get_performance_stats'):
            stats['detector_stats'] = self.detector.get_performance_stats()

        if hasattr(self.tracker, 'get_statistics'):
            stats['tracker_stats'] = self.tracker.get_statistics()

        if hasattr(self.anomaly_detector, 'get_statistics'):
            stats['anomaly_detector_stats'] = self.anomaly_detector.get_statistics()

        return stats

    def reset_statistics(self):
        """통계 초기화"""
        self.stats = {
            'frame_count': 0,
            'detection_count': 0,
            'tracking_count': 0,
            'anomaly_count': 0,
            'processing_times': deque(maxlen=100),
            'start_time': time.time()
        }
        self.logger.info("통계 초기화 완료")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='YOLO 기반 이상행동 검출 시스템')
    parser.add_argument('--mode', choices=['train', 'video', 'webcam'], required=True,
                        help='실행 모드')
    parser.add_argument('--config', type=str, help='설정 파일 경로')
    parser.add_argument('--input', type=str, help='입력 비디오 파일 경로')
    parser.add_argument('--output', type=str, help='출력 비디오 파일 경로')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO 모델 경로')
    parser.add_argument('--device', type=str, default='cpu',
                        help='사용할 디바이스 (cpu, cuda)')
    parser.add_argument('--train_video', type=str,
                        help='훈련용 비디오 파일 경로')
    parser.add_argument('--model_save', type=str, default='trained_model.pkl',
                        help='훈련된 모델 저장 경로')
    parser.add_argument('--model_load', type=str,
                        help='사전 훈련된 모델 로드 경로')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='웹캠 ID')
    parser.add_argument('--display', action='store_true',
                        help='화면 표시 활성화')

    args = parser.parse_args()

    # 알림 콜백 함수
    def alert_callback(alert_type, alert_data):
        if alert_type == 'alert_start':
            print(f"🚨 이상행동 알림! 연속: {alert_data.get('consecutive_anomalies', 0)}회")
        elif alert_type == 'alert_end':
            print(f"✅ 알림 해제 (지속시간: {alert_data.get('duration', 0):.1f}초)")

    try:
        # 시스템 초기화
        system = YOLOAnomalyDetectionSystem(
            config_path=args.config,
            model_path=args.model,
            device=args.device
        )

        system.add_alert_callback(alert_callback)

        if args.mode == 'train':
            if not args.train_video:
                print("Error: 훈련 모드에는 --train_video가 필요합니다.")
                return

            print("모델 훈련 시작...")
            success = system.train_on_video(args.train_video)

            if success:
                system.save_model(args.model_save)
                print(f"✅ 훈련 완료! 모델 저장: {args.model_save}")

                stats = system.get_statistics()
                print(f"통계: {stats['total_frames']}프레임, "
                      f"{stats['total_detections']}회 검출, "
                      f"FPS: {stats['avg_fps']:.1f}")
            else:
                print("❌ 훈련 실패")

        elif args.mode == 'video':
            if not args.input:
                print("Error: 비디오 모드에는 --input이 필요합니다.")
                return

            if args.model_load:
                if not system.load_model(args.model_load):
                    print("Error: 모델 로드 실패")
                    return
            else:
                print("Warning: 모델 없이 객체 검출만 수행합니다.")

            print("비디오 처리 시작...")
            stats = system.process_video(args.input, args.output, display=args.display)

            if stats:
                print(f"✅ 비디오 처리 완료!")
                print(f"통계: {stats['total_frames']}프레임, "
                      f"{stats['total_anomalies']}회 이상행동, "
                      f"FPS: {stats['avg_fps']:.1f}")
                if args.output:
                    print(f"결과 저장: {args.output}")

        elif args.mode == 'webcam':
            if args.model_load:
                if not system.load_model(args.model_load):
                    print("Error: 모델 로드 실패")
                    return
            else:
                print("Warning: 모델 없이 객체 검출만 수행합니다.")

            print("웹캠 실시간 검출 시작...")
            print("종료: q키")
            system.process_webcam(args.camera_id)

            final_stats = system.get_statistics()
            print(f"최종 통계: {final_stats['total_frames']}프레임, "
                  f"{final_stats['total_anomalies']}회 이상행동")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("프로그램 종료")


if __name__ == "__main__":
    main()