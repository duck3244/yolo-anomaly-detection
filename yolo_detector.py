"""
yolo_detector.py
YOLO 기반 객체 검출을 담당하는 모듈
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from pathlib import Path
import time

class YOLODetector:
    """YOLO 기반 객체 검출기"""
    
    def __init__(self, model_path='yolov8n.pt', device='cpu', confidence_threshold=0.5):
        """
        YOLO 검출기 초기화
        
        Args:
            model_path (str): YOLO 모델 경로
            device (str): 사용할 디바이스 ('cpu', 'cuda', 'mps')
            confidence_threshold (float): 검출 신뢰도 임계값
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # YOLO 모델 로드
        self.model = None
        self._load_model()
        
        # COCO 클래스 정보
        self.person_class_id = 0  # COCO dataset에서 사람 클래스 ID
        self.class_names = self._get_class_names()
        
        # 성능 통계
        self.stats = {
            'total_inferences': 0,
            'total_detections': 0,
            'avg_inference_time': 0,
            'inference_times': []
        }
        
        # 로깅
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """YOLO 모델 로드"""
        try:
            # 모델 파일 존재 확인
            if not Path(self.model_path).exists():
                self.logger.info(f"모델 파일이 없습니다. 다운로드 시작: {self.model_path}")
            
            # YOLO 모델 로드
            self.model = YOLO(self.model_path)
            
            # 디바이스 설정
            self.model.to(self.device)
            
            self.logger.info(f"YOLO 모델 로드 완료: {self.model_path}, 디바이스: {self.device}")
            
        except Exception as e:
            self.logger.error(f"YOLO 모델 로드 실패: {e}")
            raise
    
    def _get_class_names(self):
        """클래스 이름 목록 반환"""
        if self.model is not None:
            return self.model.names
        else:
            # COCO 클래스 이름 (기본값)
            return {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                # ... (필요에 따라 추가)
            }
    
    def detect_persons(self, frame, return_all_info=False):
        """
        프레임에서 사람 검출
        
        Args:
            frame (np.array): 입력 프레임
            return_all_info (bool): 상세 정보 반환 여부
            
        Returns:
            list: 검출된 사람들의 정보
        """
        if self.model is None:
            self.logger.error("YOLO 모델이 로드되지 않았습니다.")
            return []
        
        start_time = time.time()
        
        try:
            # YOLO 추론 수행
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            inference_time = time.time() - start_time
            self._update_stats(inference_time)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 사람 클래스만 필터링
                        if class_id == self.person_class_id:
                            # 바운딩 박스 좌표 추출
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            
                            detection_info = {
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.class_names.get(class_id, 'unknown')
                            }
                            
                            # 추가 정보 포함
                            if return_all_info:
                                detection_info.update({
                                    'area': float((x2 - x1) * (y2 - y1)),
                                    'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                    'width': float(x2 - x1),
                                    'height': float(y2 - y1),
                                    'aspect_ratio': float((x2 - x1) / (y2 - y1)) if y2 > y1 else 0
                                })
                            
                            detections.append(detection_info)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"객체 검출 중 오류: {e}")
            return []
    
    def detect_all_objects(self, frame, filter_classes=None):
        """
        모든 객체 검출 (사람 외 다른 객체도 포함)
        
        Args:
            frame (np.array): 입력 프레임
            filter_classes (list): 필터링할 클래스 ID 리스트
            
        Returns:
            list: 검출된 모든 객체 정보
        """
        if self.model is None:
            return []
        
        start_time = time.time()
        
        try:
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            inference_time = time.time() - start_time
            self._update_stats(inference_time)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 클래스 필터링
                        if filter_classes is None or class_id in filter_classes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self.class_names.get(class_id, 'unknown'),
                                'area': float((x2 - x1) * (y2 - y1)),
                                'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                            })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"객체 검출 중 오류: {e}")
            return []
    
    def batch_detect(self, frames, batch_size=4):
        """
        배치 단위로 여러 프레임 처리
        
        Args:
            frames (list): 프레임 리스트
            batch_size (int): 배치 크기
            
        Returns:
            list: 각 프레임별 검출 결과
        """
        if self.model is None:
            return [[] for _ in frames]
        
        all_results = []
        
        # 배치 단위로 처리
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            try:
                start_time = time.time()
                
                # 배치 추론
                results = self.model(batch_frames, verbose=False, conf=self.confidence_threshold)
                
                inference_time = time.time() - start_time
                self._update_stats(inference_time / len(batch_frames))  # 프레임당 시간
                
                # 결과 파싱
                for result in results:
                    detections = []
                    boxes = result.boxes
                    
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            
                            if class_id == self.person_class_id:
                                confidence = float(box.conf[0])
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = xyxy
                                
                                detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'class_name': 'person'
                                })
                    
                    all_results.append(detections)
                
            except Exception as e:
                self.logger.error(f"배치 검출 중 오류: {e}")
                # 실패한 배치에 대해 빈 결과 추가
                all_results.extend([[] for _ in batch_frames])
        
        return all_results
    
    def _update_stats(self, inference_time):
        """성능 통계 업데이트"""
        self.stats['total_inferences'] += 1
        self.stats['inference_times'].append(inference_time)
        
        # 최근 100개 추론 시간만 유지
        if len(self.stats['inference_times']) > 100:
            self.stats['inference_times'].pop(0)
        
        # 평균 추론 시간 계산
        self.stats['avg_inference_time'] = np.mean(self.stats['inference_times'])
    
    def get_performance_stats(self):
        """성능 통계 반환"""
        stats = self.stats.copy()
        
        if self.stats['inference_times']:
            stats['min_inference_time'] = min(self.stats['inference_times'])
            stats['max_inference_time'] = max(self.stats['inference_times'])
            stats['fps'] = 1.0 / self.stats['avg_inference_time'] if self.stats['avg_inference_time'] > 0 else 0
        else:
            stats['min_inference_time'] = 0
            stats['max_inference_time'] = 0
            stats['fps'] = 0
        
        return stats
    
    def set_confidence_threshold(self, threshold):
        """신뢰도 임계값 설정"""
        if 0.1 <= threshold <= 0.9:
            self.confidence_threshold = threshold
            self.logger.info(f"신뢰도 임계값 변경: {threshold}")
        else:
            self.logger.warning(f"유효하지 않은 임계값: {threshold}")
    
    def warm_up(self, num_iterations=5):
        """
        모델 워밍업 (초기 추론 속도 최적화)
        
        Args:
            num_iterations (int): 워밍업 반복 횟수
        """
        if self.model is None:
            return
        
        self.logger.info("YOLO 모델 워밍업 시작...")
        
        # 더미 이미지 생성
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        warm_up_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            try:
                _ = self.model(dummy_frame, verbose=False)
                warm_up_time = time.time() - start_time
                warm_up_times.append(warm_up_time)
            except Exception as e:
                self.logger.warning(f"워밍업 중 오류: {e}")
        
        if warm_up_times:
            avg_warm_up_time = np.mean(warm_up_times)
            self.logger.info(f"워밍업 완료: 평균 {avg_warm_up_time:.3f}초")
        else:
            self.logger.warning("워밍업 실패")
    
    def optimize_for_inference(self):
        """추론 최적화 설정"""
        if self.model is None:
            return
        
        try:
            # 반정밀도 사용 (GPU에서)
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.half()
                self.logger.info("GPU 반정밀도 최적화 적용")
            
            # 추론 모드 설정
            self.model.eval()
            
            # 그래디언트 비활성화
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.logger.info("추론 최적화 완료")
            
        except Exception as e:
            self.logger.error(f"추론 최적화 실패: {e}")


class MultiScaleYOLODetector(YOLODetector):
    """다중 스케일 YOLO 검출기"""
    
    def __init__(self, model_path='yolov8n.pt', device='cpu', confidence_threshold=0.5):
        super().__init__(model_path, device, confidence_threshold)
        self.scales = [0.5, 0.75, 1.0, 1.25]  # 다양한 스케일
        
    def detect_multiscale(self, frame, use_nms=True):
        """
        다중 스케일 검출
        
        Args:
            frame (np.array): 입력 프레임
            use_nms (bool): Non-Maximum Suppression 사용 여부
            
        Returns:
            list: 다중 스케일 검출 결과
        """
        all_detections = []
        original_h, original_w = frame.shape[:2]
        
        for scale in self.scales:
            # 스케일에 따른 프레임 크기 조정
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            scaled_frame = cv2.resize(frame, (new_w, new_h))
            
            # 스케일된 프레임에서 검출
            detections = self.detect_persons(scaled_frame)
            
            # 바운딩 박스를 원본 크기로 변환
            for detection in detections:
                bbox = detection['bbox']
                detection['bbox'] = [
                    bbox[0] / scale,  # x1
                    bbox[1] / scale,  # y1
                    bbox[2] / scale,  # x2
                    bbox[3] / scale   # y2
                ]
                detection['scale'] = scale
                
            all_detections.extend(detections)
        
        # NMS 적용
        if use_nms and all_detections:
            all_detections = self._apply_nms(all_detections)
        
        return all_detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression 적용"""
        if not detections:
            return detections
        
        # 바운딩 박스와 점수 추출
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # NMS 적용
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []


class OptimizedYOLODetector(YOLODetector):
    """최적화된 YOLO 검출기"""
    
    def __init__(self, model_path='yolov8n.pt', device='cpu', 
                 confidence_threshold=0.5, input_size=(640, 640)):
        super().__init__(model_path, device, confidence_threshold)
        self.input_size = input_size
        self.frame_skip = 1  # 프레임 스키핑
        self.roi_enabled = False
        self.roi_boxes = []  # 관심 영역들
        
    def set_frame_skip(self, skip_frames):
        """프레임 스키핑 설정"""
        self.frame_skip = max(1, skip_frames)
        self.logger.info(f"프레임 스키핑 설정: {self.frame_skip}")
    
    def set_roi(self, roi_boxes):
        """
        관심 영역(ROI) 설정
        
        Args:
            roi_boxes (list): ROI 박스 리스트 [[x1,y1,x2,y2], ...]
        """
        self.roi_boxes = roi_boxes
        self.roi_enabled = len(roi_boxes) > 0
        self.logger.info(f"ROI 설정: {len(roi_boxes)}개 영역")
    
    def detect_optimized(self, frame, frame_count=0):
        """
        최적화된 검출
        
        Args:
            frame (np.array): 입력 프레임
            frame_count (int): 프레임 카운트
            
        Returns:
            list: 검출 결과 (빈 리스트면 스키핑됨)
        """
        # 프레임 스키핑
        if frame_count % self.frame_skip != 0:
            return []
        
        # ROI 영역만 처리
        if self.roi_enabled:
            return self._detect_roi(frame)
        
        # 프레임 크기 최적화
        optimized_frame = self._optimize_frame_size(frame)
        
        # 기본 검출 수행
        return self.detect_persons(optimized_frame)
    
    def _detect_roi(self, frame):
        """ROI 영역에서만 검출"""
        all_detections = []
        
        for roi_box in self.roi_boxes:
            x1, y1, x2, y2 = map(int, roi_box)
            
            # ROI 영역 추출
            roi_frame = frame[y1:y2, x1:x2]
            
            if roi_frame.size == 0:
                continue
            
            # ROI에서 검출
            roi_detections = self.detect_persons(roi_frame)
            
            # 좌표를 전체 프레임 기준으로 변환
            for detection in roi_detections:
                bbox = detection['bbox']
                detection['bbox'] = [
                    bbox[0] + x1,  # x1
                    bbox[1] + y1,  # y1
                    bbox[2] + x1,  # x2
                    bbox[3] + y1   # y2
                ]
                detection['roi_id'] = len(all_detections)  # ROI 식별자
                
            all_detections.extend(roi_detections)
        
        return all_detections
    
    def _optimize_frame_size(self, frame):
        """프레임 크기 최적화"""
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        
        # 이미 최적 크기인 경우
        if w == target_w and h == target_h:
            return frame
        
        # 종횡비 유지하면서 리사이즈
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized_frame = cv2.resize(frame, (new_w, new_h))
        
        # 패딩 추가 (필요한 경우)
        if new_w != target_w or new_h != target_h:
            padded_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
            return padded_frame
        
        return resized_frame
    
    def benchmark_performance(self, test_frames, iterations=10):
        """성능 벤치마크"""
        self.logger.info("성능 벤치마크 시작...")
        
        # 워밍업
        self.warm_up()
        
        # 벤치마크 실행
        times = []
        detection_counts = []
        
        for iteration in range(iterations):
            start_time = time.time()
            
            total_detections = 0
            for frame in test_frames:
                detections = self.detect_persons(frame)
                total_detections += len(detections)
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            detection_counts.append(total_detections)
        
        # 결과 분석
        avg_time = np.mean(times)
        avg_detections = np.mean(detection_counts)
        fps = len(test_frames) / avg_time
        
        results = {
            'avg_processing_time': avg_time,
            'avg_fps': fps,
            'avg_detections_per_run': avg_detections,
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
        
        self.logger.info(f"벤치마크 결과: 평균 FPS {fps:.2f}, 평균 처리시간 {avg_time:.3f}초")
        
        return results


class YOLOEnsembleDetector:
    """YOLO 앙상블 검출기"""
    
    def __init__(self, model_paths, device='cpu', confidence_threshold=0.5):
        """
        앙상블 검출기 초기화
        
        Args:
            model_paths (list): 여러 YOLO 모델 경로들
            device (str): 사용할 디바이스
            confidence_threshold (float): 신뢰도 임계값
        """
        self.model_paths = model_paths
        self.detectors = []
        self.weights = []
        
        # 각 모델 로드
        for model_path in model_paths:
            try:
                detector = YOLODetector(model_path, device, confidence_threshold)
                self.detectors.append(detector)
                self.weights.append(1.0)  # 초기 가중치는 동일
            except Exception as e:
                logging.error(f"모델 로드 실패 {model_path}: {e}")
        
        self.logger = logging.getLogger(__name__)
        
        if not self.detectors:
            raise ValueError("사용 가능한 YOLO 모델이 없습니다.")
        
        # 가중치 정규화
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.logger.info(f"YOLO 앙상블 초기화: {len(self.detectors)}개 모델")
    
    def detect_ensemble(self, frame, fusion_method='weighted_nms'):
        """
        앙상블 검출
        
        Args:
            frame (np.array): 입력 프레임
            fusion_method (str): 융합 방법 ('weighted_nms', 'vote', 'average')
            
        Returns:
            list: 앙상블 검출 결과
        """
        all_detections = []
        
        # 각 검출기에서 결과 수집
        for i, detector in enumerate(self.detectors):
            detections = detector.detect_persons(frame, return_all_info=True)
            
            # 가중치 정보 추가
            for detection in detections:
                detection['detector_id'] = i
                detection['weight'] = self.weights[i]
                
            all_detections.extend(detections)
        
        if not all_detections:
            return []
        
        # 융합 방법에 따른 처리
        if fusion_method == 'weighted_nms':
            return self._weighted_nms(all_detections)
        elif fusion_method == 'vote':
            return self._vote_fusion(all_detections)
        elif fusion_method == 'average':
            return self._average_fusion(all_detections)
        else:
            raise ValueError(f"지원하지 않는 융합 방법: {fusion_method}")
    
    def _weighted_nms(self, detections, iou_threshold=0.5):
        """가중치 기반 NMS"""
        if not detections:
            return []
        
        # 가중치를 고려한 신뢰도 계산
        for detection in detections:
            detection['weighted_confidence'] = detection['confidence'] * detection['weight']
        
        # 가중 신뢰도 기준 정렬
        detections.sort(key=lambda x: x['weighted_confidence'], reverse=True)
        
        # NMS 수행
        final_detections = []
        
        while detections:
            # 가장 높은 점수의 검출 결과 선택
            best_detection = detections.pop(0)
            final_detections.append(best_detection)
            
            # 겹치는 박스들 제거
            remaining_detections = []
            for detection in detections:
                iou = self._calculate_iou(best_detection['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    remaining_detections.append(detection)
            
            detections = remaining_detections
        
        return final_detections
    
    def _vote_fusion(self, detections, iou_threshold=0.5, min_votes=2):
        """투표 기반 융합"""
        if len(self.detectors) < 2:
            return detections
        
        # 클러스터링을 통한 투표
        clusters = []
        
        for detection in detections:
            # 기존 클러스터와 겹치는지 확인
            assigned = False
            for cluster in clusters:
                representative = cluster[0]
                iou = self._calculate_iou(detection['bbox'], representative['bbox'])
                
                if iou > iou_threshold:
                    cluster.append(detection)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([detection])
        
        # 충분한 투표를 받은 클러스터만 유지
        final_detections = []
        
        for cluster in clusters:
            if len(cluster) >= min_votes:
                # 클러스터 내에서 가장 높은 신뢰도 선택
                best_detection = max(cluster, key=lambda x: x['confidence'])
                best_detection['votes'] = len(cluster)
                final_detections.append(best_detection)
        
        return final_detections
    
    def _average_fusion(self, detections, iou_threshold=0.5):
        """평균 기반 융합"""
        clusters = []
        
        for detection in detections:
            assigned = False
            for cluster in clusters:
                representative = cluster[0]
                iou = self._calculate_iou(detection['bbox'], representative['bbox'])
                
                if iou > iou_threshold:
                    cluster.append(detection)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([detection])
        
        final_detections = []
        
        for cluster in clusters:
            if len(cluster) > 1:
                # 바운딩 박스 평균 계산
                avg_bbox = np.mean([det['bbox'] for det in cluster], axis=0)
                avg_confidence = np.mean([det['confidence'] for det in cluster])
                
                final_detection = cluster[0].copy()
                final_detection['bbox'] = avg_bbox.tolist()
                final_detection['confidence'] = avg_confidence
                final_detection['ensemble_size'] = len(cluster)
                
                final_detections.append(final_detection)
            else:
                final_detections.append(cluster[0])
        
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """IoU (Intersection over Union) 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_weights(self, performance_feedback):
        """성능 피드백 기반 가중치 업데이트"""
        if len(performance_feedback) != len(self.detectors):
            self.logger.warning("피드백 길이가 검출기 수와 일치하지 않습니다.")
            return
        
        # 성능 점수를 가중치로 변환
        total_performance = sum(performance_feedback)
        if total_performance > 0:
            self.weights = [perf / total_performance for perf in performance_feedback]
            self.logger.info(f"가중치 업데이트: {self.weights}")
        else:
            self.logger.warning("유효한 성능 피드백이 없습니다.")