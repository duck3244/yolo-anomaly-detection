"""
feature_extractor.py
행동 특징 추출을 담당하는 모듈
"""

import numpy as np
import cv2
from collections import deque
import math

class FeatureExtractor:
    """행동 특징 추출기"""
    
    def __init__(self):
        self.prev_positions = {}
        self.feature_history = {}
        
    def extract_motion_features(self, person_id, bbox, prev_bbox=None):
        """
        모션 기반 특징 추출
        
        Args:
            person_id (int): 사람 ID
            bbox (list): 현재 바운딩 박스 [x1, y1, x2, y2]
            prev_bbox (list): 이전 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            np.array: 모션 특징 벡터
        """
        features = []
        
        # 바운딩 박스 기본 특징
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # 정규화된 크기 특징 (0-1 범위로 가정)
        normalized_width = min(width / 640.0, 1.0)  # 640은 일반적인 프레임 너비
        normalized_height = min(height / 480.0, 1.0)  # 480은 일반적인 프레임 높이
        normalized_area = min(area / (640.0 * 480.0), 1.0)
        
        features.extend([normalized_width, normalized_height, normalized_area, aspect_ratio])
        
        # 모션 특징
        if prev_bbox is not None:
            # 중심점 이동
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            prev_center_x = (prev_bbox[0] + prev_bbox[2]) / 2
            prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
            
            dx = center_x - prev_center_x
            dy = center_y - prev_center_y
            velocity = math.sqrt(dx**2 + dy**2)
            
            # 정규화된 이동량
            normalized_dx = dx / 640.0
            normalized_dy = dy / 480.0
            normalized_velocity = velocity / 100.0  # 100픽셀을 기준으로 정규화
            
            # 크기 변화
            prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
            area_change_ratio = (area - prev_area) / (prev_area + 1e-6)
            
            # 방향 (각도)
            direction = math.atan2(dy, dx) if dx != 0 or dy != 0 else 0
            direction_normalized = (direction + math.pi) / (2 * math.pi)  # 0-1 범위로 정규화
            
            features.extend([normalized_dx, normalized_dy, normalized_velocity, 
                           area_change_ratio, direction_normalized])
        else:
            # 첫 프레임인 경우 기본값
            features.extend([0, 0, 0, 0, 0])
            
        return np.array(features, dtype=np.float32)
    
    def extract_spatial_features(self, bbox, frame_shape):
        """
        공간적 특징 추출
        
        Args:
            bbox (list): 바운딩 박스 [x1, y1, x2, y2]
            frame_shape (tuple): 프레임 크기 (height, width)
            
        Returns:
            np.array: 공간 특징 벡터
        """
        features = []
        frame_height, frame_width = frame_shape
        
        # 정규화된 중심 위치
        center_x = (bbox[0] + bbox[2]) / (2 * frame_width)
        center_y = (bbox[1] + bbox[3]) / (2 * frame_height)
        
        # 정규화된 바운딩 박스 위치
        left = bbox[0] / frame_width
        top = bbox[1] / frame_height
        right = bbox[2] / frame_width
        bottom = bbox[3] / frame_height
        
        # 화면 경계와의 거리
        left_distance = left
        right_distance = 1.0 - right
        top_distance = top
        bottom_distance = 1.0 - bottom
        
        # 화면 중앙으로부터의 거리
        center_distance = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
        
        # 화면 영역 구분 (9개 영역)
        zone_x = int(center_x * 3)  # 0, 1, 2
        zone_y = int(center_y * 3)  # 0, 1, 2
        zone = zone_y * 3 + zone_x  # 0~8
        zone_one_hot = [0] * 9
        if 0 <= zone < 9:
            zone_one_hot[zone] = 1
        
        features.extend([center_x, center_y, left, top, right, bottom,
                        left_distance, right_distance, top_distance, bottom_distance,
                        center_distance])
        features.extend(zone_one_hot)
        
        return np.array(features, dtype=np.float32)
    
    def extract_shape_features(self, bbox):
        """
        형태 기반 특징 추출
        
        Args:
            bbox (list): 바운딩 박스 [x1, y1, x2, y2]
            
        Returns:
            np.array: 형태 특징 벡터
        """
        features = []
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area > 0:
            # 종횡비
            aspect_ratio = width / height
            
            # 컴팩트함 (완전한 사각형에 가까운 정도)
            perimeter = 2 * (width + height)
            compactness = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # 사각형 충실도 (실제 면적 / 바운딩 박스 면적의 비율로 근사)
            # 실제로는 실루엣 정보가 필요하지만, 여기서는 바운딩 박스만 사용
            rectangularity = 1.0  # 바운딩 박스 자체이므로 1.0
            
            # 연장성 (길쭉한 정도)
            elongation = max(width, height) / (min(width, height) + 1e-6)
            
            features.extend([aspect_ratio, compactness, rectangularity, elongation])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_temporal_features(self, person_id, current_features, window_size=10):
        """
        시간적 특징 추출
        
        Args:
            person_id (int): 사람 ID
            current_features (np.array): 현재 프레임의 특징
            window_size (int): 시간 윈도우 크기
            
        Returns:
            np.array: 시간적 특징 벡터
        """
        # 특징 히스토리 초기화
        if person_id not in self.feature_history:
            self.feature_history[person_id] = deque(maxlen=window_size)
        
        # 현재 특징 추가
        self.feature_history[person_id].append(current_features)
        
        history = list(self.feature_history[person_id])
        
        if len(history) < 2:
            # 충분한 히스토리가 없으면 기본값 반환
            return np.zeros(current_features.shape[0] * 3, dtype=np.float32)
        
        # 히스토리를 배열로 변환
        history_array = np.array(history)
        
        # 통계적 특징 추출
        mean_features = np.mean(history_array, axis=0)
        std_features = np.std(history_array, axis=0)
        
        # 변화량 특징
        if len(history) >= 2:
            diff_features = history_array[-1] - history_array[-2]
        else:
            diff_features = np.zeros_like(current_features)
        
        # 결합
        temporal_features = np.concatenate([mean_features, std_features, diff_features])
        
        return temporal_features.astype(np.float32)
    
    def extract_velocity_features(self, person_id, bbox, frame_number, fps=30):
        """
        속도 관련 특징 추출
        
        Args:
            person_id (int): 사람 ID
            bbox (list): 바운딩 박스 [x1, y1, x2, y2]
            frame_number (int): 현재 프레임 번호
            fps (float): 초당 프레임 수
            
        Returns:
            np.array: 속도 특징 벡터
        """
        current_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        current_time = frame_number / fps
        
        # 이전 위치 정보 가져오기
        if person_id not in self.prev_positions:
            self.prev_positions[person_id] = []
        
        self.prev_positions[person_id].append((current_center, current_time))
        
        # 최근 N개 위치만 유지
        if len(self.prev_positions[person_id]) > 10:
            self.prev_positions[person_id].pop(0)
        
        positions = self.prev_positions[person_id]
        
        if len(positions) < 2:
            return np.zeros(6, dtype=np.float32)
        
        # 순간 속도 계산
        last_pos, last_time = positions[-2]
        current_pos, current_time_val = positions[-1]
        
        dt = current_time_val - last_time
        if dt > 0:
            velocity = np.linalg.norm(current_pos - last_pos) / dt
            velocity_x = (current_pos[0] - last_pos[0]) / dt
            velocity_y = (current_pos[1] - last_pos[1]) / dt
        else:
            velocity = velocity_x = velocity_y = 0
        
        # 평균 속도 계산 (최근 5개 위치)
        if len(positions) >= 5:
            recent_positions = positions[-5:]
            total_distance = 0
            total_time = recent_positions[-1][1] - recent_positions[0][1]
            
            for i in range(1, len(recent_positions)):
                pos_diff = recent_positions[i][0] - recent_positions[i-1][0]
                total_distance += np.linalg.norm(pos_diff)
            
            avg_velocity = total_distance / total_time if total_time > 0 else 0
        else:
            avg_velocity = velocity
        
        # 가속도 계산
        if len(positions) >= 3:
            pos1, time1 = positions[-3]
            pos2, time2 = positions[-2]
            pos3, time3 = positions[-1]
            
            vel1 = np.linalg.norm(pos2 - pos1) / (time2 - time1) if time2 > time1 else 0
            vel2 = np.linalg.norm(pos3 - pos2) / (time3 - time2) if time3 > time2 else 0
            
            acceleration = abs(vel2 - vel1) / (time3 - time1) if time3 > time1 else 0
        else:
            acceleration = 0
        
        # 방향 변화율 계산
        direction_change = 0
        if len(positions) >= 3:
            vec1 = positions[-2][0] - positions[-3][0]
            vec2 = positions[-1][0] - positions[-2][0]
            
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                direction_change = angle / math.pi  # 0-1로 정규화
        
        return np.array([velocity, velocity_x, velocity_y, avg_velocity, 
                        acceleration, direction_change], dtype=np.float32)
    
    def extract_comprehensive_features(self, person_id, bbox, prev_bbox, 
                                     frame_shape, frame_number=0, fps=30):
        """
        종합적인 특징 추출 (모든 특징을 결합)
        
        Args:
            person_id (int): 사람 ID
            bbox (list): 현재 바운딩 박스
            prev_bbox (list): 이전 바운딩 박스
            frame_shape (tuple): 프레임 크기
            frame_number (int): 프레임 번호
            fps (float): 초당 프레임 수
            
        Returns:
            np.array: 종합 특징 벡터
        """
        # 각 특징 추출
        motion_features = self.extract_motion_features(person_id, bbox, prev_bbox)
        spatial_features = self.extract_spatial_features(bbox, frame_shape)
        shape_features = self.extract_shape_features(bbox)
        velocity_features = self.extract_velocity_features(person_id, bbox, frame_number, fps)
        
        # 기본 특징들 결합
        basic_features = np.concatenate([motion_features, spatial_features, 
                                       shape_features, velocity_features])
        
        # 시간적 특징 추가
        temporal_features = self.extract_temporal_features(person_id, basic_features)
        
        # 최종 특징 벡터
        comprehensive_features = np.concatenate([basic_features, temporal_features])
        
        return comprehensive_features
    
    def reset_person_history(self, person_id):
        """특정 사람의 히스토리 초기화"""
        if person_id in self.feature_history:
            del self.feature_history[person_id]
        if person_id in self.prev_positions:
            del self.prev_positions[person_id]
    
    def cleanup_old_histories(self, active_person_ids):
        """활성화되지 않은 사람들의 히스토리 정리"""
        # 특징 히스토리 정리
        old_ids = set(self.feature_history.keys()) - set(active_person_ids)
        for person_id in old_ids:
            del self.feature_history[person_id]
        
        # 위치 히스토리 정리
        old_ids = set(self.prev_positions.keys()) - set(active_person_ids)
        for person_id in old_ids:
            del self.prev_positions[person_id]
    
    def get_feature_statistics(self):
        """특징 추출 통계 정보 반환"""
        return {
            'tracked_persons': len(self.feature_history),
            'persons_with_positions': len(self.prev_positions),
            'total_feature_histories': sum(len(hist) for hist in self.feature_history.values()),
            'total_position_histories': sum(len(hist) for hist in self.prev_positions.values())
        }


class AdvancedFeatureExtractor(FeatureExtractor):
    """고급 특징 추출기 (추가 특징들 포함)"""
    
    def __init__(self):
        super().__init__()
        self.interaction_history = {}
        self.crowd_density_cache = {}
    
    def extract_interaction_features(self, person_id, bbox, all_bboxes, person_ids):
        """
        다른 사람들과의 상호작용 특징 추출
        
        Args:
            person_id (int): 현재 사람 ID
            bbox (list): 현재 사람의 바운딩 박스
            all_bboxes (list): 모든 사람들의 바운딩 박스
            person_ids (list): 모든 사람들의 ID
            
        Returns:
            np.array: 상호작용 특징 벡터
        """
        features = []
        
        current_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        
        # 다른 사람들과의 거리 계산
        distances = []
        relative_positions = []
        size_ratios = []
        
        for i, other_bbox in enumerate(all_bboxes):
            if person_ids[i] == person_id:
                continue
                
            other_center = np.array([(other_bbox[0] + other_bbox[2]) / 2, 
                                   (other_bbox[1] + other_bbox[3]) / 2])
            
            # 거리
            distance = np.linalg.norm(current_center - other_center)
            distances.append(distance)
            
            # 상대 위치 (방향)
            relative_pos = other_center - current_center
            if np.linalg.norm(relative_pos) > 0:
                relative_pos = relative_pos / np.linalg.norm(relative_pos)
            relative_positions.append(relative_pos)
            
            # 크기 비율
            current_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            other_area = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
            size_ratio = other_area / (current_area + 1e-6)
            size_ratios.append(size_ratio)
        
        if distances:
            # 최근접 거리
            min_distance = min(distances)
            # 평균 거리
            avg_distance = np.mean(distances)
            # 근처 사람 수 (일정 거리 내)
            nearby_count = sum(1 for d in distances if d < 100)  # 100픽셀 내
            # 군집 밀도
            crowd_density = len(distances) / (avg_distance + 1e-6)
        else:
            min_distance = avg_distance = nearby_count = crowd_density = 0
        
        features.extend([min_distance / 100.0,  # 정규화
                        avg_distance / 100.0,
                        nearby_count,
                        crowd_density / 10.0])  # 정규화
        
        # 상대 위치의 분산 (사람들이 얼마나 분산되어 있는지)
        if relative_positions:
            rel_pos_array = np.array(relative_positions)
            position_variance = np.var(rel_pos_array, axis=0).mean()
        else:
            position_variance = 0
        
        features.append(position_variance)
        
        return np.array(features, dtype=np.float32)
    
    def extract_context_features(self, bbox, frame, background_model=None):
        """
        환경 컨텍스트 특징 추출
        
        Args:
            bbox (list): 바운딩 박스
            frame (np.array): 현재 프레임
            background_model: 배경 모델 (선택사항)
            
        Returns:
            np.array: 컨텍스트 특징 벡터
        """
        features = []
        
        # ROI 추출
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(10, dtype=np.float32)
        
        # 색상 특징
        if len(roi.shape) == 3:
            # 평균 색상
            mean_color = np.mean(roi, axis=(0, 1))
            # 색상 분산
            color_variance = np.var(roi, axis=(0, 1))
            features.extend(mean_color / 255.0)  # 정규화
            features.extend(color_variance / (255.0 ** 2))  # 정규화
        else:
            # 그레이스케일
            mean_intensity = np.mean(roi)
            intensity_variance = np.var(roi)
            features.extend([mean_intensity / 255.0, intensity_variance / (255.0 ** 2)])
            features.extend([0, 0, 0, 0])  # RGB 채널 패딩
        
        # 텍스처 특징 (간단한 그래디언트 기반)
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
        
        # 소벨 필터로 에지 계산
        if gray_roi.shape[0] > 1 and gray_roi.shape[1] > 1:
            sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            edge_mean = np.mean(edge_magnitude)
            edge_variance = np.var(edge_magnitude)
        else:
            edge_mean = edge_variance = 0
        
        features.extend([edge_mean / 255.0, edge_variance / (255.0 ** 2)])
        
        return np.array(features, dtype=np.float32)
    
    def extract_anomaly_indicators(self, person_id, comprehensive_features):
        """
        이상행동 지표 특징 추출
        
        Args:
            person_id (int): 사람 ID
            comprehensive_features (np.array): 종합 특징
            
        Returns:
            np.array: 이상행동 지표 특징
        """
        features = []
        
        # 특징의 급격한 변화 감지
        if person_id in self.feature_history and len(self.feature_history[person_id]) >= 2:
            recent_features = list(self.feature_history[person_id])[-2:]
            feature_change = np.linalg.norm(recent_features[-1] - recent_features[-2])
            feature_change_normalized = min(feature_change, 10.0) / 10.0
        else:
            feature_change_normalized = 0
        
        features.append(feature_change_normalized)
        
        # 특징의 일관성 (분산)
        if person_id in self.feature_history and len(self.feature_history[person_id]) >= 5:
            recent_features = np.array(list(self.feature_history[person_id])[-5:])
            feature_consistency = np.mean(np.std(recent_features, axis=0))
            feature_consistency_normalized = min(feature_consistency, 5.0) / 5.0
        else:
            feature_consistency_normalized = 0
        
        features.append(feature_consistency_normalized)
        
        # 주기적 패턴 감지 (간단한 자기상관)
        periodicity = 0
        if person_id in self.feature_history and len(self.feature_history[person_id]) >= 10:
            recent_features = np.array(list(self.feature_history[person_id])[-10:])
            # 첫 번째 특징 차원만 사용하여 주기성 검사
            first_dim = recent_features[:, 0]
            
            # 단순한 자기상관 계산
            autocorr = np.correlate(first_dim, first_dim, mode='full')
            middle = len(autocorr) // 2
            # 지연 1-3에서의 상관관계 확인
            if middle + 3 < len(autocorr):
                max_delayed_corr = np.max(autocorr[middle+1:middle+4])
                periodicity = max_delayed_corr / (autocorr[middle] + 1e-6)
                periodicity = max(0, min(1, periodicity))
        
        features.append(periodicity)
        
        return np.array(features, dtype=np.float32)
    
    def extract_all_features(self, person_id, bbox, prev_bbox, frame_shape, 
                           frame, all_bboxes, person_ids, frame_number=0, fps=30):
        """
        모든 고급 특징 추출
        
        Returns:
            dict: 각 특징 타입별로 구분된 특징 딕셔너리
        """
        # 기본 특징
        comprehensive_features = self.extract_comprehensive_features(
            person_id, bbox, prev_bbox, frame_shape, frame_number, fps
        )
        
        # 상호작용 특징
        interaction_features = self.extract_interaction_features(
            person_id, bbox, all_bboxes, person_ids
        )
        
        # 컨텍스트 특징
        context_features = self.extract_context_features(bbox, frame)
        
        # 이상행동 지표
        anomaly_indicators = self.extract_anomaly_indicators(
            person_id, comprehensive_features
        )
        
        return {
            'comprehensive': comprehensive_features,
            'interaction': interaction_features,
            'context': context_features,
            'anomaly_indicators': anomaly_indicators,
            'combined': np.concatenate([
                comprehensive_features,
                interaction_features,
                context_features,
                anomaly_indicators
            ])
        }