"""
person_tracker.py
사람 객체 추적을 담당하는 모듈
"""

import logging
import numpy as np
from collections import defaultdict, deque
import time

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # scipy 미설치 시 graceful fallback
    linear_sum_assignment = None
    _HAS_SCIPY = False

class PersonTracker:
    """간단한 사람 추적기 (DeepSORT 대신 centroid tracking 사용)"""
    
    def __init__(self, max_disappeared=10, max_distance=100, use_hungarian=True):
        """
        추적기 초기화

        Args:
            max_disappeared (int): 객체가 사라진 후 제거하기까지의 최대 프레임 수
            max_distance (int): 객체 매칭을 위한 최대 거리 (픽셀)
            use_hungarian (bool): 헝가리안 알고리즘 사용 여부 (scipy 필요)
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.use_hungarian = use_hungarian and _HAS_SCIPY
        self.logger = logging.getLogger(__name__)

        if use_hungarian and not _HAS_SCIPY:
            self.logger.warning(
                "scipy 미설치로 헝가리안 알고리즘 사용 불가, greedy fallback"
            )

    def register(self, centroid, bbox):
        """새로운 객체 등록"""
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'features': deque(maxlen=30),  # 최근 30프레임 특징 저장
            'created_at': time.time(),
            'last_seen': time.time()
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """객체 등록 해제"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        검출 결과로 추적 정보 업데이트
        
        Args:
            detections (list): 검출된 객체들의 정보 리스트
                [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
                
        Returns:
            dict: 추적 중인 객체들의 정보
        """
        current_time = time.time()
        
        # 검출된 객체가 없으면 모든 객체의 disappeared 카운트 증가
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        # 검출된 객체들의 중심점 계산
        input_centroids = []
        input_bboxes = []
        
        for detection in detections:
            bbox = detection['bbox']
            cx = int((bbox[0] + bbox[2]) / 2.0)
            cy = int((bbox[1] + bbox[3]) / 2.0)
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)

        # 추적 중인 객체가 없으면 모든 검출 결과를 새로 등록
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i])
        else:
            # 기존 객체와 새로운 검출 결과 매칭
            self._match_objects(input_centroids, input_bboxes, current_time)

        # 추적 중인 객체 정보 업데이트
        for object_id, obj_data in self.objects.items():
            obj_data['last_seen'] = current_time

        return self.objects

    def _match_objects(self, input_centroids, input_bboxes, current_time):
        """기존 객체와 새로운 검출 결과 매칭 (헝가리안 또는 greedy)"""
        object_centroids = [obj['centroid'] for obj in self.objects.values()]
        object_ids = list(self.objects.keys())

        # 거리 행렬 계산 (rows=기존, cols=검출)
        D = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] -
            np.array(input_centroids), axis=2
        )

        used_row_indices, used_col_indices = self._solve_assignment(D, object_ids,
                                                                   input_centroids,
                                                                   input_bboxes,
                                                                   current_time)

        # 매칭되지 않은 기존/검출 인덱스
        unused_row_indices = set(range(D.shape[0])) - used_row_indices
        unused_col_indices = set(range(D.shape[1])) - used_col_indices

        # 사라진 객체 처리
        for row in unused_row_indices:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # 새로 등장한 객체 등록
        for col in unused_col_indices:
            self.register(input_centroids[col], input_bboxes[col])

    def _solve_assignment(self, D, object_ids, input_centroids, input_bboxes,
                          current_time):
        """할당 문제 풀이: 헝가리안 또는 greedy"""
        used_row_indices = set()
        used_col_indices = set()

        if D.size == 0:
            return used_row_indices, used_col_indices

        if self.use_hungarian:
            # scipy 헝가리안 알고리즘 - 전역 최적 매칭
            row_ind, col_ind = linear_sum_assignment(D)
            for row, col in zip(row_ind, col_ind):
                if D[row, col] > self.max_distance:
                    continue
                self._apply_match(row, col, object_ids, input_centroids,
                                  input_bboxes, current_time)
                used_row_indices.add(row)
                used_col_indices.add(col)
        else:
            # 기존 greedy fallback
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            for row, col in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                self._apply_match(row, col, object_ids, input_centroids,
                                  input_bboxes, current_time)
                used_row_indices.add(row)
                used_col_indices.add(col)

        return used_row_indices, used_col_indices

    def _apply_match(self, row, col, object_ids, input_centroids, input_bboxes,
                     current_time):
        """단일 매칭 적용"""
        object_id = object_ids[row]
        self.objects[object_id]['centroid'] = input_centroids[col]
        self.objects[object_id]['bbox'] = input_bboxes[col]
        self.objects[object_id]['last_seen'] = current_time
        self.disappeared[object_id] = 0

    def get_object_info(self, object_id):
        """특정 객체의 상세 정보 반환"""
        if object_id in self.objects:
            obj = self.objects[object_id]
            return {
                'id': object_id,
                'centroid': obj['centroid'],
                'bbox': obj['bbox'],
                'age': time.time() - obj['created_at'],
                'last_seen': obj['last_seen'],
                'feature_history_length': len(obj['features'])
            }
        return None

    def get_all_active_objects(self):
        """모든 활성 객체 ID 반환"""
        return list(self.objects.keys())

    def cleanup_old_objects(self, max_age_seconds=300):
        """오래된 객체들 정리 (5분 이상)"""
        current_time = time.time()
        old_objects = []
        
        for object_id, obj_data in self.objects.items():
            if current_time - obj_data['last_seen'] > max_age_seconds:
                old_objects.append(object_id)
        
        for object_id in old_objects:
            self.deregister(object_id)
        
        return len(old_objects)

    def get_statistics(self):
        """추적 통계 정보 반환"""
        current_time = time.time()
        active_objects = len(self.objects)
        
        if active_objects == 0:
            return {
                'active_objects': 0,
                'total_created': self.next_object_id,
                'avg_age': 0,
                'oldest_object_age': 0
            }
        
        ages = [current_time - obj['created_at'] for obj in self.objects.values()]
        
        return {
            'active_objects': active_objects,
            'total_created': self.next_object_id,
            'avg_age': np.mean(ages),
            'oldest_object_age': max(ages),
            'avg_disappeared_count': np.mean(list(self.disappeared.values()))
        }

    def reset(self):
        """추적기 초기화"""
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}


class AdvancedPersonTracker(PersonTracker):
    """고급 기능을 가진 사람 추적기"""
    
    def __init__(self, max_disappeared=10, max_distance=100, 
                 track_features=True, smooth_trajectory=True):
        super().__init__(max_disappeared, max_distance)
        self.track_features = track_features
        self.smooth_trajectory = smooth_trajectory
        self.trajectory_history = defaultdict(list)
        
    def update(self, detections):
        """고급 추적 기능을 포함한 업데이트"""
        # 기본 추적 수행
        tracked_objects = super().update(detections)
        
        # 궤적 평활화
        if self.smooth_trajectory:
            self._smooth_trajectories()
        
        # 궤적 히스토리 업데이트
        self._update_trajectory_history()
        
        return tracked_objects
    
    def _smooth_trajectories(self):
        """칼만 필터 또는 간단한 이동평균을 사용한 궤적 평활화"""
        for object_id, obj_data in self.objects.items():
            if len(self.trajectory_history[object_id]) >= 3:
                # 최근 3개 위치의 이동평균
                recent_positions = self.trajectory_history[object_id][-3:]
                avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
                avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
                
                # 급격한 변화 억제
                current_x, current_y = obj_data['centroid']
                smoothed_x = 0.7 * current_x + 0.3 * avg_x
                smoothed_y = 0.7 * current_y + 0.3 * avg_y
                
                obj_data['centroid'] = (int(smoothed_x), int(smoothed_y))
    
    def _update_trajectory_history(self):
        """궤적 히스토리 업데이트"""
        for object_id, obj_data in self.objects.items():
            self.trajectory_history[object_id].append(obj_data['centroid'])
            
            # 히스토리 길이 제한 (최근 50개 위치만 저장)
            if len(self.trajectory_history[object_id]) > 50:
                self.trajectory_history[object_id].pop(0)
    
    def get_trajectory(self, object_id, num_points=10):
        """특정 객체의 궤적 반환"""
        if object_id in self.trajectory_history:
            return self.trajectory_history[object_id][-num_points:]
        return []
    
    def predict_next_position(self, object_id):
        """다음 위치 예측 (간단한 선형 예측)"""
        trajectory = self.get_trajectory(object_id, 5)
        if len(trajectory) < 2:
            return None
        
        # 최근 위치들의 평균 이동량 계산
        movements = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            movements.append((dx, dy))
        
        if movements:
            avg_dx = sum(mv[0] for mv in movements) / len(movements)
            avg_dy = sum(mv[1] for mv in movements) / len(movements)
            
            current_pos = trajectory[-1]
            predicted_pos = (
                int(current_pos[0] + avg_dx),
                int(current_pos[1] + avg_dy)
            )
            
            return predicted_pos
        
        return None
    
    def detect_sudden_movement(self, object_id, threshold=50):
        """급격한 움직임 감지"""
        trajectory = self.get_trajectory(object_id, 3)
        if len(trajectory) < 3:
            return False
        
        # 최근 두 이동의 거리 계산
        dist1 = np.linalg.norm(np.array(trajectory[-2]) - np.array(trajectory[-3]))
        dist2 = np.linalg.norm(np.array(trajectory[-1]) - np.array(trajectory[-2]))
        
        # 급격한 속도 변화 감지
        if abs(dist2 - dist1) > threshold:
            return True
        
        # 급격한 방향 변화 감지
        if len(trajectory) >= 3:
            vec1 = np.array(trajectory[-2]) - np.array(trajectory[-3])
            vec2 = np.array(trajectory[-1]) - np.array(trajectory[-2])
            
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                # 90도 이상 방향 변화
                if angle > np.pi / 2:
                    return True
        
        return False
