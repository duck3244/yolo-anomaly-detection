"""
anomaly_detector.py
이상행동 검출을 담당하는 모듈
"""

import numpy as np
import pickle
import json
import logging
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """기본 이상행동 검출기"""

    def __init__(self, window_size=30, contamination=0.1, algorithm='isolation_forest',
                 anomaly_ratio_threshold=0.3, validation_split=0.2,
                 min_training_samples=100, random_seed=42):
        """
        이상 검출기 초기화

        Args:
            window_size (int): 특징 윈도우 크기
            contamination (float): 이상치 비율 추정값 (0.05~0.2)
            algorithm (str): 사용할 알고리즘 ('isolation_forest', 'one_class_svm', 'elliptic_envelope', 'lof')
            anomaly_ratio_threshold (float): 윈도우 내 이상 판정 비율 임계값
            validation_split (float): 검증 데이터 비율 (0~0.5)
            min_training_samples (int): 최소 훈련 샘플 수
            random_seed (int): 난수 시드
        """
        self.window_size = window_size
        self.contamination = contamination
        self.algorithm = algorithm
        self.anomaly_ratio_threshold = anomaly_ratio_threshold
        self.validation_split = validation_split
        self.min_training_samples = min_training_samples
        self.random_seed = random_seed

        # 전처리 도구
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 분산의 95% 유지
        self.use_pca = False

        # 모델 초기화
        self.detector = self._create_detector()
        self.is_trained = False

        # 훈련 데이터 저장
        self.feature_history = deque(maxlen=10000)  # 최대 10000개 샘플 저장

        # 통계 정보
        self.stats = {
            'total_samples': 0,
            'anomaly_count': 0,
            'normal_count': 0,
            'false_positive_count': 0
        }

        # 훈련 성능 메트릭 (validation에서 계산)
        self.training_metrics = {}

        # 로깅
        self.logger = logging.getLogger(__name__)
    
    def _create_detector(self):
        """선택된 알고리즘에 따라 검출기 생성"""
        if self.algorithm == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=self.random_seed,
                n_estimators=100,
                n_jobs=-1
            )
        elif self.algorithm == 'one_class_svm':
            return OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=self.contamination
            )
        elif self.algorithm == 'elliptic_envelope':
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_seed
            )
        elif self.algorithm == 'lof':
            return LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20
            )
        else:
            raise ValueError(f"지원하지 않는 알고리즘: {self.algorithm}")
    
    def add_training_features(self, features):
        """
        훈련용 특징 추가
        
        Args:
            features (list or np.array): 특징 벡터들
        """
        if isinstance(features, (list, tuple)):
            for feature in features:
                if isinstance(feature, np.ndarray) and feature.ndim == 1:
                    self.feature_history.append(feature.copy())
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                self.feature_history.append(features.copy())
            elif features.ndim == 2:
                for feature in features:
                    self.feature_history.append(feature.copy())
    
    def train(self, use_pca=False):
        """
        정상 행동 패턴 학습 (train/validation split 포함)

        Args:
            use_pca (bool): PCA 차원 축소 사용 여부

        Returns:
            bool: 훈련 성공 여부
        """
        if len(self.feature_history) < self.min_training_samples:
            self.logger.warning(
                f"훈련 데이터가 부족합니다. 현재: {len(self.feature_history)}, "
                f"최소 필요: {self.min_training_samples}"
            )
            return False

        try:
            features_array = np.array(list(self.feature_history))

            # 유효하지 않은 값 제거 및 로깅
            valid_mask = ~(np.isnan(features_array).any(axis=1) |
                           np.isinf(features_array).any(axis=1))
            invalid_count = int((~valid_mask).sum())
            features_array = features_array[valid_mask]

            if invalid_count > 0:
                self.logger.warning(f"NaN/Inf 특징 {invalid_count}개 제거됨")

            if len(features_array) < self.min_training_samples:
                self.logger.warning(
                    f"유효한 특징 데이터가 부족합니다. "
                    f"현재: {len(features_array)}, 최소 필요: {self.min_training_samples}"
                )
                return False

            # train/validation 분할 (셔플 후 비율 분할, 재현성 보장)
            rng = np.random.default_rng(self.random_seed)
            indices = rng.permutation(len(features_array))
            features_array = features_array[indices]

            val_ratio = max(0.0, min(0.5, self.validation_split))
            val_size = int(len(features_array) * val_ratio)

            if val_size > 0:
                val_features = features_array[:val_size]
                train_features = features_array[val_size:]
            else:
                val_features = np.empty((0, features_array.shape[1]),
                                        dtype=features_array.dtype)
                train_features = features_array

            self.logger.info(
                f"데이터 분할: train={len(train_features)}, "
                f"val={len(val_features)} (split={val_ratio:.2f})"
            )

            # 특징 정규화
            self.scaler.fit(train_features)
            normalized_train = self.scaler.transform(train_features)

            # PCA 적용 (선택사항)
            self.use_pca = use_pca
            if self.use_pca:
                self.pca.fit(normalized_train)
                normalized_train = self.pca.transform(normalized_train)
                self.logger.info(
                    f"PCA 적용: {train_features.shape[1]} -> "
                    f"{normalized_train.shape[1]} 차원"
                )

            # 이상 검출 모델 훈련
            self.detector.fit(normalized_train)
            self.is_trained = True

            self.logger.info(
                f"모델 훈련 완료: {len(train_features)}개 샘플, 알고리즘: {self.algorithm}"
            )

            # 검증 메트릭 계산
            self.training_metrics = self._evaluate_training(train_features, val_features)
            if self.training_metrics:
                self.logger.info(
                    f"검증 결과 - train_anomaly_rate: "
                    f"{self.training_metrics.get('train_anomaly_rate', 0):.3f}, "
                    f"val_anomaly_rate: "
                    f"{self.training_metrics.get('val_anomaly_rate', 0):.3f}"
                )

            return True

        except Exception as e:
            self.logger.exception(f"훈련 중 오류 발생: {e}")
            return False

    def _evaluate_training(self, train_features, val_features):
        """train/val 셋에서 정상 데이터 기준 이상 비율 등 기본 메트릭 계산"""
        metrics = {}

        def _anomaly_rate(features):
            if len(features) == 0:
                return None
            transformed = self.scaler.transform(features)
            if self.use_pca:
                transformed = self.pca.transform(transformed)
            preds = self.detector.predict(transformed)
            # sklearn novelty 계열: 1=정상, -1=이상
            return float(np.mean(preds == -1))

        try:
            train_rate = _anomaly_rate(train_features)
            val_rate = _anomaly_rate(val_features)
            if train_rate is not None:
                metrics['train_anomaly_rate'] = train_rate
            if val_rate is not None:
                metrics['val_anomaly_rate'] = val_rate
                # 정상 데이터를 검증셋으로 가정하면 낮을수록 좋음
                metrics['val_false_positive_rate'] = val_rate
            metrics['train_size'] = int(len(train_features))
            metrics['val_size'] = int(len(val_features))
        except Exception as e:
            self.logger.warning(f"검증 메트릭 계산 실패: {e}")

        return metrics
    
    def detect_anomaly(self, features_window):
        """
        이상행동 검출
        
        Args:
            features_window (list): 특징 윈도우 (최근 N개 특징)
            
        Returns:
            tuple: (anomaly_score, is_anomaly, confidence)
        """
        if not self.is_trained:
            return 0.0, False, 0.0
        
        if len(features_window) < self.window_size:
            return 0.0, False, 0.0
        
        try:
            # 최근 윈도우 특징 추출
            recent_features = np.array(features_window[-self.window_size:])
            
            # 유효성 검사
            if np.any(np.isnan(recent_features)) or np.any(np.isinf(recent_features)):
                return 0.0, False, 0.0
            
            # 특징 정규화
            normalized_features = self.scaler.transform(recent_features)
            
            # PCA 적용 (훈련 시 사용했다면)
            if self.use_pca:
                normalized_features = self.pca.transform(normalized_features)
            
            # 이상 점수 계산
            if self.algorithm == 'lof':
                # LOF는 predict 대신 decision_function 사용
                anomaly_scores = self.detector.decision_function(normalized_features)
                predictions = self.detector.predict(normalized_features)
                is_anomaly_array = predictions == -1
            else:
                # 다른 알고리즘들
                anomaly_scores = self.detector.decision_function(normalized_features)
                if hasattr(self.detector, 'predict'):
                    predictions = self.detector.predict(normalized_features)
                    is_anomaly_array = predictions == -1
                else:
                    # threshold 기반 판단
                    is_anomaly_array = anomaly_scores < 0
            
            # 평균 점수 계산
            avg_anomaly_score = np.mean(anomaly_scores)
            
            # 이상 여부 판단 (과반수 투표 또는 평균 기준)
            anomaly_ratio = np.mean(is_anomaly_array)
            is_anomaly = anomaly_ratio > self.anomaly_ratio_threshold
            
            # 점수를 0-1 범위로 정규화 (높을수록 이상)
            if self.algorithm == 'isolation_forest':
                # Isolation Forest: 음수가 이상, 양수가 정상
                anomaly_probability = max(0, min(1, (0.5 - avg_anomaly_score)))
            else:
                # 기타 알고리즘: 음수가 이상, 양수가 정상
                anomaly_probability = max(0, min(1, (0.5 - avg_anomaly_score * 0.5)))
            
            # 신뢰도 계산 (점수의 일관성)
            score_std = np.std(anomaly_scores)
            confidence = max(0, min(1, 1 - score_std))
            
            # 통계 업데이트
            self.stats['total_samples'] += 1
            if is_anomaly:
                self.stats['anomaly_count'] += 1
            else:
                self.stats['normal_count'] += 1
            
            return anomaly_probability, is_anomaly, confidence
            
        except Exception as e:
            self.logger.error(f"이상 검출 중 오류: {e}")
            return 0.0, False, 0.0
    
    def update_contamination(self, new_contamination):
        """오염률 동적 업데이트"""
        if 0.01 <= new_contamination <= 0.5:
            self.contamination = new_contamination
            self.detector = self._create_detector()
            if self.is_trained:
                # 재훈련 필요
                self.is_trained = False
                self.logger.info(f"오염률 업데이트: {new_contamination}, 재훈련 필요")
    
    def get_anomaly_threshold(self, percentile=90):
        """이상 임계값 계산"""
        if not self.is_trained or len(self.feature_history) == 0:
            return 0.5
        
        try:
            features_array = np.array(list(self.feature_history))
            normalized_features = self.scaler.transform(features_array)
            
            if self.use_pca:
                normalized_features = self.pca.transform(normalized_features)
            
            scores = self.detector.decision_function(normalized_features)
            threshold = np.percentile(-scores, percentile)  # 높은 percentile = 더 엄격한 기준
            
            return threshold
            
        except Exception:
            return 0.5
    
    def save_model(self, filepath):
        """모델 저장"""
        try:
            model_data = {
                'scaler': self.scaler,
                'detector': self.detector,
                'pca': self.pca if self.use_pca else None,
                'is_trained': self.is_trained,
                'use_pca': self.use_pca,
                'window_size': self.window_size,
                'contamination': self.contamination,
                'algorithm': self.algorithm,
                'stats': self.stats,
                'anomaly_ratio_threshold': self.anomaly_ratio_threshold,
                'validation_split': self.validation_split,
                'min_training_samples': self.min_training_samples,
                'random_seed': self.random_seed,
                'training_metrics': self.training_metrics
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"모델 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, filepath):
        """모델 로드"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.detector = model_data['detector']
            self.pca = model_data.get('pca')
            self.is_trained = model_data['is_trained']
            self.use_pca = model_data.get('use_pca', False)
            self.window_size = model_data['window_size']
            self.contamination = model_data['contamination']
            self.algorithm = model_data['algorithm']
            self.stats = model_data.get('stats', {})
            self.anomaly_ratio_threshold = model_data.get(
                'anomaly_ratio_threshold', self.anomaly_ratio_threshold)
            self.validation_split = model_data.get(
                'validation_split', self.validation_split)
            self.min_training_samples = model_data.get(
                'min_training_samples', self.min_training_samples)
            self.random_seed = model_data.get('random_seed', self.random_seed)
            self.training_metrics = model_data.get('training_metrics', {})
            
            self.logger.info(f"모델 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            return False
    
    def get_statistics(self):
        """통계 정보 반환"""
        stats = self.stats.copy()
        if stats['total_samples'] > 0:
            stats['anomaly_rate'] = stats['anomaly_count'] / stats['total_samples']
            stats['normal_rate'] = stats['normal_count'] / stats['total_samples']
        else:
            stats['anomaly_rate'] = 0
            stats['normal_rate'] = 0

        stats['training_samples'] = len(self.feature_history)
        stats['is_trained'] = self.is_trained
        stats['algorithm'] = self.algorithm
        stats['training_metrics'] = dict(self.training_metrics)

        return stats

    def evaluate(self, labeled_features, labels):
        """
        레이블된 데이터로 Precision/Recall/F1/Accuracy 계산

        Args:
            labeled_features (np.ndarray): shape (N, D) 특징 배열
            labels (np.ndarray): shape (N,) 정답 레이블 (1=이상, 0=정상)

        Returns:
            dict: 분류 지표
        """
        if not self.is_trained:
            self.logger.warning("evaluate 호출 전에 모델을 훈련해야 합니다.")
            return {}

        features = np.asarray(labeled_features, dtype=np.float32)
        labels = np.asarray(labels).astype(int)

        if len(features) == 0 or len(features) != len(labels):
            self.logger.warning("평가 데이터 크기가 유효하지 않습니다.")
            return {}

        try:
            transformed = self.scaler.transform(features)
            if self.use_pca:
                transformed = self.pca.transform(transformed)

            preds = self.detector.predict(transformed)
            pred_anomaly = (preds == -1).astype(int)

            tp = int(np.sum((pred_anomaly == 1) & (labels == 1)))
            fp = int(np.sum((pred_anomaly == 1) & (labels == 0)))
            tn = int(np.sum((pred_anomaly == 0) & (labels == 0)))
            fn = int(np.sum((pred_anomaly == 0) & (labels == 1)))

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0.0)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'support': total
            }
            self.logger.info(
                f"평가 결과 - acc={accuracy:.3f}, prec={precision:.3f}, "
                f"recall={recall:.3f}, f1={f1:.3f}"
            )
            return metrics
        except Exception as e:
            self.logger.exception(f"평가 중 오류: {e}")
            return {}


class EnsembleAnomalyDetector:
    """앙상블 이상행동 검출기"""

    def __init__(self, algorithms=None, window_size=30, contamination=0.1,
                 ensemble_vote_threshold=0.5, anomaly_ratio_threshold=0.3,
                 random_seed=42):
        """
        앙상블 검출기 초기화

        Args:
            algorithms (list): 사용할 알고리즘 리스트
            window_size (int): 특징 윈도우 크기
            contamination (float): 이상치 비율
            ensemble_vote_threshold (float): 가중 평균 점수 이상 판단 기준
            anomaly_ratio_threshold (float): 개별 검출기 이상 비율 임계값
            random_seed (int): 난수 시드
        """
        if algorithms is None:
            algorithms = ['isolation_forest', 'one_class_svm', 'elliptic_envelope']

        self.algorithms = algorithms
        self.ensemble_vote_threshold = ensemble_vote_threshold
        self.detectors = {}
        self.weights = {}

        # 각 알고리즘별 검출기 생성
        for algorithm in algorithms:
            self.detectors[algorithm] = AnomalyDetector(
                window_size=window_size,
                contamination=contamination,
                algorithm=algorithm,
                anomaly_ratio_threshold=anomaly_ratio_threshold,
                random_seed=random_seed
            )
            self.weights[algorithm] = 1.0  # 초기 가중치는 동일

        self.is_trained = False
        self.logger = logging.getLogger(__name__)
    
    def add_training_features(self, features):
        """모든 검출기에 훈련 특징 추가"""
        for detector in self.detectors.values():
            detector.add_training_features(features)
    
    def train(self, use_pca=False):
        """모든 검출기 훈련"""
        success_count = 0
        
        for algorithm, detector in self.detectors.items():
            if detector.train(use_pca):
                success_count += 1
                self.logger.info(f"{algorithm} 훈련 성공")
            else:
                self.logger.warning(f"{algorithm} 훈련 실패")
        
        self.is_trained = success_count > 0
        
        # 가중치 조정 (훈련 성공한 모델들만)
        if self.is_trained:
            successful_algorithms = [alg for alg, det in self.detectors.items() if det.is_trained]
            equal_weight = 1.0 / len(successful_algorithms)
            
            for algorithm in self.algorithms:
                if algorithm in successful_algorithms:
                    self.weights[algorithm] = equal_weight
                else:
                    self.weights[algorithm] = 0.0
        
        self.logger.info(f"앙상블 훈련 완료: {success_count}/{len(self.algorithms)} 성공")
        return self.is_trained
    
    def detect_anomaly(self, features_window, voting_method='weighted_average'):
        """
        앙상블 이상 검출
        
        Args:
            features_window (list): 특징 윈도우
            voting_method (str): 투표 방식 ('weighted_average', 'majority_vote', 'max_score')
            
        Returns:
            tuple: (anomaly_score, is_anomaly, confidence)
        """
        if not self.is_trained:
            return 0.0, False, 0.0
        
        results = {}
        
        # 각 검출기의 결과 수집
        for algorithm, detector in self.detectors.items():
            if detector.is_trained and self.weights[algorithm] > 0:
                score, is_anomaly, confidence = detector.detect_anomaly(features_window)
                results[algorithm] = {
                    'score': score,
                    'is_anomaly': is_anomaly,
                    'confidence': confidence,
                    'weight': self.weights[algorithm]
                }
        
        if not results:
            return 0.0, False, 0.0
        
        # 앙상블 결과 계산
        if voting_method == 'weighted_average':
            total_weight = sum(r['weight'] for r in results.values())
            weighted_score = sum(r['score'] * r['weight'] for r in results.values()) / total_weight
            weighted_confidence = sum(r['confidence'] * r['weight'] for r in results.values()) / total_weight
            
            # 가중 평균 기준으로 이상 여부 판단
            is_anomaly = weighted_score > self.ensemble_vote_threshold

            return weighted_score, is_anomaly, weighted_confidence
        
        elif voting_method == 'majority_vote':
            anomaly_votes = sum(1 for r in results.values() if r['is_anomaly'])
            total_votes = len(results)
            
            is_anomaly = anomaly_votes > total_votes / 2
            avg_score = sum(r['score'] for r in results.values()) / total_votes
            avg_confidence = sum(r['confidence'] for r in results.values()) / total_votes
            
            return avg_score, is_anomaly, avg_confidence
        
        elif voting_method == 'max_score':
            max_result = max(results.values(), key=lambda x: x['score'])
            return max_result['score'], max_result['is_anomaly'], max_result['confidence']
        
        else:
            raise ValueError(f"지원하지 않는 투표 방식: {voting_method}")
    
    def update_weights(self, feedback_data):
        """
        피드백 기반 가중치 업데이트
        
        Args:
            feedback_data (dict): 각 알고리즘별 성능 피드백
                {'algorithm': {'correct': int, 'total': int}, ...}
        """
        total_accuracy = 0
        algorithm_accuracies = {}
        
        for algorithm, data in feedback_data.items():
            if algorithm in self.algorithms and data['total'] > 0:
                accuracy = data['correct'] / data['total']
                algorithm_accuracies[algorithm] = accuracy
                total_accuracy += accuracy
        
        if total_accuracy > 0:
            # 정확도에 비례하여 가중치 조정
            for algorithm in self.algorithms:
                if algorithm in algorithm_accuracies:
                    self.weights[algorithm] = algorithm_accuracies[algorithm] / total_accuracy * len(algorithm_accuracies)
                else:
                    self.weights[algorithm] = 0.1  # 기본 최소 가중치
            
            self.logger.info(f"가중치 업데이트 완료: {self.weights}")
    
    def save_ensemble(self, filepath):
        """앙상블 모델 저장"""
        try:
            ensemble_data = {
                'algorithms': self.algorithms,
                'weights': self.weights,
                'is_trained': self.is_trained,
                'detectors': {}
            }
            
            # 각 검출기 저장
            for algorithm, detector in self.detectors.items():
                detector_filepath = f"{filepath}_{algorithm}.pkl"
                if detector.save_model(detector_filepath):
                    ensemble_data['detectors'][algorithm] = detector_filepath
            
            # 앙상블 메타데이터 저장
            with open(f"{filepath}_ensemble.json", 'w') as f:
                json.dump(ensemble_data, f, indent=2)
            
            self.logger.info(f"앙상블 모델 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"앙상블 모델 저장 실패: {e}")
            return False
    
    def load_ensemble(self, filepath):
        """앙상블 모델 로드"""
        try:
            # 앙상블 메타데이터 로드
            with open(f"{filepath}_ensemble.json", 'r') as f:
                ensemble_data = json.load(f)
            
            self.algorithms = ensemble_data['algorithms']
            self.weights = ensemble_data['weights']
            self.is_trained = ensemble_data['is_trained']
            
            # 각 검출기 로드
            self.detectors = {}
            for algorithm, detector_filepath in ensemble_data['detectors'].items():
                detector = AnomalyDetector(algorithm=algorithm)
                if detector.load_model(detector_filepath):
                    self.detectors[algorithm] = detector
            
            self.logger.info(f"앙상블 모델 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"앙상블 모델 로드 실패: {e}")
            return False
    
    def get_individual_results(self, features_window):
        """각 검출기의 개별 결과 반환"""
        individual_results = {}
        
        for algorithm, detector in self.detectors.items():
            if detector.is_trained:
                score, is_anomaly, confidence = detector.detect_anomaly(features_window)
                individual_results[algorithm] = {
                    'score': score,
                    'is_anomaly': is_anomaly,
                    'confidence': confidence,
                    'weight': self.weights[algorithm]
                }
        
        return individual_results


class AdaptiveAnomalyDetector(AnomalyDetector):
    """적응형 이상행동 검출기"""

    def __init__(self, window_size=30, contamination=0.1, adaptation_rate=0.01,
                 adaptive_alpha=1.5, adaptive_threshold_min=0.1,
                 adaptive_threshold_max=0.9, false_positive_limit=0.2,
                 false_negative_limit=0.2, f1_warning_threshold=0.6,
                 initial_threshold=0.5, random_seed=42):
        """
        적응형 검출기 초기화

        Args:
            adaptation_rate (float): 적응 속도 (0.001~0.1)
            adaptive_alpha (float): 임계값 산출용 표준편차 배수
            adaptive_threshold_min/max (float): 임계값 상/하한
            false_positive_limit (float): FPR 상한 (초과 시 임계값 상향)
            false_negative_limit (float): FNR 상한 (초과 시 임계값 하향)
            f1_warning_threshold (float): F1 경고 기준
            initial_threshold (float): 초기 임계값
            random_seed (int): 난수 시드
        """
        super().__init__(window_size, contamination, 'isolation_forest',
                         random_seed=random_seed)
        self.adaptation_rate = adaptation_rate
        self.adaptive_alpha = adaptive_alpha
        self.adaptive_threshold_min = adaptive_threshold_min
        self.adaptive_threshold_max = adaptive_threshold_max
        self.false_positive_limit = false_positive_limit
        self.false_negative_limit = false_negative_limit
        self.f1_warning_threshold = f1_warning_threshold
        self.recent_scores = deque(maxlen=1000)
        self.feedback_buffer = deque(maxlen=100)

        # 적응형 임계값
        self.adaptive_threshold = initial_threshold
        self.threshold_history = deque(maxlen=100)
        
        # 성능 추적
        self.performance_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }
    
    def detect_anomaly_adaptive(self, features_window, true_label=None):
        """
        적응형 이상 검출
        
        Args:
            features_window (list): 특징 윈도우
            true_label (bool): 실제 레이블 (피드백용, 선택사항)
            
        Returns:
            tuple: (anomaly_score, is_anomaly, confidence, adaptive_threshold)
        """
        # 기본 검출 수행
        score, _, confidence = super().detect_anomaly(features_window)
        
        # 최근 점수 저장
        self.recent_scores.append(score)
        
        # 적응형 임계값 계산
        if len(self.recent_scores) >= 50:
            recent_scores_array = np.array(list(self.recent_scores))
            
            # 점수 분포 기반 임계값 조정
            mean_score = np.mean(recent_scores_array)
            std_score = np.std(recent_scores_array)
            
            # 동적 임계값 (평균 + α * 표준편차)
            dynamic_threshold = mean_score + self.adaptive_alpha * std_score

            # 점진적 적응
            self.adaptive_threshold = (
                (1 - self.adaptation_rate) * self.adaptive_threshold +
                self.adaptation_rate * dynamic_threshold
            )

            # 임계값 범위 제한
            self.adaptive_threshold = max(
                self.adaptive_threshold_min,
                min(self.adaptive_threshold_max, self.adaptive_threshold)
            )
            
            self.threshold_history.append(self.adaptive_threshold)
        
        # 적응형 임계값으로 이상 여부 판단
        is_anomaly_adaptive = score > self.adaptive_threshold
        
        # 피드백이 있는 경우 성능 업데이트
        if true_label is not None:
            self.update_performance_metrics(is_anomaly_adaptive, true_label)
            self.feedback_buffer.append((score, is_anomaly_adaptive, true_label))
            
            # 주기적 재훈련 조건 확인
            if len(self.feedback_buffer) >= 50:
                self.adapt_model()
        
        return score, is_anomaly_adaptive, confidence, self.adaptive_threshold
    
    def update_performance_metrics(self, predicted, actual):
        """성능 메트릭 업데이트"""
        if predicted and actual:
            self.performance_metrics['true_positives'] += 1
        elif predicted and not actual:
            self.performance_metrics['false_positives'] += 1
        elif not predicted and actual:
            self.performance_metrics['false_negatives'] += 1
        else:
            self.performance_metrics['true_negatives'] += 1
    
    def get_performance_metrics(self):
        """성능 메트릭 계산"""
        tp = self.performance_metrics['true_positives']
        fp = self.performance_metrics['false_positives']
        tn = self.performance_metrics['true_negatives']
        fn = self.performance_metrics['false_negatives']
        
        total = tp + fp + tn + fn
        
        if total == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_samples': total
        }
    
    def adapt_model(self):
        """모델 적응 (재훈련 또는 매개변수 조정)"""
        if len(self.feedback_buffer) < 20:
            return
        
        # 최근 피드백 분석
        recent_feedback = list(self.feedback_buffer)[-50:]
        false_positive_rate = sum(1 for _, pred, actual in recent_feedback 
                                if pred and not actual) / len(recent_feedback)
        false_negative_rate = sum(1 for _, pred, actual in recent_feedback 
                                if not pred and actual) / len(recent_feedback)
        
        # 오탐지율이 높으면 임계값 상향 조정
        if false_positive_rate > self.false_positive_limit:
            self.adaptive_threshold = min(self.adaptive_threshold_max,
                                          self.adaptive_threshold * 1.1)
            self.logger.info(f"오탐지율 높음 -> 임계값 상향: {self.adaptive_threshold:.3f}")

        # 미탐지율이 높으면 임계값 하향 조정
        elif false_negative_rate > self.false_negative_limit:
            self.adaptive_threshold = max(self.adaptive_threshold_min,
                                          self.adaptive_threshold * 0.9)
            self.logger.info(f"미탐지율 높음 -> 임계값 하향: {self.adaptive_threshold:.3f}")

        # 성능이 현저히 떨어지면 재훈련 고려
        metrics = self.get_performance_metrics()
        if metrics['f1'] < self.f1_warning_threshold and metrics['total_samples'] > 100:
            self.logger.warning("성능 저하 감지 - 재훈련을 고려하세요")
    
    def reset_adaptation(self):
        """적응 상태 초기화"""
        self.recent_scores.clear()
        self.feedback_buffer.clear()
        self.adaptive_threshold = (self.adaptive_threshold_min +
                                   self.adaptive_threshold_max) / 2
        self.threshold_history.clear()
        self.performance_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0
        }


class RealTimeAnomalyDetector:
    """실시간 이상행동 검출기"""

    def __init__(self, base_detector, buffer_size=100, alert_threshold=0.7,
                 consecutive_threshold=3, recent_avg_threshold=0.6,
                 recent_avg_window=10):
        """
        실시간 검출기 초기화

        Args:
            base_detector: 기본 이상 검출기
            buffer_size (int): 실시간 버퍼 크기
            alert_threshold (float): 알림 임계값
            consecutive_threshold (int): 연속 이상 알림 발동 횟수
            recent_avg_threshold (float): 최근 평균 점수 알림 기준
            recent_avg_window (int): 최근 평균 계산 윈도우
        """
        self.base_detector = base_detector
        self.buffer_size = buffer_size
        self.alert_threshold = alert_threshold
        self.consecutive_threshold = consecutive_threshold
        self.recent_avg_threshold = recent_avg_threshold
        self.recent_avg_window = recent_avg_window
        
        # 실시간 버퍼
        self.score_buffer = deque(maxlen=buffer_size)
        self.alert_buffer = deque(maxlen=50)
        
        # 알림 상태
        self.is_alerting = False
        self.alert_start_time = None
        self.consecutive_anomalies = 0
        
        # 콜백 함수들
        self.alert_callbacks = []
        
        self.logger = logging.getLogger(__name__)
    
    def add_alert_callback(self, callback):
        """알림 콜백 함수 추가"""
        self.alert_callbacks.append(callback)
    
    def detect_realtime(self, features_window, timestamp=None):
        """
        실시간 이상 검출
        
        Args:
            features_window (list): 특징 윈도우
            timestamp: 타임스탬프 (선택사항)
            
        Returns:
            dict: 실시간 검출 결과
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        
        # 기본 검출 수행
        score, is_anomaly, confidence = self.base_detector.detect_anomaly(features_window)
        
        # 점수 버퍼 업데이트
        self.score_buffer.append({
            'score': score,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'timestamp': timestamp
        })
        
        # 연속 이상 카운트 업데이트
        if is_anomaly:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0
        
        # 알림 조건 확인
        alert_triggered = self._check_alert_conditions(score, is_anomaly)
        
        # 결과 구성
        result = {
            'score': score,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'timestamp': timestamp,
            'consecutive_anomalies': self.consecutive_anomalies,
            'alert_triggered': alert_triggered,
            'is_alerting': self.is_alerting,
            'recent_trend': self._get_recent_trend()
        }
        
        return result
    
    def _check_alert_conditions(self, score, is_anomaly):
        """알림 조건 확인"""
        alert_triggered = False

        # 조건 1: 높은 이상 점수
        if score > self.alert_threshold:
            alert_triggered = True

        # 조건 2: 연속 이상 검출
        elif self.consecutive_anomalies >= self.consecutive_threshold:
            alert_triggered = True

        # 조건 3: 최근 평균 점수가 높음
        elif len(self.score_buffer) >= self.recent_avg_window:
            recent_scores = [item['score']
                             for item in list(self.score_buffer)[-self.recent_avg_window:]]
            if np.mean(recent_scores) > self.recent_avg_threshold:
                alert_triggered = True
        
        # 알림 상태 업데이트
        if alert_triggered and not self.is_alerting:
            self._trigger_alert()
        elif not alert_triggered and self.is_alerting:
            self._clear_alert()
        
        return alert_triggered
    
    def _trigger_alert(self):
        """알림 발생"""
        import time
        self.is_alerting = True
        self.alert_start_time = time.time()
        
        # 알림 버퍼에 기록
        self.alert_buffer.append({
            'timestamp': self.alert_start_time,
            'type': 'alert_start',
            'consecutive_anomalies': self.consecutive_anomalies,
            'recent_scores': list(self.score_buffer)[-5:] if self.score_buffer else []
        })
        
        # 콜백 실행
        for callback in self.alert_callbacks:
            try:
                callback('alert_start', {
                    'timestamp': self.alert_start_time,
                    'consecutive_anomalies': self.consecutive_anomalies
                })
            except Exception as e:
                self.logger.error(f"알림 콜백 오류: {e}")
        
        self.logger.warning(f"이상행동 알림 발생: 연속 {self.consecutive_anomalies}회")
    
    def _clear_alert(self):
        """알림 해제"""
        import time
        if self.is_alerting:
            alert_duration = time.time() - self.alert_start_time
            self.is_alerting = False
            
            # 알림 버퍼에 기록
            self.alert_buffer.append({
                'timestamp': time.time(),
                'type': 'alert_end',
                'duration': alert_duration
            })
            
            # 콜백 실행
            for callback in self.alert_callbacks:
                try:
                    callback('alert_end', {
                        'duration': alert_duration
                    })
                except Exception as e:
                    self.logger.error(f"알림 콜백 오류: {e}")
            
            self.logger.info(f"이상행동 알림 해제: 지속시간 {alert_duration:.1f}초")
    
    def _get_recent_trend(self):
        """최근 트렌드 분석"""
        if len(self.score_buffer) < 10:
            return 'insufficient_data'
        
        recent_scores = [item['score'] for item in list(self.score_buffer)[-10:]]
        
        # 트렌드 계산 (선형 회귀)
        x = np.arange(len(recent_scores))
        coeffs = np.polyfit(x, recent_scores, 1)
        slope = coeffs[0]
        
        if slope > 0.05:
            return 'increasing'
        elif slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_alert_history(self, last_n=10):
        """최근 알림 히스토리 반환"""
        return list(self.alert_buffer)[-last_n:]
    
    def get_realtime_statistics(self):
        """실시간 통계 정보"""
        if not self.score_buffer:
            return {}
        
        recent_data = list(self.score_buffer)
        scores = [item['score'] for item in recent_data]
        anomaly_count = sum(1 for item in recent_data if item['is_anomaly'])
        
        return {
            'buffer_size': len(self.score_buffer),
            'recent_avg_score': np.mean(scores),
            'recent_max_score': np.max(scores),
            'anomaly_ratio': anomaly_count / len(recent_data),
            'consecutive_anomalies': self.consecutive_anomalies,
            'is_alerting': self.is_alerting,
            'total_alerts': len(self.alert_buffer),
            'recent_trend': self._get_recent_trend()
        }