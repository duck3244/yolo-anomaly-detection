# utils.py - 유틸리티 함수들
"""
utils.py
시스템에서 사용되는 유틸리티 함수들
"""

import cv2
import numpy as np
import json
import logging
import time
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def setup_directories():
    """필요한 디렉토리 생성"""
    directories = ['models', 'data/train', 'data/test', 'data/output', 'logs', 'screenshots', 'results']

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("✅ 디렉토리 구조 생성 완료")


def validate_video_file(video_path):
    """비디오 파일 유효성 검사"""
    if not os.path.exists(video_path):
        return False, "파일이 존재하지 않습니다."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "비디오 파일을 열 수 없습니다."

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if frame_count <= 0:
        return False, "유효한 프레임이 없습니다."

    info = {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': frame_count / fps if fps > 0 else 0
    }

    return True, info


def create_test_video(output_path, duration=30, fps=30):
    """테스트용 비디오 생성"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    total_frames = duration * fps

    for frame_idx in range(total_frames):
        # 배경 생성
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)

        # 시간 정보
        t = frame_idx / fps

        # 정상 행동 시뮬레이션
        if t < duration * 0.7:  # 70%는 정상
            # 사람 1: 좌우 이동
            x1 = int(100 + (t * 30) % 400)
            y1 = 200
            cv2.rectangle(frame, (x1, y1), (x1 + 40, y1 + 80), (0, 255, 0), -1)

            # 사람 2: 상하 이동
            x2 = 300
            y2 = int(150 + 50 * np.sin(t * 0.8))
            cv2.rectangle(frame, (x2, y2), (x2 + 35, y2 + 75), (0, 255, 0), -1)

        else:  # 30%는 이상 행동
            # 급격한 움직임
            x3 = int(200 + 150 * np.sin(t * 10))
            y3 = int(200 + 100 * np.cos(t * 8))
            cv2.rectangle(frame, (x3, y3), (x3 + 50, y3 + 90), (0, 0, 255), -1)

        # 프레임 번호 표시
        cv2.putText(frame, f"Frame: {frame_idx}, Time: {t:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ 테스트 비디오 생성 완료: {output_path}")


def calculate_iou(box1, box2):
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


def non_max_suppression(boxes, scores, score_threshold=0.5, iou_threshold=0.5):
    """Non-Maximum Suppression"""
    indices = cv2.dnn.NMSBoxes(
        boxes, scores, score_threshold, iou_threshold
    )

    if len(indices) > 0:
        return indices.flatten()
    else:
        return []


def resize_with_padding(image, target_size):
    """종횡비를 유지하면서 패딩으로 리사이즈"""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 스케일 계산
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 리사이즈
    resized = cv2.resize(image, (new_w, new_h))

    # 패딩 추가
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded, scale, (x_offset, y_offset)


def save_detection_results(results, output_path):
    """검출 결과를 JSON 파일로 저장"""
    serializable_results = []

    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_result[key] = float(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results_count': len(serializable_results),
            'results': serializable_results
        }, f, indent=2)


def load_detection_results(input_path):
    """JSON 파일에서 검출 결과 로드"""
    with open(input_path, 'r') as f:
        data = json.load(f)

    return data['results']


def create_performance_report(stats, output_path):
    """성능 리포트 생성"""
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_runtime': stats.get('total_runtime', 0),
            'total_frames': stats.get('total_frames', 0),
            'average_fps': stats.get('avg_fps', 0),
            'total_detections': stats.get('total_detections', 0),
            'total_anomalies': stats.get('total_anomalies', 0),
            'anomaly_rate': stats.get('anomaly_rate', 0)
        },
        'detailed_stats': stats
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ 성능 리포트 생성: {output_path}")


def plot_statistics(stats, save_path=None):
    """통계 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('YOLO Anomaly Detection System Statistics', fontsize=16)

    # FPS 히스토리
    if 'processing_times' in stats and stats['processing_times']:
        fps_history = [1.0 / t for t in stats['processing_times'] if t > 0]
        axes[0, 0].plot(fps_history)
        axes[0, 0].set_title('FPS History')
        axes[0, 0].set_ylabel('FPS')
        axes[0, 0].grid(True)

    # 검출 통계
    categories = ['Total Frames', 'Detections', 'Anomalies']
    values = [
        stats.get('total_frames', 0),
        stats.get('total_detections', 0),
        stats.get('total_anomalies', 0)
    ]
    axes[0, 1].bar(categories, values)
    axes[0, 1].set_title('Detection Statistics')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 이상 검출율
    if stats.get('total_frames', 0) > 0:
        normal_rate = 1 - stats.get('anomaly_rate', 0)
        sizes = [normal_rate, stats.get('anomaly_rate', 0)]
        labels = ['Normal', 'Anomaly']
        colors = ['lightgreen', 'lightcoral']
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[1, 0].set_title('Normal vs Anomaly Ratio')

    # 성능 지표
    perf_labels = ['Avg FPS', 'Detection Rate', 'Processing Time (ms)']
    perf_values = [
        stats.get('avg_fps', 0),
        stats.get('detection_rate', 0) * 100,  # 백분율로 변환
        stats.get('avg_processing_time', 0) * 1000  # 밀리초로 변환
    ]
    axes[1, 1].bar(perf_labels, perf_values)
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 통계 그래프 저장: {save_path}")

    plt.show()


def check_system_requirements():
    """시스템 요구사항 확인"""
    import torch
    import platform
    import psutil

    print("=== 시스템 요구사항 확인 ===")

    # Python 버전
    python_version = platform.python_version()
    print(f"Python 버전: {python_version}")

    # PyTorch 정보
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"사용 가능한 GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"총 메모리: {memory.total / (1024 ** 3):.1f} GB")
    print(f"사용 가능한 메모리: {memory.available / (1024 ** 3):.1f} GB")

    # CPU 정보
    cpu_count = psutil.cpu_count()
    print(f"CPU 코어 수: {cpu_count}")

    # 운영체제
    print(f"운영체제: {platform.system()} {platform.release()}")

    print("=" * 30)


def benchmark_system(detector, test_frames=None, iterations=10):
    """시스템 벤치마크"""
    if test_frames is None:
        # 테스트 프레임 생성
        test_frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_frames.append(frame)

    print(f"벤치마크 시작: {len(test_frames)}개 프레임, {iterations}회 반복")

    # 워밍업
    for frame in test_frames[:3]:
        _ = detector.detect_persons(frame)

    # 벤치마크 실행
    times = []
    detection_counts = []

    for i in range(iterations):
        start_time = time.time()
        total_detections = 0

        for frame in test_frames:
            detections = detector.detect_persons(frame)
            total_detections += len(detections)

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        detection_counts.append(total_detections)

        print(f"반복 {i + 1}/{iterations}: {elapsed_time:.3f}초, {total_detections}개 검출")

    # 결과 분석
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_fps = len(test_frames) / avg_time
    avg_detections = np.mean(detection_counts)

    results = {
        'avg_processing_time': avg_time,
        'std_processing_time': std_time,
        'avg_fps': avg_fps,
        'min_fps': len(test_frames) / np.max(times),
        'max_fps': len(test_frames) / np.min(times),
        'avg_detections': avg_detections,
        'total_iterations': iterations,
        'total_frames': len(test_frames) * iterations
    }

    print("\n=== 벤치마크 결과 ===")
    print(f"평균 처리시간: {avg_time:.3f}±{std_time:.3f}초")
    print(f"평균 FPS: {avg_fps:.1f}")
    print(f"FPS 범위: {results['min_fps']:.1f} ~ {results['max_fps']:.1f}")
    print(f"평균 검출 수: {avg_detections:.1f}개/프레임")

    return results


class Logger:
    """커스텀 로거 클래스"""

    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 파일 핸들러
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)


def create_config_template(output_path='config.json'):
    """설정 파일 템플릿 생성"""
    config_template = {
        "system": {
            "device": "cpu",
            "model_path": "yolov8n.pt",
            "confidence_threshold": 0.5,
            "log_level": "INFO",
            "use_optimized_detector": False
        },
        "tracking": {
            "max_disappeared": 10,
            "max_distance": 100,
            "feature_history_length": 30,
            "use_advanced_tracking": False
        },
        "feature_extraction": {
            "use_advanced_features": False,
            "window_size": 30,
            "include_interaction_features": False,
            "include_context_features": False
        },
        "anomaly_detection": {
            "algorithm": "isolation_forest",
            "window_size": 30,
            "contamination": 0.1,
            "use_ensemble": False,
            "use_realtime": True,
            "use_adaptive": False
        },
        "display": {
            "show_id": True,
            "show_score": True,
            "show_bbox": True,
            "show_confidence": True,
            "anomaly_color": [0, 0, 255],
            "normal_color": [0, 255, 0],
            "font_scale": 0.6,
            "thickness": 2
        },
        "output": {
            "save_video": True,
            "save_logs": True,
            "save_screenshots": False,
            "output_directory": "output",
            "log_directory": "logs"
        },
        "performance": {
            "frame_skip": 1,
            "batch_size": 1,
            "use_multithreading": False,
            "roi_enabled": False,
            "roi_boxes": []
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_template, f, indent=2, ensure_ascii=False)

    print(f"✅ 설정 파일 템플릿 생성: {output_path}")