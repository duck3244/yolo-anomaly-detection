#!/usr/bin/env python3
"""
test_with_real_data.py
실제 CCTV 데이터로 이상행동 검출 시스템 테스트
"""

import cv2
import numpy as np
import json
import os
from main_system import YOLOAnomalyDetectionSystem
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_annotation_data(json_path):
    """JSON 어노테이션 파일 로드"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"어노테이션 로드 실패: {e}")
        return None

def visualize_annotations(image_path, annotation_data, output_path=None):
    """어노테이션 시각화"""
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로드 실패: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # matplotlib으로 시각화
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image_rgb)
    ax.set_title("실제 CCTV 데이터 - 어노테이션 시각화")
    
    # 어노테이션 그리기
    annotations = annotation_data.get('annotations', [])
    
    colors = {'사람': 'green', '몰카': 'red', 'default': 'blue'}
    
    for ann in annotations:
        class_name = ann.get('class', 'unknown')
        bbox_data = ann.get('data', {})
        
        x = bbox_data.get('x', 0)
        y = bbox_data.get('y', 0)
        width = bbox_data.get('width', 0)
        height = bbox_data.get('height', 0)
        
        color = colors.get(class_name, colors['default'])
        
        # 바운딩 박스 그리기
        rect = patches.Rectangle((x, y), width, height, 
                                linewidth=3, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 레이블 추가
        ax.text(x, y-10, f"{class_name} ({width}x{height})", 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
               fontsize=12, color='white', weight='bold')
    
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"시각화 결과 저장: {output_path}")
    
    plt.show()
    return fig

def convert_annotations_to_detections(annotation_data):
    """어노테이션을 시스템 검출 형식으로 변환"""
    annotations = annotation_data.get('annotations', [])
    detections = []
    
    for ann in annotations:
        class_name = ann.get('class', '')
        if class_name == '사람':  # 사람만 검출 결과로 변환
            bbox_data = ann.get('data', {})
            x = bbox_data.get('x', 0)
            y = bbox_data.get('y', 0)
            width = bbox_data.get('width', 0)
            height = bbox_data.get('height', 0)
            
            # YOLO 형식으로 변환 (x1, y1, x2, y2)
            detection = {
                'bbox': [x, y, x + width, y + height],
                'confidence': 0.9,  # 실제 어노테이션이므로 높은 신뢰도
                'class_id': 0,
                'class_name': 'person',
                'annotation_id': ann.get('id', '')
            }
            detections.append(detection)
    
    return detections

def find_anomaly_region(annotation_data):
    """어노테이션에서 이상 객체 찾기"""
    annotations = annotation_data.get('annotations', [])
    
    for ann in annotations:
        class_name = ann.get('class', '')
        if class_name == '몰카':  # 이상 객체 발견
            bbox_data = ann.get('data', {})
            return {
                'class': class_name,
                'bbox': [
                    bbox_data.get('x', 0),
                    bbox_data.get('y', 0),
                    bbox_data.get('x', 0) + bbox_data.get('width', 0),
                    bbox_data.get('y', 0) + bbox_data.get('height', 0)
                ],
                'annotation_id': ann.get('id', '')
            }
    
    return None

def test_system_with_real_image(image_path, annotation_path):
    """실제 이미지로 시스템 테스트"""
    print("🧪 실제 CCTV 데이터로 시스템 테스트 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    annotation_data = load_annotation_data(annotation_path)
    if annotation_data is None:
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로드 실패: {image_path}")
        return False
    
    print(f"✅ 이미지 로드 성공: {image.shape}")
    print(f"✅ 어노테이션 로드 성공: {len(annotation_data.get('annotations', []))}개 객체")
    
    # 2. 어노테이션 분석
    print("\n2. 어노테이션 분석...")
    annotations = annotation_data.get('annotations', [])
    person_count = sum(1 for ann in annotations if ann.get('class') == '사람')
    anomaly_objects = [ann for ann in annotations if ann.get('class') != '사람']
    
    print(f"   검출된 사람: {person_count}명")
    print(f"   이상 객체: {len(anomaly_objects)}개")
    
    for obj in anomaly_objects:
        print(f"     - {obj.get('class', 'unknown')}: {obj.get('data', {})}")
    
    # 3. 시스템 초기화
    print("\n3. 시스템 초기화...")
    system = YOLOAnomalyDetectionSystem(device='cpu')
    print("✅ 시스템 초기화 완료")
    
    # 4. 실제 어노테이션으로 검출 시뮬레이션
    print("\n4. 어노테이션 기반 검출 시뮬레이션...")
    real_detections = convert_annotations_to_detections(annotation_data)
    print(f"✅ {len(real_detections)}개 사람 검출 시뮬레이션")
    
    # 5. 시스템으로 프레임 처리 (실제 검출기 사용)
    print("\n5. 시스템 검출기로 실제 처리...")
    
    # 고해상도 이미지를 처리 가능한 크기로 리사이즈
    original_h, original_w = image.shape[:2]
    target_w, target_h = 1920, 1080  # Full HD로 리사이즈
    
    scale_x = target_w / original_w
    scale_y = target_h / original_h
    
    resized_image = cv2.resize(image, (target_w, target_h))
    
    # 시스템으로 처리
    process_result = system.process_frame(resized_image, frame_number=0)
    
    print(f"✅ 시스템 검출 결과:")
    print(f"   총 검출: {process_result['total_detections']}개")
    print(f"   추적 객체: {process_result['total_tracked']}개")
    print(f"   처리 시간: {process_result['processing_time']:.3f}초")
    
    # 6. 결과 비교
    print("\n6. 결과 비교...")
    print(f"   실제 사람 수: {person_count}명")
    print(f"   시스템 검출: {process_result['total_detections']}개")
    print(f"   실제 이상 객체: {len(anomaly_objects)}개")
    print(f"   시스템 이상 검출: {len([r for r in process_result['anomaly_results'].values() if r['is_anomaly']])}개")
    
    # 7. 시각화
    print("\n7. 결과 시각화...")
    
    # 원본 어노테이션 시각화
    visualize_annotations(image_path, annotation_data, "annotation_visualization.png")
    
    # 시스템 결과 시각화
    result_frame = system.draw_results(resized_image, process_result)
    cv2.imwrite("system_result.jpg", result_frame)
    print("✅ 시스템 결과 저장: system_result.jpg")
    
    # 8. 상세 분석
    print("\n8. 상세 분석...")
    anomaly_region = find_anomaly_region(annotation_data)
    
    if anomaly_region:
        print(f"⚠️  실제 이상 영역 발견:")
        print(f"   클래스: {anomaly_region['class']}")
        print(f"   위치: {anomaly_region['bbox']}")
        
        # 해당 영역 근처에서 이상행동이 검출되었는지 확인
        anomaly_bbox = anomaly_region['bbox']
        anomaly_center = ((anomaly_bbox[0] + anomaly_bbox[2]) / 2 * scale_x,
                         (anomaly_bbox[1] + anomaly_bbox[3]) / 2 * scale_y)
        
        nearby_anomalies = []
        for person_id, result in process_result['anomaly_results'].items():
            if result['is_anomaly']:
                person_bbox = result['bbox']
                person_center = ((person_bbox[0] + person_bbox[2]) / 2,
                               (person_bbox[1] + person_bbox[3]) / 2)
                
                distance = np.sqrt((anomaly_center[0] - person_center[0])**2 + 
                                 (anomaly_center[1] - person_center[1])**2)
                
                if distance < 300:  # 300픽셀 내
                    nearby_anomalies.append({
                        'person_id': person_id,
                        'distance': distance,
                        'score': result['anomaly_score']
                    })
        
        if nearby_anomalies:
            print(f"   ✅ 이상 영역 근처에서 {len(nearby_anomalies)}개 이상행동 검출!")
            for anomaly in nearby_anomalies:
                print(f"     - 사람 {anomaly['person_id']}: 거리 {anomaly['distance']:.1f}px, 점수 {anomaly['score']:.2f}")
        else:
            print(f"   ⚠️  이상 영역 근처에서 이상행동 미검출")
    
    print("\n" + "=" * 60)
    print("🎉 실제 데이터 테스트 완료!")
    
    return True

def create_training_video_from_image(image_path, output_video_path, duration=10, fps=10):
    """이미지에서 훈련용 비디오 생성 (정상 행동 시뮬레이션)"""
    print(f"📹 훈련용 비디오 생성: {output_video_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print("이미지 로드 실패")
        return False
    
    # 처리 가능한 크기로 리사이즈
    image = cv2.resize(image, (640, 480))
    h, w = image.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        # 원본 이미지 복사
        frame = image.copy()
        
        # 시간에 따른 정상적인 움직임 시뮬레이션
        t = frame_idx / fps
        
        # 여러 명의 사람이 정상적으로 이동하는 것처럼 시뮬레이션
        for i in range(3):  # 3명의 가상 사람
            # 좌우 이동
            x = int(100 + i * 200 + 50 * np.sin(t * 0.5 + i))
            y = int(200 + i * 50 + 20 * np.cos(t * 0.3 + i))
            
            # 사람 모양 시뮬레이션 (사각형)
            cv2.rectangle(frame, (x, y), (x + 30, y + 60), (0, 255, 0), -1)
            
            # 움직임 궤적
            if frame_idx > 5:
                prev_x = int(100 + i * 200 + 50 * np.sin((t - 0.5) * 0.5 + i))
                prev_y = int(200 + i * 50 + 20 * np.cos((t - 0.5) * 0.3 + i))
                cv2.line(frame, (prev_x + 15, prev_y + 30), (x + 15, y + 30), (0, 255, 0), 2)
        
        # 프레임 정보 추가
        cv2.putText(frame, f"Training Frame: {frame_idx}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Normal Behavior Simulation", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ 훈련용 비디오 생성 완료: {total_frames}프레임")
    return True

def main():
    """메인 테스트 함수"""
    # 파일 경로 설정 (실제 파일명으로 수정하세요)
    image_path = "22443290_00_(SSM Server_2번 출입구 계단_중단_AI(192.168.15.116)_20211223_151224_153340_ID_0001).jpg"
    annotation_path = "22443290_00_(SSM Server_2번 출입구 계단_중단_AI(192.168.15.116)_20211223_151224_153340_ID_0001).json"
    
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일이 없습니다: {image_path}")
        print("파일명을 확인하고 같은 폴더에 저장해주세요.")
        return
    
    if not os.path.exists(annotation_path):
        print(f"❌ JSON 파일이 없습니다: {annotation_path}")
        print("파일명을 확인하고 같은 폴더에 저장해주세요.")
        return
    
    print("🎯 실제 CCTV 데이터 테스트 시작!")
    print(f"이미지: {image_path}")
    print(f"어노테이션: {annotation_path}")
    print()
    
    # 1. 단일 이미지 테스트
    success = test_system_with_real_image(image_path, annotation_path)
    
    if success:
        # 2. 훈련용 비디오 생성 및 테스트
        print("\n" + "="*60)
        print("📚 훈련 시나리오 테스트")
        print("="*60)
        
        train_video_path = "training_video_from_real_data.mp4"
        if create_training_video_from_image(image_path, train_video_path):
            print("\n훈련용 비디오로 모델 훈련 테스트...")
            
            system = YOLOAnomalyDetectionSystem(device='cpu')
            train_success = system.train_on_video(train_video_path, max_frames=50)
            
            if train_success:
                # 모델 저장
                model_path = "real_data_trained_model.pkl"
                system.save_model(model_path)
                print(f"✅ 실제 데이터 기반 모델 훈련 완료: {model_path}")
                
                # 다시 실제 이미지로 테스트
                print("\n훈련된 모델로 실제 이미지 재테스트...")
                trained_system = YOLOAnomalyDetectionSystem(device='cpu')
                trained_system.load_model(model_path)
                
                # 테스트 처리
                test_image = cv2.imread(image_path)
                test_image = cv2.resize(test_image, (640, 480))
                
                result = trained_system.process_frame(test_image)
                result_frame = trained_system.draw_results(test_image, result)
                
                cv2.imwrite("trained_model_result.jpg", result_frame)
                print("✅ 훈련된 모델 결과 저장: trained_model_result.jpg")
                
                # 통계 출력
                stats = trained_system.get_statistics()
                print(f"\n📊 최종 통계:")
                print(f"   검출률: {stats['detection_rate']:.2f}")
                print(f"   이상행동률: {stats['anomaly_rate']:.2f}")
                print(f"   평균 FPS: {stats['avg_fps']:.1f}")
    
    print("\n🎉 모든 테스트 완료!")
    print("\n생성된 파일들:")
    print("  - annotation_visualization.png (어노테이션 시각화)")
    print("  - system_result.jpg (시스템 검출 결과)")
    print("  - training_video_from_real_data.mp4 (훈련용 비디오)")
    print("  - real_data_trained_model.pkl (훈련된 모델)")
    print("  - trained_model_result.jpg (훈련된 모델 결과)")

if __name__ == "__main__":
    main()
