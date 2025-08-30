#!/usr/bin/env python3
"""
create_test_video.py
테스트용 MP4 비디오 생성 스크립트
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def create_normal_behavior_video(output_path, duration=60, fps=30):
    """정상 행동 패턴 비디오 생성"""
    print(f"📹 정상 행동 비디오 생성: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        # 배경 생성 (회색)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (80, 80, 80)
        
        # 시간 정보
        t = frame_idx / fps
        
        # 정상적인 사람 움직임 시뮬레이션 (3명)
        for person_id in range(3):
            # 좌우 이동 (다른 속도와 위치)
            x_base = 50 + person_id * 200
            y_base = 150 + person_id * 80
            
            x = int(x_base + 100 * np.sin(t * 0.3 + person_id * np.pi / 3))
            y = int(y_base + 30 * np.cos(t * 0.2 + person_id * np.pi / 4))
            
            # 사람 모양 (사각형)
            color = [(0, 255, 0), (0, 200, 0), (0, 150, 0)][person_id]
            cv2.rectangle(frame, (x, y), (x + 30, y + 60), color, -1)
            
            # ID 표시
            cv2.putText(frame, f'P{person_id+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 프레임 정보 표시
        cv2.putText(frame, f"Normal Behavior - Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        # 진행상황 출력
        if frame_idx % (fps * 10) == 0:
            print(f"  진행: {frame_idx}/{total_frames} ({t:.1f}초)")
    
    out.release()
    print(f"✅ 정상 행동 비디오 완성: {total_frames}프레임")

def create_anomaly_behavior_video(output_path, duration=30, fps=30):
    """이상 행동 패턴 비디오 생성"""
    print(f"🚨 이상 행동 비디오 생성: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 80)  # 어두운 배경
        
        t = frame_idx / fps
        
        # 처음 10초는 정상, 나머지는 이상 행동
        if t < 10:
            # 정상 행동
            x = int(100 + t * 20)
            y = 200
            color = (0, 255, 0)
            cv2.putText(frame, "Normal Phase", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # 이상 행동들
            anomaly_type = int((t - 10) / 5) % 4  # 5초마다 다른 이상행동
            
            if anomaly_type == 0:  # 급격한 움직임
                x = int(300 + 200 * np.sin(t * 15))
                y = int(200 + 150 * np.cos(t * 12))
                color = (0, 0, 255)
                cv2.putText(frame, "ANOMALY: Erratic Movement", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            elif anomaly_type == 1:  # 정지 (오랜 시간)
                x, y = 320, 240
                color = (255, 0, 0)
                cv2.putText(frame, "ANOMALY: Loitering", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
            elif anomaly_type == 2:  # 빠른 이동
                x = int(50 + (t - 10) * 100 % 540)
                y = int(100 + 50 * np.sin(t * 8))
                color = (255, 255, 0)
                cv2.putText(frame, "ANOMALY: Fast Movement", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
            else:  # 크기 변화 (가까이/멀리)
                scale = 0.5 + 0.8 * (1 + np.sin(t * 6)) / 2
                size = int(40 * scale)
                x, y = 320, 240
                color = (255, 0, 255)
                cv2.putText(frame, "ANOMALY: Size Change", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # 사람 그리기
        size = locals().get('size', 40)
        cv2.rectangle(frame, (x-size//2, y-size//2), 
                     (x+size//2, y+size//2), color, -1)
        
        # 프레임 정보
        cv2.putText(frame, f"Frame: {frame_idx}, Time: {t:.1f}s", (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ 이상 행동 비디오 완성: {total_frames}프레임")

def create_mixed_behavior_video(output_path, duration=90, fps=30):
    """정상+이상 혼합 비디오 생성"""
    print(f"🎭 혼합 행동 비디오 생성: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)
        
        t = frame_idx / fps
        
        # 여러 사람이 다른 행동을 보임
        behaviors = []
        
        # 사람 1: 대부분 정상, 가끔 이상
        if int(t) % 20 < 15:  # 75% 정상
            x1 = int(100 + t * 15 % 400)
            y1 = 150
            color1 = (0, 255, 0)
            behaviors.append("Normal")
        else:  # 25% 이상
            x1 = int(250 + 150 * np.sin(t * 10))
            y1 = int(150 + 100 * np.cos(t * 8))
            color1 = (0, 0, 255)
            behaviors.append("Anomaly")
        
        # 사람 2: 주로 정상 행동
        x2 = int(300 + 80 * np.sin(t * 0.5))
        y2 = int(300 + 40 * np.cos(t * 0.3))
        color2 = (0, 255, 0)
        behaviors.append("Normal")
        
        # 사람 3: 무작위로 이상 행동
        if np.random.random() < 0.15:  # 15% 확률로 이상
            x3 = int(np.random.randint(50, 590))
            y3 = int(np.random.randint(50, 430))
            color3 = (255, 0, 0)
            behaviors.append("Random Anomaly")
        else:
            x3 = int(450 + 50 * np.sin(t * 0.8))
            y3 = int(200 + 100 * np.cos(t * 0.4))
            color3 = (0, 200, 0)
            behaviors.append("Normal")
        
        # 사람들 그리기
        people = [(x1, y1, color1, "P1"), (x2, y2, color2, "P2"), (x3, y3, color3, "P3")]
        
        for i, (x, y, color, label) in enumerate(people):
            cv2.rectangle(frame, (x-15, y-30), (x+15, y+30), color, -1)
            cv2.putText(frame, label, (x-10, y-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 상태 정보 표시
        cv2.putText(frame, f"Mixed Behavior Test - Time: {t:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, behavior in enumerate(behaviors):
            color = (0, 255, 0) if "Normal" in behavior else (0, 0, 255)
            cv2.putText(frame, f"P{i+1}: {behavior}", (10, 60 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 이상 행동 카운터
        anomaly_count = sum(1 for b in behaviors if "Anomaly" in b)
        if anomaly_count > 0:
            cv2.putText(frame, f"ANOMALIES DETECTED: {anomaly_count}", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ 혼합 행동 비디오 완성: {total_frames}프레임")

def main():
    parser = argparse.ArgumentParser(description='테스트용 MP4 비디오 생성')
    parser.add_argument('--type', choices=['normal', 'anomaly', 'mixed', 'all'], 
                        default='all', help='생성할 비디오 타입')
    parser.add_argument('--duration', type=int, default=60, help='비디오 길이(초)')
    parser.add_argument('--fps', type=int, default=30, help='초당 프레임 수')
    parser.add_argument('--output_dir', default='test_videos', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎬 테스트용 MP4 비디오 생성 시작!")
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 50)
    
    if args.type in ['normal', 'all']:
        create_normal_behavior_video(
            output_dir / 'normal_behavior.mp4', 
            args.duration, args.fps
        )
    
    if args.type in ['anomaly', 'all']:
        create_anomaly_behavior_video(
            output_dir / 'anomaly_behavior.mp4', 
            args.duration // 2, args.fps
        )
    
    if args.type in ['mixed', 'all']:
        create_mixed_behavior_video(
            output_dir / 'mixed_behavior.mp4', 
            args.duration * 1.5, args.fps
        )
    
    print("\n🎉 테스트 비디오 생성 완료!")
    print(f"📁 생성된 파일들:")
    for video_file in output_dir.glob('*.mp4'):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"  - {video_file.name} ({size_mb:.1f}MB)")
    
    print("\n📋 사용 예시:")
    print("# 1. 정상 행동으로 모델 훈련")
    print(f"python main_system.py --mode train --train_video {output_dir}/normal_behavior.mp4")
    print("\n# 2. 이상 행동 비디오 테스트")
    print(f"python main_system.py --mode video --input {output_dir}/anomaly_behavior.mp4 --model_load trained_model.pkl")
    print("\n# 3. 혼합 비디오로 성능 테스트")
    print(f"python main_system.py --mode video --input {output_dir}/mixed_behavior.mp4 --model_load trained_model.pkl --display")

if __name__ == "__main__":
    main()
