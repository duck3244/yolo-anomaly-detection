#!/usr/bin/env python3
"""
full_mp4_test.py
완전한 MP4 테스트 시나리오 실행 스크립트
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse

def run_command(cmd, description):
    """명령어 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"명령어: {cmd}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ 성공! (소요시간: {elapsed_time:.1f}초)")
            if result.stdout:
                print("출력:")
                print(result.stdout)
        else:
            print(f"❌ 실패! (소요시간: {elapsed_time:.1f}초)")
            print("오류:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 명령어 실행 실패: {e}")
        return False
    
    return True

def check_requirements():
    """필요한 파일들 확인"""
    print("📋 요구사항 확인 중...")
    
    required_files = [
        'main_system.py',
        'convert_image_to_video.py',
        'config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 누락된 파일들: {missing_files}")
        return False
    
    print("✅ 모든 필수 파일 확인됨")
    return True

def test_scenario_1_basic():
    """시나리오 1: 기본 테스트"""
    print("\n" + "🎯" + "="*50)
    print("시나리오 1: 기본 객체 검출 테스트")
    print("="*51)
    
    # 1. 테스트 비디오 생성
    if not run_command(
        "python create_test_video.py --type normal --duration 30", 
        "정상 행동 테스트 비디오 생성"
    ):
        return False
    
    # 2. 기본 객체 검출 (모델 없이)
    if not run_command(
        "python main_system.py --mode video --input test_videos/normal_behavior.mp4 --output results/basic_detection.mp4",
        "기본 객체 검출 테스트"
    ):
        return False
    
    print("✅ 시나리오 1 완료!")
    return True

def test_scenario_2_training():
    """시나리오 2: 모델 훈련 및 테스트"""
    print("\n" + "🎓" + "="*50)
    print("시나리오 2: 모델 훈련 및 이상 검출 테스트")
    print("="*51)
    
    # 1. 정상 행동 비디오로 훈련
    if not run_command(
        "python main_system.py --mode train --train_video test_videos/normal_behavior.mp4 --model_save models/test_model.pkl",
        "정상 행동 패턴으로 모델 훈련"
    ):
        return False
    
    # 2. 이상 행동 비디오 생성
    if not run_command(
        "python create_test_video.py --type anomaly --duration 20",
        "이상 행동 테스트 비디오 생성"
    ):
        return False
    
    # 3. 훈련된 모델로 이상 검출
    if not run_command(
        "python main_system.py --mode video --input test_videos/anomaly_behavior.mp4 --model_load models/test_model.pkl --output results/anomaly_detection.mp4",
        "훈련된 모델로 이상 검출 테스트"
    ):
        return False
    
    print("✅ 시나리오 2 완료!")
    return True

def test_scenario_3_comprehensive():
    """시나리오 3: 종합 성능 테스트"""
    print("\n" + "🏆" + "="*50)
    print("시나리오 3: 종합 성능 테스트")
    print("="*51)
    
    # 1. 혼합 행동 비디오 생성
    if not run_command(
        "python create_test_video.py --type mixed --duration 60",
        "혼합 행동 테스트 비디오 생성"
    ):
        return False
    
    # 2. 혼합 비디오로 종합 테스트
    if not run_command(
        "python main_system.py --mode video --input test_videos/mixed_behavior.mp4 --model_load models/test_model.pkl --output results/comprehensive_test.mp4 --display",
        "종합 성능 테스트 (화면 표시 포함)"
    ):
        return False
    
    print("✅ 시나리오 3 완료!")
    return True

def test_scenario_4_image_conversion():
    """시나리오 4: 이미지 변환 테스트"""
    print("\n" + "🖼️" + "="*50)
    print("시나리오 4: 이미지-비디오 변환 테스트")
    print("="*51)
    
    # 1. 테스트 이미지 폴더가 있다면 변환
    image_folders = ['images', 'test_images', 'data/test']
    
    for folder in image_folders:
        if Path(folder).exists() and any(Path(folder).glob('*.jpg')):
            if not run_command(
                f"python convert_image_to_video.py {folder} -o results/converted_from_images.mp4 --fps 10 --duration 2",
                f"이미지 폴더 '{folder}'를 비디오로 변환"
            ):
                continue
            
            # 변환된 비디오 테스트
            if not run_command(
                "python main_system.py --mode video --input results/converted_from_images.mp4 --output results/converted_test.mp4",
                "변환된 비디오 테스트"
            ):
                continue
            
            print("✅ 시나리오 4 완료!")
            return True
    
    print("⚠️ 시나리오 4: 테스트 이미지 폴더를 찾을 수 없어 건너뜀")
    return True

def test_scenario_5_performance():
    """시나리오 5: 성능 벤치마크"""
    print("\n" + "⚡" + "="*50)
    print("시나리오 5: 성능 벤치마크")
    print("="*51)
    
    # GPU 테스트 (가능한 경우)
    if not run_command(
        "python main_system.py --mode video --input test_videos/normal_behavior.mp4 --device cuda --output results/gpu_test.mp4 2>/dev/null || python main_system.py --mode video --input test_videos/normal_behavior.mp4 --device cpu --output results/cpu_test.mp4",
        "GPU/CPU 성능 테스트"
    ):
        return False
    
    print("✅ 시나리오 5 완료!")
    return True

def generate_test_report():
    """테스트 리포트 생성"""
    print("\n" + "📊" + "="*50)
    print("테스트 리포트 생성")
    print("="*51)
    
    results_dir = Path('results')
    if not results_dir.exists():
        print("❌ 결과 폴더가 없습니다.")
        return
    
    report_lines = [
        "# YOLO 이상행동 검출 시스템 - 테스트 리포트",
        f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 생성된 파일들",
        ""
    ]
    
    # 결과 파일들 나열
    for result_file in results_dir.glob('*'):
        if result_file.is_file():
            size_mb = result_file.stat().st_size / (1024 * 1024)
            report_lines.append(f"- {result_file.name} ({size_mb:.1f}MB)")
    
    report_lines.extend([
        "",
        "## 테스트 비디오들",
        ""
    ])
    
    # 테스트 비디오들 나열
    test_videos_dir = Path('test_videos')
    if test_videos_dir.exists():
        for video_file in test_videos_dir.glob('*.mp4'):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            report_lines.append(f"- {video_file.name} ({size_mb:.1f}MB)")
    
    report_lines.extend([
        "",
        "## 추가 테스트 방법",
        "",
        "### 실시간 웹캠 테스트",
        "```bash",
        "python main_system.py --mode webcam --model_load models/test_model.pkl",
        "```",
        "",
        "### 사용자 정의 설정으로 테스트",
        "```bash",
        "python main_system.py --mode video --input your_video.mp4 --config custom_config.json",
        "```",
        "",
        "### 고성능 테스트 (GPU)",
        "```bash",
        "python main_system.py --mode video --input test_video.mp4 --device cuda --model yolov8s.pt",
        "```"
    ])
    
    # 리포트 저장
    report_path = results_dir / 'test_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📋 테스트 리포트 저장: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLO 이상행동 검출 시스템 완전 테스트')
    parser.add_argument('--scenario', choices=['1', '2', '3', '4', '5', 'all'], 
                        default='all', help='실행할 테스트 시나리오')
    parser.add_argument('--skip-check', action='store_true', help='요구사항 확인 건너뛰기')
    
    args = parser.parse_args()
    
    print("🚀 YOLO 이상행동 검출 시스템 - 완전 테스트 시작!")
    print("=" * 60)
    
    # 디렉토리 생성
    for directory in ['test_videos', 'results', 'models']:
        Path(directory).mkdir(exist_ok=True)
    
    # 요구사항 확인
    if not args.skip_check and not check_requirements():
        print("❌ 요구사항 확인 실패")
        sys.exit(1)
    
    # 시나리오 실행
    scenarios = {
        '1': test_scenario_1_basic,
        '2': test_scenario_2_training,
        '3': test_scenario_3_comprehensive,
        '4': test_scenario_4_image_conversion,
        '5': test_scenario_5_performance
    }
    
    success_count = 0
    total_count = 0
    
    if args.scenario == 'all':
        test_scenarios = scenarios.values()
    else:
        test_scenarios = [scenarios[args.scenario]]
    
    start_time = time.time()
    
    for scenario_func in test_scenarios:
        total_count += 1
        if scenario_func():
            success_count += 1
        else:
            print(f"⚠️ {scenario_func.__name__} 실패")
    
    total_time = time.time() - start_time
    
    # 최종 결과
    print("\n" + "🎉" + "="*50)
    print("테스트 완료!")
    print("="*51)
    print(f"성공: {success_count}/{total_count}")
    print(f"총 소요시간: {total_time:.1f}초")
    
    # 테스트 리포트 생성
    generate_test_report()
    
    print(f"\n📁 결과 파일들:")
    results_dir = Path('results')
    if results_dir.exists():
        for result_file in results_dir.glob('*'):
            if result_file.is_file():
                size_mb = result_file.stat().st_size / (1024 * 1024)
                print(f"  - {result_file.name} ({size_mb:.1f}MB)")
    
    print(f"\n🎬 테스트 비디오들:")
    test_videos_dir = Path('test_videos')
    if test_videos_dir.exists():
        for video_file in test_videos_dir.glob('*.mp4'):
            size_mb = video_file.stat().st_size / (1024 * 1024)
            print(f"  - {video_file.name} ({size_mb:.1f}MB)")
    
    if success_count == total_count:
        print(f"\n🎊 모든 테스트 성공! 시스템이 정상 작동합니다.")
    else:
        print(f"\n⚠️ 일부 테스트 실패. 로그를 확인하세요.")

if __name__ == "__main__":
    main()