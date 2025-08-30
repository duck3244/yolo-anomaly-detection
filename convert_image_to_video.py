import cv2
import os
import glob
from pathlib import Path
import argparse


def create_slideshow_video(input_folder, output_path, fps=30, duration_per_image=2,
                           transition_frames=10, resize_to=None):
    """
    JPG 이미지들을 슬라이드쇼 형태의 MP4 동영상으로 변환

    Args:
        input_folder: 이미지 폴더 경로
        output_path: 출력 MP4 파일 경로
        fps: 초당 프레임 수
        duration_per_image: 각 이미지 표시 시간(초)
        transition_frames: 전환 효과 프레임 수
        resize_to: 리사이즈할 크기 (width, height) 튜플
    """

    # 이미지 파일 찾기
    extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_folder, ext)))

    images = sorted(images)

    if not images:
        print("이미지 파일을 찾을 수 없습니다.")
        return False

    print(f"총 {len(images)}개의 이미지를 찾았습니다.")

    # 첫 번째 이미지로부터 크기 설정
    first_image = cv2.imread(images[0])
    if resize_to:
        width, height = resize_to
        first_image = cv2.resize(first_image, (width, height))
    else:
        height, width = first_image.shape[:2]

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_per_image = int(fps * duration_per_image)

    for i, image_path in enumerate(images):
        print(f"처리 중: {os.path.basename(image_path)} ({i + 1}/{len(images)})")

        img = cv2.imread(image_path)
        if img is None:
            continue

        # 리사이즈
        if resize_to or img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        # 페이드 인 효과 (첫 번째 이미지)
        if i == 0:
            for frame in range(transition_frames):
                alpha = frame / transition_frames
                faded_img = (img * alpha).astype('uint8')
                video_writer.write(faded_img)

        # 일반 프레임들
        for _ in range(frames_per_image - (transition_frames if i == 0 else 0)):
            video_writer.write(img)

        # 페이드 아웃 효과 (마지막 이미지)
        if i == len(images) - 1:
            for frame in range(transition_frames):
                alpha = 1 - (frame / transition_frames)
                faded_img = (img * alpha).astype('uint8')
                video_writer.write(faded_img)

    video_writer.release()
    print(f"동영상 생성 완료: {output_path}")
    return True


# 명령행 인터페이스
def main():
    parser = argparse.ArgumentParser(description='JPG 파일들을 MP4로 변환')
    parser.add_argument('input_folder', help='이미지가 있는 폴더')
    parser.add_argument('-o', '--output', default='slideshow.mp4', help='출력 파일명')
    parser.add_argument('--fps', type=int, default=30, help='초당 프레임 수')
    parser.add_argument('--duration', type=float, default=2.0, help='이미지당 표시 시간')
    parser.add_argument('--width', type=int, help='출력 비디오 너비')
    parser.add_argument('--height', type=int, help='출력 비디오 높이')

    args = parser.parse_args()

    resize_to = None
    if args.width and args.height:
        resize_to = (args.width, args.height)

    create_slideshow_video(
        args.input_folder,
        args.output,
        fps=args.fps,
        duration_per_image=args.duration,
        resize_to=resize_to
    )


if __name__ == "__main__":
    # 직접 실행할 때
    input_folder = "/home/duck/Downloads/대전교통공사_지하철 역사 내 CCTV 이상행동 영상 샘플데이터_20210917/원천데이터_2120699/2120699"  # 이미지 폴더
    output_video = "test.mp4"

    create_slideshow_video(
        input_folder,
        output_video,
        fps=30,
        duration_per_image=3,
        resize_to=(1920, 1080)  # Full HD 크기로 리사이즈
    )