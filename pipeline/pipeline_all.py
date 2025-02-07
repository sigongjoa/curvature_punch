import os
import subprocess
import sys
import argparse
import numpy as np
from utils.animation import create_curvature_video, create_curvature_animation

def run_pipeline(video_path):
    """
    단일 비디오에 대해 포즈 예측 파이프라인 실행.
    
    :param video_path: 처리할 비디오 파일 경로
    """
    # AlphaPose 디렉토리로 이동
    alphapose_dir = r'D:\pose_prediction\AlphaPose'
    save_dir = r'D:\pose_prediction'
    os.chdir(alphapose_dir)

    # 비디오 파일 이름 추출 (확장자 제외)
    video_name = os.path.basename(video_path).split('.')[0]

    # output 폴더 경로 설정
    output_dir = os.path.join(save_dir, 'results', video_name)

    # output 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # AlphaPose 실행
    command = [
        sys.executable, 'scripts/demo_inference.py', 
        '--cfg', 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
        '--checkpoint', 'pretrained_models/halpe26_fast_res50_256x192.pth',
        '--video', video_path,
        '--outdir', output_dir
    ]
    subprocess.run(command)

    # MotionBERT 디렉토리로 이동
    motionbert_dir = r'D:\pose_prediction\MotionBERT'
    os.chdir(motionbert_dir)

    # 결과 JSON 경로 설정
    json_path = os.path.join(output_dir, 'alphapose-results.json')

    # MotionBERT 실행
    command_infer = [
        sys.executable, 'infer_wild.py', 
        '--vid_path', video_path, 
        '--json_path', json_path,
        '--out_path', output_dir
    ]
    subprocess.run(command_infer)

    command_infer_mesh = [
        sys.executable, 'infer_wild_mesh.py', 
        '--vid_path', video_path, 
        '--json_path', json_path,
        '--out_path', output_dir
    ]
    subprocess.run(command_infer_mesh)

    # Mesh 데이터 로드 및 애니메이션 생성
    mesh_path = os.path.join(output_dir, 'mesh.npy')
    mesh = np.load(mesh_path)

    create_curvature_video(mesh, os.path.join(output_dir, "curvature_video.mp4"))
    create_curvature_animation(mesh, output_dir)

    print(f"✅ '{video_name}' 처리 완료!\n")


def run_pipeline_for_all_videos(video_dir):
    """
    특정 디렉터리 내 모든 비디오 파일을 대상으로 파이프라인 실행.

    :param video_dir: 비디오 파일이 저장된 폴더 경로
    """
    # 지원하는 비디오 확장자 목록
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    # 디렉토리 내 모든 파일 검색
    video_files = [f for f in os.listdir(video_dir) if f.endswith(video_extensions)]
    
    if not video_files:
        print("⚠️ 처리할 비디오 파일이 없습니다.")
        return

    print(f"🎥 총 {len(video_files)}개의 비디오를 처리합니다.\n")

    for video in video_files:
        video_path = os.path.join(video_dir, video)
        print(f"▶ '{video}' 처리 시작...")
        run_pipeline(video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose prediction pipeline for all videos in a directory.")
    parser.add_argument('--video_dir', type=str, default=r'D:\pose_prediction\video_split', help='Path to the video directory')

    args = parser.parse_args()

    # 디렉터리 내 모든 비디오 처리
    run_pipeline_for_all_videos(args.video_dir)
