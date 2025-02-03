import os
import subprocess
import sys
import argparse
import numpy as np
from utils.animation import create_curvature_video, create_curvature_animation

def run_pipeline(video_path):
    # AlphaPose 디렉토리로 이동
    alphapose_dir = r'D:\pose_prediction\AlphaPose'
    save_dir = r'D:\pose_prediction'
    os.chdir(alphapose_dir)

    # # 비디오 파일 이름 추출 (확장자 제외)
    video_name = os.path.basename(video_path).split('.')[0]

    # # output 폴더 경로 설정
    output_dir = os.path.join(save_dir, 'results', video_name)

    # output 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # pipeline.py 실행
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

    # # 결과로 생성된 JSON 경로 설정
    json_path = os.path.join(output_dir, 'alphapose-results.json')  # AlphaPose 결과 JSON 파일

    # infer_wild.py 실행
    command_infer = [
        sys.executable, 'infer_wild_mesh.py', 
        '--vid_path', video_path, 
        '--json_path', json_path,
        '--out_path', output_dir
    ]
    
    subprocess.run(command_infer)

    mesh_path = os.path.join(output_dir, 'mesh.npy')
    mesh = np.load(mesh_path)
    # 애니메이션 생성
    # Generate curvature video
    create_curvature_video(mesh, os.path.join(output_dir, "curvature_video.mp4"))

    # Generate curvature animation
    create_curvature_animation(mesh, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose prediction pipeline.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    run_pipeline(args.video)
