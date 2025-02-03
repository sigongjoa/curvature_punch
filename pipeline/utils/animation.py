import cv2
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


def estimate_curvature_pca(mesh):
    """
    PCA를 사용하여 각 점에 대한 곡률을 추정
    :param mesh: (N, 3) 형태의 포인트 클라우드
    :return: (N,) 형태의 곡률 값
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    curvatures = np.zeros(len(mesh))
    for i, point in enumerate(mesh):
        neighbors = np.asarray(pcd.points)[max(0, i-5): min(len(mesh), i+5)]  # 이웃 점들
        if len(neighbors) < 3:
            continue
        cov_matrix = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eig(cov_matrix)
        curvatures[i] = eigvals.min()  # 최소 고유값을 곡률로 사용
    
    return curvatures

def create_curvature_video(mesh, video_filename, frame_width=800, frame_height=600):
    """
    주어진 메쉬 데이터를 기반으로 곡률 변화 영상을 생성
    :param mesh: 메쉬 데이터 (프레임 별 포인트 클라우드 데이터)
    :param video_filename: 생성될 동영상 파일 경로
    :param frame_width: 동영상의 가로 크기
    :param frame_height: 동영상의 세로 크기
    """
    # 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 30.0, (frame_width, frame_height))

    # 동영상 생성
    num_frames = len(mesh)  # mesh에서 각 프레임의 수
    prev_curvature_values = None  # 이전 프레임의 곡률 저장

    for i in tqdm(range(num_frames)):
        curvature_values = estimate_curvature_pca(mesh[i])
        
        if prev_curvature_values is None:
            delta_curvature = np.zeros_like(curvature_values)  # 첫 프레임은 변화량 없음
        else:
            delta_curvature = np.abs(curvature_values - prev_curvature_values)  # 변화량 계산

        prev_curvature_values = curvature_values  # 현재 곡률을 다음 프레임의 기준으로 저장

        # 변화량이 일정 값 이상이면 빨간색 강조
        threshold = np.percentile(delta_curvature, 90)  # 상위 10% 변화량을 강조
        highlight_color = np.where(delta_curvature > threshold, 'red', 'blue')  # 색상 변경

        # Matplotlib로 시각화
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(mesh[i][:, 0], mesh[i][:, 1], mesh[i][:, 2], 
                             c=delta_curvature, cmap='jet', s=5)

        plt.colorbar(scatter, ax=ax, label="Curvature Change")

        ax.set_title(f"Frame {i} - PCA Curvature Change")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Matplotlib 그래프를 이미지로 저장
        plt.tight_layout()
        fig.canvas.draw()  # 그리기

        # buffer_rgba()로 이미지를 얻고, RGB로 변환
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA로 변환
        img = img[..., :3]  # RGBA에서 RGB로 변환

        # 동영상에 프레임 추가
        out.write(img)

        # 그래프 리셋
        plt.close()

    # 동영상 파일 저장
    out.release()
    cv2.destroyAllWindows()

def estimate_curvature_pca(mesh):
    """
    PCA를 사용하여 각 점에 대한 곡률을 추정
    :param mesh: (N, 3) 형태의 포인트 클라우드
    :return: (N,) 형태의 곡률 값
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    curvatures = np.zeros(len(mesh))
    for i, point in enumerate(mesh):
        neighbors = np.asarray(pcd.points)[max(0, i-5): min(len(mesh), i+5)]  # 이웃 점들
        if len(neighbors) < 3:
            continue
        cov_matrix = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eig(cov_matrix)
        curvatures[i] = eigvals.min()  # 최소 고유값을 곡률로 사용
    
    return curvatures

def create_curvature_animation(mesh, output_dir):
    """
    Plotly를 사용하여 곡률 변화 애니메이션을 생성하고 HTML로 저장
    :param mesh: 메쉬 데이터 (프레임 별 포인트 클라우드 데이터)
    :param output_dir: 결과 저장 디렉토리
    """
    num_frames = len(mesh)
    prev_curvature_values = None
    frames = []

    for i in tqdm(range(num_frames)):
        curvature_values = estimate_curvature_pca(mesh[i])
        
        if prev_curvature_values is None:
            delta_curvature = np.zeros_like(curvature_values)  
        else:
            delta_curvature = np.abs(curvature_values - prev_curvature_values)  

        prev_curvature_values = curvature_values  

        threshold = np.percentile(delta_curvature, 90)
        
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=mesh[i][:, 0],
                y=mesh[i][:, 1],
                z=mesh[i][:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=delta_curvature,
                    colorscale='jet',
                    colorbar=dict(title="Curvature Change"),
                    opacity=0.8
                )
            )],
            name=f"Frame {i}"
        ))

    fig = go.Figure(
        data=[go.Scatter3d(
            x=mesh[0][:, 0],
            y=mesh[0][:, 1],
            z=mesh[0][:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=np.zeros(len(mesh[0])),
                colorscale='jet',
                colorbar=dict(title="Curvature Change"),
                opacity=0.8
            )
        )],
        frames=frames
    )

    fig.update_layout(
        title="3D Curvature Change Visualization",
        width=800,
        height=600,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=200, redraw=True),
                                      fromcurrent=True)]),
                dict(label="❚❚ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        fromcurrent=True)]),
            ]
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[f.name], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))],
                        label=f.name) for f in frames],
            active=0
        )]
    )

    # 결과 HTML 저장 경로 설정
    output_path = os.path.join(output_dir, "curvature_difference_animation.html")
    
    # HTML 파일로 저장
    fig.write_html(output_path)
    print(f"Curvature animation saved at: {output_path}")