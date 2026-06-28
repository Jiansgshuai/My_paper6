"""
可视化工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import List, Dict, Tuple


def plot_trajectories_2d(trajectories: Dict[int, np.ndarray],
                         title: str = "Vehicle Trajectories",
                         save_path: str = None):
    """绘制2D轨迹"""
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    for i, (track_id, traj) in enumerate(trajectories.items()):
        plt.plot(traj[:, 0], traj[:, 1], '-o',
                color=colors[i], linewidth=2, markersize=4,
                label=f'Track {track_id}')

    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_trajectories_3d(trajectories: Dict[int, np.ndarray],
                         title: str = "Vehicle Trajectories (3D)",
                         save_path: str = None):
    """绘制3D轨迹"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    for i, (track_id, traj) in enumerate(trajectories.items()):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
               '-o', color=colors[i], linewidth=2, markersize=3,
               label=f'Track {track_id}')

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_velocity_profiles(tracks: List,
                           title: str = "Velocity Profiles",
                           save_path: str = None):
    """绘制速度曲线"""
    plt.figure(figsize=(12, 6))

    for track in tracks:
        if len(track.observations) < 2:
            continue

        times = [obs.timestamp for obs in track.observations]
        velocities = []

        for obs in track.observations:
            if obs.velocity is not None:
                velocities.append(np.linalg.norm(obs.velocity))
            else:
                velocities.append(0)

        plt.plot(times, velocities, '-o', linewidth=2,
                markersize=4, label=f'Track {track.track_id}')

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_lane_division(trajectories: Dict[int, np.ndarray],
                      lanes: List,
                      track_to_lane: Dict[int, int],
                      title: str = "Lane Division",
                      save_path: str = None):
    """绘制车道划分结果"""
    plt.figure(figsize=(14, 8))

    colors_lane = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

    # 绘制轨迹
    for track_id, traj in trajectories.items():
        lane_id = track_to_lane.get(track_id, -1)
        color = colors_lane[lane_id % len(colors_lane)]

        plt.plot(traj[:, 0], traj[:, 1], '-',
                color=color, linewidth=2.5, alpha=0.7)
        plt.plot(traj[0, 0], traj[0, 1], 'o',
                color=color, markersize=10, markeredgecolor='black', markeredgewidth=1.5)

    # 绘制车道中心线
    for lane in lanes:
        plt.plot(lane.center_line[:, 0], lane.center_line[:, 1],
                '--', color='black', linewidth=2, alpha=0.5,
                label=f'Lane {lane.lane_id} center')

    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_vehicle_classification(trajectories: Dict[int, np.ndarray],
                                vehicle_types: Dict[int, str],
                                title: str = "Vehicle Classification",
                                save_path: str = None):
    """绘制车型分类结果"""
    plt.figure(figsize=(14, 8))

    type_colors = {'car': 'blue', 'truck': 'red', 'bus': 'green', 'unknown': 'gray'}
    type_markers = {'car': 'o', 'truck': 's', 'bus': '^', 'unknown': 'x'}

    for track_id, traj in trajectories.items():
        vtype = vehicle_types.get(track_id, 'unknown')
        color = type_colors.get(vtype, 'gray')
        marker = type_markers.get(vtype, 'x')

        plt.plot(traj[:, 0], traj[:, 1], '-',
                color=color, linewidth=2, alpha=0.6)

        mid_idx = len(traj) // 2
        plt.scatter(traj[mid_idx, 0], traj[mid_idx, 1],
                   color=color, marker=marker, s=200,
                   edgecolors='black', linewidths=2,
                   label=f'{vtype} (ID {track_id})')

    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_point_cloud_comparison(original: o3d.geometry.PointCloud,
                                     processed: o3d.geometry.PointCloud,
                                     window_name: str = "Point Cloud Comparison"):
    """对比可视化原始和处理后的点云"""
    # 着色
    original.paint_uniform_color([1, 0, 0])  # 红色
    processed.paint_uniform_color([0, 1, 0])  # 绿色

    # 显示
    o3d.visualization.draw_geometries(
        [original, processed],
        window_name=window_name,
        width=1200,
        height=800,
        left=50,
        top=50
    )


def create_coordinate_frame(size: float = 1.0) -> o3d.geometry.TriangleMesh:
    """创建坐标系"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def plot_confusion_matrix(y_true: List[str],
                         y_pred: List[str],
                         classes: List[str],
                         title: str = "Confusion Matrix",
                         save_path: str = None):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classes, yticklabels=classes)

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
