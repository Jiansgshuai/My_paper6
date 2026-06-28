"""
完整流程演示：三大创新点集成
整合三个创新方法，展示从原始点云到完整车辆轨迹数据的全流程
"""

import numpy as np
import open3d as o3d
from typing import List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入三个创新模块
from innovation1_svo_background_removal import SVOBackgroundRemover
from innovation2_temporal_association import (
    TemporalAssociator, VehicleObservation, LShapeFitting
)
from innovation3_lane_vehicle_classification import (
    SpatialLaneDivision, SVMVehicleClassifier,
    VehicleFeatureExtractor, generate_synthetic_training_data
)


class VehicleTrajectoryPipeline:
    """车辆轨迹采集完整流程"""

    def __init__(self):
        # 创新1: 背景去除
        self.background_remover = SVOBackgroundRemover(
            voxel_size=0.5,
            background_threshold=5,
            sor_nb_neighbors=20,
            sor_std_ratio=2.0
        )

        # 创新2: 跨帧关联
        self.associator = TemporalAssociator(
            max_distance=3.0,
            max_missing_frames=3,
            angle_weight=0.3,
            dt=0.1
        )

        # 创新3: 车道划分与车型分类
        self.lane_divider = SpatialLaneDivision(num_lanes=None)
        self.vehicle_classifier = SVMVehicleClassifier()

        # 训练车型分类器
        train_features, train_labels = generate_synthetic_training_data()
        self.vehicle_classifier.train(train_features, train_labels)

    def process(self, point_cloud_sequence: List[o3d.geometry.PointCloud]) -> Dict:
        """
        完整处理流程

        Args:
            point_cloud_sequence: 原始点云序列

        Returns:
            完整的轨迹数据字典
        """
        print("=" * 70)
        print("开始完整处理流程")
        print("=" * 70)

        # ===== 步骤1: 背景去除 =====
        print("\n[步骤 1/4] 背景去除 (创新点1)")
        print("-" * 70)
        foreground_sequence = self.background_remover.process_sequence(point_cloud_sequence)
        print(f"背景去除完成，处理 {len(foreground_sequence)} 帧")

        # ===== 步骤2: 车辆检测与聚类 =====
        print("\n[步骤 2/4] 车辆检测与聚类")
        print("-" * 70)
        observations_per_frame = []

        for frame_id, pcd in enumerate(foreground_sequence):
            points = np.asarray(pcd.points)

            if len(points) < 10:
                observations_per_frame.append([])
                continue

            # 简单的DBSCAN聚类
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=1.0, min_samples=10).fit(points)
            labels = clustering.labels_

            # 为每个簇创建观测
            frame_observations = []
            unique_labels = set(labels)

            for label in unique_labels:
                if label == -1:  # 噪声点
                    continue

                mask = labels == label
                cluster_points = points[mask]

                if len(cluster_points) < 50:  # 过滤小簇
                    continue

                centroid = cluster_points.mean(axis=0)

                obs = VehicleObservation(
                    frame_id=frame_id,
                    cluster_id=label,
                    points=cluster_points,
                    centroid=centroid,
                    timestamp=frame_id * 0.1
                )

                frame_observations.append(obs)

            observations_per_frame.append(frame_observations)
            print(f"帧 {frame_id+1}/{len(foreground_sequence)}: 检测到 {len(frame_observations)} 个车辆", end='\r')

        print(f"\n车辆检测完成")

        # ===== 步骤3: 跨帧关联 (创新点2) =====
        print("\n[步骤 3/4] 跨帧关联与轨迹生成 (创新点2)")
        print("-" * 70)

        for frame_obs in observations_per_frame:
            self.associator.process_frame(frame_obs)

        trajectories_dict = self.associator.get_trajectories()
        print(f"生成 {len(trajectories_dict)} 条轨迹")

        # 提取车辆点云
        vehicle_points_dict = {}
        for track in self.associator.tracks:
            if len(track.observations) >= 2:
                # 使用中间帧的点云
                mid_obs = track.observations[len(track.observations) // 2]
                vehicle_points_dict[track.track_id] = mid_obs.points

        # ===== 步骤4: 车道划分与车型分类 (创新点3) =====
        print("\n[步骤 4/4] 车道划分与车型分类 (创新点3)")
        print("-" * 70)

        # 车道划分
        track_to_lane = self.lane_divider.divide_lanes(trajectories_dict)
        print(f"识别 {len(self.lane_divider.lanes)} 条车道")

        # 车型分类
        vehicle_types = {}
        for track_id, points in vehicle_points_dict.items():
            features_dict = VehicleFeatureExtractor.extract_features(points)
            feature_vec = VehicleFeatureExtractor.features_to_vector(features_dict)
            vehicle_type = self.vehicle_classifier.predict(feature_vec)
            vehicle_types[track_id] = vehicle_type

        type_counts = {}
        for vtype in vehicle_types.values():
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

        print("车型统计:")
        for vtype, count in type_counts.items():
            print(f"  {vtype}: {count} 辆")

        # ===== 整合结果 =====
        print("\n" + "=" * 70)
        print("处理完成！生成完整轨迹数据")
        print("=" * 70)

        results = {
            'trajectories': trajectories_dict,
            'lanes': self.lane_divider.lanes,
            'track_to_lane': track_to_lane,
            'vehicle_types': vehicle_types,
            'vehicle_points': vehicle_points_dict,
            'tracks': self.associator.tracks
        }

        return results


def generate_demo_scene(num_frames: int = 30) -> List[o3d.geometry.PointCloud]:
    """生成演示场景"""
    print("生成模拟场景数据...")

    # 背景点云
    background_points = []

    # 地面
    x = np.random.uniform(-30, 30, 8000)
    y = np.random.uniform(-15, 15, 8000)
    z = np.random.uniform(-0.3, 0, 8000)
    background_points.append(np.column_stack([x, y, z]))

    # 路边建筑
    x = np.random.uniform(-32, -30, 2000)
    y = np.random.uniform(-15, 15, 2000)
    z = np.random.uniform(0, 6, 2000)
    background_points.append(np.column_stack([x, y, z]))

    x = np.random.uniform(30, 32, 2000)
    y = np.random.uniform(-15, 15, 2000)
    z = np.random.uniform(0, 6, 2000)
    background_points.append(np.column_stack([x, y, z]))

    background_points = np.vstack(background_points)

    # 车辆配置 (lane_y, type, length, width, height, speed)
    vehicles = [
        (-4, 'car', 4.5, 1.8, 1.5, 2.0),
        (-4, 'truck', 8.0, 2.4, 3.0, 1.5),
        (0, 'car', 4.3, 1.75, 1.45, 2.2),
        (0, 'bus', 11.0, 2.6, 3.2, 1.8),
        (4, 'car', 4.6, 1.85, 1.55, 2.1),
        (4, 'car', 4.2, 1.7, 1.4, 2.3),
    ]

    # 生成点云序列
    pcd_sequence = []

    for t in range(num_frames):
        frame_points = [background_points.copy()]

        # 为每辆车生成点云
        for v_idx, (lane_y, vtype, length, width, height, speed) in enumerate(vehicles):
            # 车辆位置
            start_x = -25 + v_idx * 8
            vehicle_x = start_x + t * speed
            vehicle_y = lane_y + np.random.normal(0, 0.15)
            vehicle_z = height / 2

            # 生成车辆点云
            vehicle_points = []
            for dx in np.linspace(-length/2, length/2, int(length * 5)):
                for dy in np.linspace(-width/2, width/2, int(width * 5)):
                    for dz in np.linspace(0, height, int(height * 5)):
                        vehicle_points.append([
                            vehicle_x + dx,
                            vehicle_y + dy,
                            vehicle_z + dz - height/2
                        ])

            vehicle_points = np.array(vehicle_points)
            vehicle_points += np.random.normal(0, 0.02, vehicle_points.shape)
            frame_points.append(vehicle_points)

        # 合并所有点
        all_points = np.vstack(frame_points)

        # 添加噪声
        noise = np.random.normal(0, 0.01, all_points.shape)
        all_points += noise

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)

        pcd_sequence.append(pcd)

    print(f"生成 {num_frames} 帧场景，每帧约 {len(all_points)} 个点")

    return pcd_sequence


def visualize_results(results: Dict):
    """可视化完整结果"""
    print("\n生成可视化结果...")

    fig = plt.figure(figsize=(18, 10))

    # 子图1: 俯视图 - 车道划分
    ax1 = fig.add_subplot(221)
    colors_lane = ['red', 'green', 'blue', 'orange', 'purple']

    trajectories = results['trajectories']
    track_to_lane = results['track_to_lane']

    for track_id, traj in trajectories.items():
        lane_id = track_to_lane.get(track_id, 0)
        color = colors_lane[lane_id % len(colors_lane)]
        ax1.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, alpha=0.7)
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8)

    # 车道中心线
    for lane in results['lanes']:
        ax1.plot(lane.center_line[:, 0], lane.center_line[:, 1],
                '--k', linewidth=1.5, alpha=0.5)

    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('车道划分结果', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 子图2: 车型分类
    ax2 = fig.add_subplot(222)
    type_colors = {'car': 'blue', 'truck': 'red', 'bus': 'green'}
    type_markers = {'car': 'o', 'truck': 's', 'bus': '^'}

    vehicle_types = results['vehicle_types']

    for track_id, traj in trajectories.items():
        vtype = vehicle_types.get(track_id, 'unknown')
        color = type_colors.get(vtype, 'gray')
        marker = type_markers.get(vtype, 'x')

        ax2.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, alpha=0.6)
        ax2.scatter(traj[len(traj)//2, 0], traj[len(traj)//2, 1],
                   color=color, marker=marker, s=200,
                   edgecolors='black', linewidths=2)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=10, label='Car'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=10, label='Truck'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=10, label='Bus')
    ]
    ax2.legend(handles=legend_elements, fontsize=10)

    ax2.set_xlabel('X (m)', fontsize=11)
    ax2.set_ylabel('Y (m)', fontsize=11)
    ax2.set_title('车型分类结果', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 子图3: 3D轨迹
    ax3 = fig.add_subplot(223, projection='3d')

    for track_id, traj in trajectories.items():
        lane_id = track_to_lane.get(track_id, 0)
        color = colors_lane[lane_id % len(colors_lane)]
        ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                '-', color=color, linewidth=2)

    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_zlabel('Z (m)', fontsize=10)
    ax3.set_title('车辆轨迹 (3D)', fontsize=13, fontweight='bold')

    # 子图4: 统计信息
    ax4 = fig.add_subplot(224)
    ax4.axis('off')

    # 生成统计文本
    stats_text = "=" * 35 + "\n"
    stats_text += "    处理结果统计\n"
    stats_text += "=" * 35 + "\n\n"

    stats_text += f"总轨迹数: {len(trajectories)}\n\n"

    stats_text += f"车道数量: {len(results['lanes'])}\n"
    for lane in results['lanes']:
        vehicles_in_lane = sum(1 for lid in track_to_lane.values() if lid == lane.lane_id)
        stats_text += f"  车道 {lane.lane_id}: {vehicles_in_lane} 辆\n"

    stats_text += f"\n车型分布:\n"
    type_counts = {}
    for vtype in vehicle_types.values():
        type_counts[vtype] = type_counts.get(vtype, 0) + 1

    for vtype, count in sorted(type_counts.items()):
        stats_text += f"  {vtype.capitalize()}: {count} 辆\n"

    stats_text += f"\n轨迹长度:\n"
    traj_lengths = [len(traj) for traj in trajectories.values()]
    stats_text += f"  平均: {np.mean(traj_lengths):.1f} 帧\n"
    stats_text += f"  最小: {np.min(traj_lengths)} 帧\n"
    stats_text += f"  最大: {np.max(traj_lengths)} 帧\n"

    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig('d:/研究/paper6/My_paper6/results/complete_pipeline_results.png', dpi=150)
    print("完整结果已保存到 results/complete_pipeline_results.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("路侧激光雷达车辆轨迹采集 - 三大创新点集成演示")
    print("=" * 70)

    # 1. 生成演示数据
    pcd_sequence = generate_demo_scene(num_frames=30)

    # 2. 创建处理流程
    pipeline = VehicleTrajectoryPipeline()

    # 3. 执行完整流程
    results = pipeline.process(pcd_sequence)

    # 4. 可视化结果
    visualize_results(results)

    # 5. 输出详细信息
    print("\n" + "=" * 70)
    print("详细轨迹信息:")
    print("=" * 70)

    for track in results['tracks']:
        if len(track.observations) < 2:
            continue

        track_id = track.track_id
        lane_id = results['track_to_lane'].get(track_id, -1)
        vehicle_type = results['vehicle_types'].get(track_id, 'unknown')

        # 计算统计量
        traj = track.trajectory
        velocities = np.diff(traj, axis=0, prepend=traj[0:1]) / 0.1
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = speeds.mean()

        distance = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))

        print(f"\n车辆 {track_id}:")
        print(f"  车道: {lane_id}")
        print(f"  车型: {vehicle_type}")
        print(f"  观测帧数: {len(track.observations)}")
        print(f"  轨迹长度: {distance:.2f} m")
        print(f"  平均速度: {avg_speed:.2f} m/s ({avg_speed * 3.6:.2f} km/h)")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
