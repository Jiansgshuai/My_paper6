"""
创新点2: 基于时空质心与运动主方向的跨帧关联算法
Innovation 2: Cross-Frame Association Based on Spatio-Temporal Centroid and Motion Direction

核心贡献:
1. 利用时空质心位置分析解决车辆轨迹中断问题
2. 结合运动主方向进行跨帧关联
3. L-shape拟合算法生成优化的车辆3D边界框
4. 保持车辆形态位姿一致性

方法流程:
输入: 逐帧分割的车辆点云簇
输出: 连续的车辆轨迹（包含位置、朝向、速度）
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


@dataclass
class VehicleObservation:
    """车辆观测"""
    frame_id: int
    cluster_id: int
    points: np.ndarray  # N x 3
    centroid: np.ndarray  # 3D质心
    bbox: Optional[np.ndarray] = None  # 3D边界框 [center, extent, rotation]
    velocity: Optional[np.ndarray] = None  # 速度
    timestamp: float = 0.0


@dataclass
class VehicleTrack:
    """车辆轨迹"""
    track_id: int
    observations: List[VehicleObservation]
    active: bool = True

    @property
    def last_observation(self) -> VehicleObservation:
        return self.observations[-1]

    @property
    def trajectory(self) -> np.ndarray:
        """返回轨迹中心点序列"""
        return np.array([obs.centroid for obs in self.observations])

    def predict_next_position(self, dt: float = 0.1) -> np.ndarray:
        """预测下一位置"""
        if len(self.observations) < 2:
            return self.last_observation.centroid

        # 使用最近两帧估计速度
        last = self.observations[-1]
        prev = self.observations[-2]
        velocity = (last.centroid - prev.centroid) / dt

        return last.centroid + velocity * dt

    def estimate_motion_direction(self) -> np.ndarray:
        """估计运动主方向"""
        if len(self.observations) < 2:
            return np.array([1, 0, 0])

        trajectory = self.trajectory
        # 使用PCA提取主方向
        centered = trajectory - trajectory.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]

        # 确保方向一致
        if len(trajectory) >= 2:
            motion = trajectory[-1] - trajectory[-2]
            if np.dot(main_direction, motion) < 0:
                main_direction = -main_direction

        return main_direction


class LShapeFitting:
    """L-shape拟合算法"""

    @staticmethod
    def fit_lshape(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        L-shape拟合获取车辆边界框

        Args:
            points: N x 3点云

        Returns:
            center: 中心点
            extent: [length, width, height]
            yaw: 偏航角
        """
        if len(points) < 4:
            return LShapeFitting._fallback_bbox(points)

        # 只使用XY平面
        points_2d = points[:, :2]

        # 寻找最佳角度
        best_angle = LShapeFitting._search_angle(points_2d)

        # 旋转点云
        rotation_matrix = np.array([
            [np.cos(best_angle), -np.sin(best_angle)],
            [np.sin(best_angle), np.cos(best_angle)]
        ])
        rotated_points = points_2d @ rotation_matrix.T

        # 计算边界框
        min_vals = rotated_points.min(axis=0)
        max_vals = rotated_points.max(axis=0)
        center_2d = (min_vals + max_vals) / 2

        # 转回原坐标系
        center_2d = center_2d @ rotation_matrix
        center = np.array([center_2d[0], center_2d[1], points[:, 2].mean()])

        # 计算尺寸
        length = max_vals[0] - min_vals[0]
        width = max_vals[1] - min_vals[1]
        height = points[:, 2].max() - points[:, 2].min()

        extent = np.array([length, width, height])

        return center, extent, best_angle

    @staticmethod
    def _search_angle(points_2d: np.ndarray, num_angles: int = 180) -> float:
        """搜索最佳L-shape角度"""
        angles = np.linspace(0, np.pi, num_angles)
        best_angle = 0
        min_area = float('inf')

        for angle in angles:
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            rotated = points_2d @ rotation_matrix.T

            min_vals = rotated.min(axis=0)
            max_vals = rotated.max(axis=0)

            # L-shape特征：计算矩形面积
            length = max_vals[0] - min_vals[0]
            width = max_vals[1] - min_vals[1]
            area = length * width

            if area < min_area:
                min_area = area
                best_angle = angle

        return best_angle

    @staticmethod
    def _fallback_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """后备边界框计算"""
        center = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        return center, extent, 0.0


class TemporalAssociator:
    """基于时空质心与运动方向的跨帧关联器"""

    def __init__(self,
                 max_distance: float = 5.0,
                 max_missing_frames: int = 5,
                 angle_weight: float = 0.3,
                 dt: float = 0.1):
        """
        Args:
            max_distance: 最大关联距离
            max_missing_frames: 最大允许丢失帧数
            angle_weight: 角度权重
            dt: 时间间隔
        """
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.angle_weight = angle_weight
        self.dt = dt

        self.tracks: List[VehicleTrack] = []
        self.next_track_id = 0
        self.current_frame = 0

    def process_frame(self, observations: List[VehicleObservation]) -> List[VehicleTrack]:
        """
        处理一帧观测

        Args:
            observations: 当前帧的车辆观测列表

        Returns:
            更新后的轨迹列表
        """
        # 1. 为每个观测计算L-shape边界框
        for obs in observations:
            center, extent, yaw = LShapeFitting.fit_lshape(obs.points)
            obs.bbox = np.concatenate([center, extent, [yaw]])
            obs.frame_id = self.current_frame

        # 2. 数据关联
        if len(self.tracks) == 0:
            # 初始化轨迹
            for obs in observations:
                self._create_track(obs)
        else:
            self._associate_observations(observations)

        # 3. 更新轨迹状态
        self._update_tracks()

        self.current_frame += 1

        return [t for t in self.tracks if t.active]

    def _create_track(self, obs: VehicleObservation):
        """创建新轨迹"""
        track = VehicleTrack(
            track_id=self.next_track_id,
            observations=[obs]
        )
        self.tracks.append(track)
        self.next_track_id += 1

    def _associate_observations(self, observations: List[VehicleObservation]):
        """关联观测到轨迹"""
        if len(observations) == 0:
            return

        # 计算代价矩阵
        active_tracks = [t for t in self.tracks if t.active]
        cost_matrix = np.zeros((len(active_tracks), len(observations)))

        for i, track in enumerate(active_tracks):
            predicted_pos = track.predict_next_position(self.dt)
            motion_dir = track.estimate_motion_direction()

            for j, obs in enumerate(observations):
                # 位置距离
                pos_dist = np.linalg.norm(obs.centroid - predicted_pos)

                # 运动方向一致性
                if len(track.observations) >= 2:
                    obs_dir = obs.centroid - track.last_observation.centroid
                    obs_dir_norm = np.linalg.norm(obs_dir)

                    if obs_dir_norm > 0.01:
                        obs_dir = obs_dir / obs_dir_norm
                        angle_diff = np.arccos(np.clip(
                            np.dot(motion_dir[:2], obs_dir[:2]), -1, 1
                        ))
                        angle_cost = angle_diff / np.pi
                    else:
                        angle_cost = 0
                else:
                    angle_cost = 0

                # 综合代价
                cost = pos_dist + self.angle_weight * angle_cost * self.max_distance
                cost_matrix[i, j] = cost

        # 匈牙利算法匹配（简化版：贪心）
        matched_tracks = set()
        matched_obs = set()

        while True:
            if len(matched_tracks) == len(active_tracks) or \
               len(matched_obs) == len(observations):
                break

            # 找最小代价
            min_cost = float('inf')
            min_i, min_j = -1, -1

            for i in range(len(active_tracks)):
                if i in matched_tracks:
                    continue
                for j in range(len(observations)):
                    if j in matched_obs:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j

            if min_cost > self.max_distance:
                break

            # 匹配
            matched_tracks.add(min_i)
            matched_obs.add(min_j)
            active_tracks[min_i].observations.append(observations[min_j])

        # 创建新轨迹
        for j in range(len(observations)):
            if j not in matched_obs:
                self._create_track(observations[j])

    def _update_tracks(self):
        """更新轨迹状态"""
        for track in self.tracks:
            if not track.active:
                continue

            # 检查是否丢失
            frames_since_update = self.current_frame - track.last_observation.frame_id
            if frames_since_update > self.max_missing_frames:
                track.active = False

            # 更新速度
            if len(track.observations) >= 2:
                last = track.observations[-1]
                prev = track.observations[-2]
                last.velocity = (last.centroid - prev.centroid) / self.dt

    def get_trajectories(self) -> Dict[int, np.ndarray]:
        """获取所有轨迹"""
        trajectories = {}
        for track in self.tracks:
            if len(track.observations) >= 2:
                trajectories[track.track_id] = track.trajectory
        return trajectories


def demo_innovation2():
    """演示创新点2"""
    print("=" * 60)
    print("创新点2: 基于时空质心与运动主方向的跨帧关联")
    print("=" * 60)

    # 1. 生成模拟车辆轨迹数据
    print("\n生成模拟车辆观测...")

    num_frames = 20
    num_vehicles = 3

    # 生成车辆真实轨迹
    true_trajectories = []
    for v in range(num_vehicles):
        start_x = -15 + v * 10
        start_y = -2 + v * 2
        trajectory = []
        for t in range(num_frames):
            x = start_x + t * 1.5
            y = start_y + 0.3 * np.sin(t * 0.3)
            z = 0.8
            trajectory.append([x, y, z])
        true_trajectories.append(np.array(trajectory))

    # 生成观测点云（添加噪声和间断）
    observations_per_frame = []

    for t in range(num_frames):
        frame_obs = []

        for v in range(num_vehicles):
            # 模拟遮挡（随机丢失）
            if np.random.rand() > 0.2:  # 80%检测率
                center = true_trajectories[v][t]

                # 生成车辆点云（简化长方体）
                points = []
                for dx in np.linspace(-2, 2, 20):
                    for dy in np.linspace(-1, 1, 10):
                        for dz in np.linspace(0, 1.5, 8):
                            points.append(center + np.array([dx, dy, dz]))

                points = np.array(points)
                # 添加噪声
                points += np.random.normal(0, 0.05, points.shape)

                obs = VehicleObservation(
                    frame_id=t,
                    cluster_id=v,
                    points=points,
                    centroid=center,
                    timestamp=t * 0.1
                )
                frame_obs.append(obs)

        observations_per_frame.append(frame_obs)

    print(f"生成 {num_frames} 帧，{num_vehicles} 辆车的观测")

    # 2. 创建关联器
    associator = TemporalAssociator(
        max_distance=3.0,
        max_missing_frames=3,
        angle_weight=0.3,
        dt=0.1
    )

    # 3. 逐帧处理
    print("\n开始跨帧关联...")
    import time
    start_time = time.time()

    for t, frame_obs in enumerate(observations_per_frame):
        print(f"处理帧 {t+1}/{num_frames}，检测到 {len(frame_obs)} 个目标", end='\r')
        associator.process_frame(frame_obs)

    end_time = time.time()
    print(f"\n关联完成! 耗时: {end_time - start_time:.2f}秒")

    # 4. 结果分析
    trajectories = associator.get_trajectories()
    print(f"\n生成 {len(trajectories)} 条轨迹:")

    for track_id, traj in trajectories.items():
        print(f"  轨迹 {track_id}: {len(traj)} 个观测点")

    # 5. 可视化
    print("\n可视化轨迹...")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 5))

    # 子图1: XY平面
    ax1 = fig.add_subplot(131)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # 真实轨迹
    for v, traj in enumerate(true_trajectories):
        ax1.plot(traj[:, 0], traj[:, 1], '--', color=colors[v % len(colors)],
                alpha=0.5, label=f'真实轨迹 {v+1}')

    # 估计轨迹
    for track_id, traj in trajectories.items():
        color = colors[track_id % len(colors)]
        ax1.plot(traj[:, 0], traj[:, 1], '-o', color=color,
                linewidth=2, markersize=4, label=f'估计轨迹 {track_id}')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('车辆轨迹 (俯视图)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # 子图2: 3D轨迹
    ax2 = fig.add_subplot(132, projection='3d')
    for track_id, traj in trajectories.items():
        color = colors[track_id % len(colors)]
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-o', color=color,
                linewidth=2, markersize=3, label=f'轨迹 {track_id}')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('车辆轨迹 (3D视图)')
    ax2.legend()

    # 子图3: 速度曲线
    ax3 = fig.add_subplot(133)
    for track in associator.tracks:
        if len(track.observations) < 2:
            continue

        velocities = []
        times = []
        for obs in track.observations:
            if obs.velocity is not None:
                velocities.append(np.linalg.norm(obs.velocity))
                times.append(obs.timestamp)

        if velocities:
            ax3.plot(times, velocities, '-o', label=f'轨迹 {track.track_id}')

    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('速度 (m/s)')
    ax3.set_title('车辆速度')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('d:/研究/paper6/My_paper6/results/innovation2_trajectories.png', dpi=150)
    print("结果已保存到 results/innovation2_trajectories.png")
    plt.show()

    return trajectories


if __name__ == "__main__":
    trajectories = demo_innovation2()
    print("\n创新点2演示完成!")
