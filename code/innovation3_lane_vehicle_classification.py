"""
创新点3: 基于点云空间分布的车道划分与SVM车型分类
Innovation 3: Lane Division and Vehicle Classification Based on Point Cloud Spatial Distribution

核心贡献:
1. 利用车辆点云空间分布进行车道划分（无需车道线标注）
2. 基于支持向量机（SVM）的车型分类
3. 提取多维度特征（长度、宽度、高度、点云密度等）
4. 生成高精度、多维度的车辆轨迹数据

方法流程:
输入: 车辆轨迹数据（包含位置、点云）
输出: 车道信息、车型分类、完整轨迹数据
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple
from dataclasses import dataclass
import joblib


@dataclass
class VehicleFeatures:
    """车辆特征"""
    track_id: int
    length: float
    width: float
    height: float
    volume: float
    point_density: float
    aspect_ratio: float  # 长宽比
    centroid_height: float
    num_points: int


@dataclass
class LaneInfo:
    """车道信息"""
    lane_id: int
    center_line: np.ndarray  # N x 2 (x, y)
    width: float
    direction: np.ndarray  # 主方向


@dataclass
class EnrichedTrajectory:
    """增强轨迹数据"""
    track_id: int
    trajectory: np.ndarray  # N x 3 (x, y, z)
    timestamps: np.ndarray
    velocities: np.ndarray
    lane_id: int
    vehicle_type: str  # 'car', 'truck', 'bus'
    features: VehicleFeatures


class VehicleFeatureExtractor:
    """车辆特征提取器"""

    @staticmethod
    def extract_features(points: np.ndarray, bbox: np.ndarray = None) -> Dict:
        """
        提取车辆特征

        Args:
            points: N x 3 点云
            bbox: [center_x, center_y, center_z, length, width, height, yaw]

        Returns:
            特征字典
        """
        features = {}

        # 基本几何特征
        if bbox is not None and len(bbox) >= 6:
            length, width, height = bbox[3:6]
        else:
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)
            extents = max_pt - min_pt
            length = extents[0]
            width = extents[1]
            height = extents[2]

        features['length'] = length
        features['width'] = width
        features['height'] = height
        features['volume'] = length * width * height

        # 点云统计特征
        features['num_points'] = len(points)
        features['point_density'] = len(points) / (features['volume'] + 1e-6)

        # 形状特征
        features['aspect_ratio'] = length / (width + 1e-6)
        features['centroid_height'] = points[:, 2].mean()

        # 点云分布特征
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        features['mean_distance_to_centroid'] = distances.mean()
        features['std_distance_to_centroid'] = distances.std()

        # 高度分布
        features['height_std'] = points[:, 2].std()
        features['height_range'] = points[:, 2].max() - points[:, 2].min()

        return features

    @staticmethod
    def features_to_vector(features: Dict) -> np.ndarray:
        """特征字典转向量"""
        feature_keys = [
            'length', 'width', 'height', 'volume',
            'aspect_ratio', 'point_density', 'centroid_height',
            'mean_distance_to_centroid', 'std_distance_to_centroid',
            'height_std', 'height_range'
        ]
        return np.array([features.get(k, 0) for k in feature_keys])


class SpatialLaneDivision:
    """基于空间分布的车道划分"""

    def __init__(self, num_lanes: int = None, cluster_eps: float = 2.0):
        """
        Args:
            num_lanes: 车道数量（None表示自动检测）
            cluster_eps: DBSCAN聚类半径
        """
        self.num_lanes = num_lanes
        self.cluster_eps = cluster_eps
        self.lanes: List[LaneInfo] = []

    def divide_lanes(self, trajectories: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        根据轨迹空间分布划分车道

        Args:
            trajectories: {track_id: trajectory (N x 3)}

        Returns:
            {track_id: lane_id}
        """
        if len(trajectories) == 0:
            return {}

        # 1. 提取所有轨迹的横向位置（y坐标）
        y_positions = []
        track_ids = []

        for track_id, traj in trajectories.items():
            # 使用轨迹中段的平均y坐标
            mid_start = len(traj) // 3
            mid_end = 2 * len(traj) // 3
            if mid_end > mid_start:
                mean_y = traj[mid_start:mid_end, 1].mean()
            else:
                mean_y = traj[:, 1].mean()

            y_positions.append([mean_y])
            track_ids.append(track_id)

        y_positions = np.array(y_positions)

        # 2. 聚类划分车道
        if self.num_lanes is not None:
            # K-means聚类
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_lanes, random_state=42)
            lane_labels = kmeans.fit_predict(y_positions)
            cluster_centers = kmeans.cluster_centers_
        else:
            # DBSCAN自动检测车道数
            dbscan = DBSCAN(eps=self.cluster_eps, min_samples=1)
            lane_labels = dbscan.fit_predict(y_positions)

            # 计算聚类中心
            unique_labels = np.unique(lane_labels)
            cluster_centers = []
            for label in unique_labels:
                if label == -1:
                    continue
                mask = lane_labels == label
                cluster_centers.append(y_positions[mask].mean(axis=0))
            cluster_centers = np.array(cluster_centers)

        # 3. 按y坐标排序车道（从小到大）
        sorted_indices = np.argsort(cluster_centers[:, 0])
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
        lane_labels = np.array([label_mapping.get(label, label) for label in lane_labels])

        # 4. 生成车道信息
        self.lanes = []
        unique_labels = np.unique(lane_labels)

        for lane_id in sorted(unique_labels):
            if lane_id == -1:
                continue

            # 获取该车道的所有轨迹
            lane_tracks = [trajectories[track_ids[i]]
                          for i, label in enumerate(lane_labels)
                          if label == lane_id]

            if len(lane_tracks) == 0:
                continue

            # 计算车道中心线
            all_points = np.vstack(lane_tracks)
            x_sorted = all_points[all_points[:, 0].argsort()]

            # 分段计算中心线
            num_segments = min(20, len(x_sorted) // 10)
            center_line = []

            for i in range(num_segments):
                start_idx = i * len(x_sorted) // num_segments
                end_idx = (i + 1) * len(x_sorted) // num_segments
                segment = x_sorted[start_idx:end_idx]

                if len(segment) > 0:
                    center_x = segment[:, 0].mean()
                    center_y = segment[:, 1].mean()
                    center_line.append([center_x, center_y])

            center_line = np.array(center_line)

            # 计算车道方向
            if len(center_line) > 1:
                direction = center_line[-1] - center_line[0]
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            else:
                direction = np.array([1, 0])

            # 估计车道宽度
            y_values = [traj[:, 1] for traj in lane_tracks]
            y_values = np.concatenate(y_values)
            lane_width = y_values.std() * 2  # 2个标准差

            lane_info = LaneInfo(
                lane_id=int(lane_id),
                center_line=center_line,
                width=lane_width,
                direction=direction
            )
            self.lanes.append(lane_info)

        # 5. 构建track_id到lane_id的映射
        track_to_lane = {track_ids[i]: int(lane_labels[i])
                        for i in range(len(track_ids))}

        return track_to_lane


class SVMVehicleClassifier:
    """基于SVM的车型分类器"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.classes = ['car', 'truck', 'bus']

    def train(self, features: List[np.ndarray], labels: List[str]):
        """
        训练分类器

        Args:
            features: 特征向量列表
            labels: 车型标签列表
        """
        X = np.array(features)
        y = np.array(labels)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练SVM
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        self.model.fit(X_scaled, y)

        print(f"训练完成! 训练样本数: {len(X)}")

    def predict(self, features: np.ndarray) -> str:
        """
        预测车型

        Args:
            features: 特征向量

        Returns:
            车型类别
        """
        if self.model is None:
            raise ValueError("模型未训练!")

        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)

        return prediction[0]

    def predict_batch(self, features_list: List[np.ndarray]) -> List[str]:
        """批量预测"""
        if self.model is None:
            raise ValueError("模型未训练!")

        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions.tolist()

    def save(self, path: str):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes
        }, path)

    def load(self, path: str):
        """加载模型"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.classes = data['classes']


def generate_synthetic_training_data() -> Tuple[List[np.ndarray], List[str]]:
    """生成合成训练数据"""
    features_list = []
    labels_list = []

    # 小汽车特征
    for _ in range(50):
        features = {
            'length': np.random.uniform(4.0, 5.5),
            'width': np.random.uniform(1.6, 2.0),
            'height': np.random.uniform(1.3, 1.7),
            'volume': 0,
            'aspect_ratio': 0,
            'point_density': np.random.uniform(50, 100),
            'centroid_height': np.random.uniform(0.6, 0.9),
            'mean_distance_to_centroid': np.random.uniform(1.0, 1.5),
            'std_distance_to_centroid': np.random.uniform(0.3, 0.6),
            'height_std': np.random.uniform(0.2, 0.4),
            'height_range': np.random.uniform(1.2, 1.6)
        }
        features['volume'] = features['length'] * features['width'] * features['height']
        features['aspect_ratio'] = features['length'] / features['width']

        feature_vec = VehicleFeatureExtractor.features_to_vector(features)
        features_list.append(feature_vec)
        labels_list.append('car')

    # 卡车特征
    for _ in range(30):
        features = {
            'length': np.random.uniform(7.0, 12.0),
            'width': np.random.uniform(2.2, 2.6),
            'height': np.random.uniform(2.5, 3.5),
            'volume': 0,
            'aspect_ratio': 0,
            'point_density': np.random.uniform(40, 80),
            'centroid_height': np.random.uniform(1.2, 1.8),
            'mean_distance_to_centroid': np.random.uniform(2.0, 3.5),
            'std_distance_to_centroid': np.random.uniform(0.6, 1.2),
            'height_std': np.random.uniform(0.4, 0.8),
            'height_range': np.random.uniform(2.3, 3.3)
        }
        features['volume'] = features['length'] * features['width'] * features['height']
        features['aspect_ratio'] = features['length'] / features['width']

        feature_vec = VehicleFeatureExtractor.features_to_vector(features)
        features_list.append(feature_vec)
        labels_list.append('truck')

    # 公交车特征
    for _ in range(20):
        features = {
            'length': np.random.uniform(10.0, 13.0),
            'width': np.random.uniform(2.4, 2.8),
            'height': np.random.uniform(2.8, 3.5),
            'volume': 0,
            'aspect_ratio': 0,
            'point_density': np.random.uniform(60, 100),
            'centroid_height': np.random.uniform(1.4, 2.0),
            'mean_distance_to_centroid': np.random.uniform(2.5, 4.0),
            'std_distance_to_centroid': np.random.uniform(0.8, 1.4),
            'height_std': np.random.uniform(0.5, 0.9),
            'height_range': np.random.uniform(2.6, 3.4)
        }
        features['volume'] = features['length'] * features['width'] * features['height']
        features['aspect_ratio'] = features['length'] / features['width']

        feature_vec = VehicleFeatureExtractor.features_to_vector(features)
        features_list.append(feature_vec)
        labels_list.append('bus')

    return features_list, labels_list


def demo_innovation3():
    """演示创新点3"""
    print("=" * 60)
    print("创新点3: 基于点云空间分布的车道划分与车型分类")
    print("=" * 60)

    # 1. 生成模拟轨迹数据
    print("\n生成模拟轨迹...")

    trajectories = {}
    vehicle_types_true = {}
    vehicle_points = {}

    # 3条车道，每道若干车辆
    lane_y_positions = [-4, 0, 4]  # 三条车道的y坐标
    vehicle_configs = [
        # (lane, type, length, width, height)
        (0, 'car', 4.5, 1.8, 1.5),
        (0, 'car', 4.2, 1.7, 1.4),
        (1, 'truck', 8.0, 2.4, 3.0),
        (1, 'car', 4.8, 1.9, 1.6),
        (2, 'bus', 11.0, 2.6, 3.2),
        (2, 'car', 4.3, 1.75, 1.45),
    ]

    for track_id, (lane_idx, vtype, length, width, height) in enumerate(vehicle_configs):
        lane_y = lane_y_positions[lane_idx]

        # 生成轨迹
        num_frames = 30
        traj = []
        for t in range(num_frames):
            x = -20 + t * 1.5
            y = lane_y + np.random.normal(0, 0.2)  # 添加横向抖动
            z = height / 2
            traj.append([x, y, z])

        trajectories[track_id] = np.array(traj)
        vehicle_types_true[track_id] = vtype

        # 生成点云
        center = traj[len(traj) // 2]  # 中间帧
        points = []
        for dx in np.linspace(-length/2, length/2, int(length * 10)):
            for dy in np.linspace(-width/2, width/2, int(width * 10)):
                for dz in np.linspace(0, height, int(height * 10)):
                    points.append(center + np.array([dx, dy, dz]))

        points = np.array(points)
        points += np.random.normal(0, 0.02, points.shape)
        vehicle_points[track_id] = points

    print(f"生成 {len(trajectories)} 条轨迹")

    # 2. 车道划分
    print("\n进行车道划分...")
    lane_divider = SpatialLaneDivision(num_lanes=3)
    track_to_lane = lane_divider.divide_lanes(trajectories)

    print(f"检测到 {len(lane_divider.lanes)} 条车道:")
    for lane in lane_divider.lanes:
        vehicles_in_lane = [tid for tid, lid in track_to_lane.items() if lid == lane.lane_id]
        print(f"  车道 {lane.lane_id}: {len(vehicles_in_lane)} 辆车，宽度 {lane.width:.2f}m")

    # 3. 车型分类
    print("\n训练车型分类器...")

    # 生成训练数据
    train_features, train_labels = generate_synthetic_training_data()

    # 训练分类器
    classifier = SVMVehicleClassifier()
    classifier.train(train_features, train_labels)

    # 提取测试特征并分类
    print("\n对测试车辆进行分类...")
    predictions = {}

    for track_id, points in vehicle_points.items():
        # 提取特征
        features_dict = VehicleFeatureExtractor.extract_features(points)
        feature_vec = VehicleFeatureExtractor.features_to_vector(features_dict)

        # 预测
        pred_type = classifier.predict(feature_vec)
        predictions[track_id] = pred_type

        true_type = vehicle_types_true[track_id]
        match = "✓" if pred_type == true_type else "✗"
        print(f"  车辆 {track_id}: 真实={true_type:5s}, 预测={pred_type:5s} {match}")

    # 4. 计算准确率
    correct = sum(1 for tid in predictions if predictions[tid] == vehicle_types_true[tid])
    accuracy = correct / len(predictions) * 100
    print(f"\n分类准确率: {accuracy:.1f}%")

    # 5. 生成增强轨迹数据
    enriched_trajectories = []

    for track_id, traj in trajectories.items():
        enriched = EnrichedTrajectory(
            track_id=track_id,
            trajectory=traj,
            timestamps=np.arange(len(traj)) * 0.1,
            velocities=np.diff(traj, axis=0, prepend=traj[0:1]) / 0.1,
            lane_id=track_to_lane.get(track_id, -1),
            vehicle_type=predictions[track_id],
            features=None
        )
        enriched_trajectories.append(enriched)

    # 6. 可视化
    print("\n可视化结果...")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: 车道划分
    ax1 = axes[0]
    colors_lane = ['red', 'green', 'blue', 'orange']

    for track_id, traj in trajectories.items():
        lane_id = track_to_lane.get(track_id, -1)
        color = colors_lane[lane_id % len(colors_lane)]
        ax1.plot(traj[:, 0], traj[:, 1], '-o', color=color,
                linewidth=2, markersize=3, label=f'车道 {lane_id}, 车辆 {track_id}')

    # 绘制车道中心线
    for lane in lane_divider.lanes:
        ax1.plot(lane.center_line[:, 0], lane.center_line[:, 1],
                '--', linewidth=2, color='black', alpha=0.5)

    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('车道划分结果', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.axis('equal')

    # 子图2: 车型分类
    ax2 = axes[1]
    type_colors = {'car': 'blue', 'truck': 'red', 'bus': 'green'}
    type_markers = {'car': 'o', 'truck': 's', 'bus': '^'}

    for track_id, traj in trajectories.items():
        vtype = predictions[track_id]
        color = type_colors.get(vtype, 'gray')
        marker = type_markers.get(vtype, 'x')

        ax2.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2, alpha=0.6)
        ax2.scatter(traj[0, 0], traj[0, 1], color=color, marker=marker,
                   s=150, edgecolors='black', linewidths=2,
                   label=f'{vtype} (ID {track_id})')

    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('车型分类结果', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('d:/研究/paper6/My_paper6/results/innovation3_lane_classification.png', dpi=150)
    print("结果已保存到 results/innovation3_lane_classification.png")
    plt.show()

    # 7. 输出统计信息
    print("\n" + "=" * 60)
    print("增强轨迹数据统计:")
    print("=" * 60)

    for enriched in enriched_trajectories:
        avg_speed = np.linalg.norm(enriched.velocities, axis=1).mean()
        print(f"车辆 {enriched.track_id}:")
        print(f"  - 车道: {enriched.lane_id}")
        print(f"  - 车型: {enriched.vehicle_type}")
        print(f"  - 轨迹长度: {len(enriched.trajectory)} 个点")
        print(f"  - 平均速度: {avg_speed:.2f} m/s")

    return enriched_trajectories, classifier


if __name__ == "__main__":
    trajectories, classifier = demo_innovation3()
    print("\n创新点3演示完成!")
