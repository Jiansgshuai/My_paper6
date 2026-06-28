"""
工具函数模块
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple


def load_pcd(file_path: str) -> o3d.geometry.PointCloud:
    """加载PCD文件"""
    return o3d.read_point_cloud(file_path)


def save_pcd(pcd: o3d.geometry.PointCloud, file_path: str):
    """保存PCD文件"""
    o3d.io.write_point_cloud(file_path, pcd)


def downsample_pcd(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.1) -> o3d.geometry.PointCloud:
    """体素下采样"""
    return pcd.voxel_down_sample(voxel_size)


def remove_outliers(pcd: o3d.geometry.PointCloud,
                   nb_neighbors: int = 20,
                   std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """统计离群点去除"""
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return pcd.select_by_index(ind)


def compute_normals(pcd: o3d.geometry.PointCloud,
                   radius: float = 0.1,
                   max_nn: int = 30):
    """计算法向量"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )


def cluster_dbscan(pcd: o3d.geometry.PointCloud,
                  eps: float = 0.5,
                  min_points: int = 10) -> List[np.ndarray]:
    """DBSCAN聚类"""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    clusters = []
    max_label = labels.max()

    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_points = np.asarray(pcd.points)[cluster_indices]
        clusters.append(cluster_points)

    return clusters


def compute_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算轴对齐边界框"""
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    center = (min_bound + max_bound) / 2
    extent = max_bound - min_bound

    return center, extent


def transform_points(points: np.ndarray,
                    translation: np.ndarray = None,
                    rotation: np.ndarray = None) -> np.ndarray:
    """变换点云"""
    transformed = points.copy()

    if rotation is not None:
        transformed = transformed @ rotation.T

    if translation is not None:
        transformed += translation

    return transformed


def compute_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """计算两组点之间的距离矩阵"""
    from scipy.spatial.distance import cdist
    return cdist(points1, points2)


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    拟合平面

    Returns:
        normal: 平面法向量
        d: 平面方程参数 (ax + by + cz + d = 0)
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    # SVD分解
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]

    # 确保法向量朝上
    if normal[2] < 0:
        normal = -normal

    d = -np.dot(normal, centroid)

    return normal, d


def estimate_ground_plane(pcd: o3d.geometry.PointCloud,
                         distance_threshold: float = 0.1,
                         ransac_n: int = 3,
                         num_iterations: int = 1000) -> Tuple[np.ndarray, List[int]]:
    """
    使用RANSAC估计地面平面

    Returns:
        plane_model: [a, b, c, d] 平面方程参数
        inliers: 内点索引
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    return plane_model, inliers


def filter_by_height(pcd: o3d.geometry.PointCloud,
                    min_height: float = 0.0,
                    max_height: float = 3.0) -> o3d.geometry.PointCloud:
    """根据高度过滤点云"""
    points = np.asarray(pcd.points)
    mask = (points[:, 2] >= min_height) & (points[:, 2] <= max_height)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    return filtered_pcd


def create_bbox_lines(center: np.ndarray,
                     extent: np.ndarray,
                     rotation: float = 0.0) -> o3d.geometry.LineSet:
    """创建3D边界框的线集"""
    # 边界框的8个顶点
    half_extent = extent / 2
    corners = np.array([
        [-half_extent[0], -half_extent[1], -half_extent[2]],
        [half_extent[0], -half_extent[1], -half_extent[2]],
        [half_extent[0], half_extent[1], -half_extent[2]],
        [-half_extent[0], half_extent[1], -half_extent[2]],
        [-half_extent[0], -half_extent[1], half_extent[2]],
        [half_extent[0], -half_extent[1], half_extent[2]],
        [half_extent[0], half_extent[1], half_extent[2]],
        [-half_extent[0], half_extent[1], half_extent[2]],
    ])

    # 旋转
    if rotation != 0:
        rot_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation), 0],
            [np.sin(rotation), np.cos(rotation), 0],
            [0, 0, 1]
        ])
        corners = corners @ rot_matrix.T

    # 平移
    corners += center

    # 边界框的12条边
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 竖边
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

    return line_set
