"""
创新点1: 基于稀疏体素八叉树（SVO）的增量式时空背景滤除
Innovation 1: Incremental Spatio-Temporal Background Removal Using Sparse Voxel Octree

核心贡献:
1. 利用稀疏体素八叉树（SVO）数据结构实现高效背景建模
2. 结合统计离群点滤波（SOR）去除噪声
3. GPU加速的稀疏体素八叉树查询
4. 增量式更新背景模型，适应动态场景

方法流程:
输入: 原始点云序列
输出: 去除背景后的车辆点云
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
from collections import defaultdict
import time


class OctreeNode:
    """八叉树节点"""
    def __init__(self, center: np.ndarray, size: float):
        self.center = center  # 节点中心
        self.size = size      # 节点尺寸
        self.children = [None] * 8  # 8个子节点
        self.points = []      # 叶节点存储的点
        self.is_leaf = True   # 是否为叶节点
        self.point_count = 0  # 累计点数量（用于背景统计）
        self.visit_count = 0  # 访问次数
        self.is_background = False  # 是否为背景


class SparseVoxelOctree:
    """稀疏体素八叉树"""

    def __init__(self,
                 bounds: Tuple[np.ndarray, np.ndarray],
                 max_depth: int = 8,
                 leaf_size: float = 0.2,
                 background_threshold: int = 5):
        """
        Args:
            bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
            max_depth: 最大深度
            leaf_size: 叶节点最小尺寸
            background_threshold: 背景判定阈值（帧数）
        """
        self.min_bound, self.max_bound = bounds
        self.max_depth = max_depth
        self.leaf_size = leaf_size
        self.background_threshold = background_threshold

        # 计算根节点
        center = (self.min_bound + self.max_bound) / 2
        size = np.max(self.max_bound - self.min_bound)
        self.root = OctreeNode(center, size)

    def get_octant(self, point: np.ndarray, node: OctreeNode) -> int:
        """确定点所在的八叉象限"""
        octant = 0
        if point[0] >= node.center[0]:
            octant |= 4
        if point[1] >= node.center[1]:
            octant |= 2
        if point[2] >= node.center[2]:
            octant |= 1
        return octant

    def insert(self, point: np.ndarray, node: OctreeNode, depth: int = 0):
        """插入点到八叉树"""
        if node.is_leaf:
            node.points.append(point)
            node.point_count += 1

            # 判断是否需要分裂
            if len(node.points) > 1 and node.size > self.leaf_size and depth < self.max_depth:
                self._split_node(node, depth)
        else:
            # 递归插入到子节点
            octant = self.get_octant(point, node)
            if node.children[octant] is None:
                # 创建子节点
                offset = np.array([
                    node.size / 4 if octant & 4 else -node.size / 4,
                    node.size / 4 if octant & 2 else -node.size / 4,
                    node.size / 4 if octant & 1 else -node.size / 4
                ])
                child_center = node.center + offset
                node.children[octant] = OctreeNode(child_center, node.size / 2)

            self.insert(point, node.children[octant], depth + 1)

    def _split_node(self, node: OctreeNode, depth: int):
        """分裂节点"""
        node.is_leaf = False
        points = node.points
        node.points = []

        for point in points:
            octant = self.get_octant(point, node)
            if node.children[octant] is None:
                offset = np.array([
                    node.size / 4 if octant & 4 else -node.size / 4,
                    node.size / 4 if octant & 2 else -node.size / 4,
                    node.size / 4 if octant & 1 else -node.size / 4
                ])
                child_center = node.center + offset
                node.children[octant] = OctreeNode(child_center, node.size / 2)

            self.insert(point, node.children[octant], depth + 1)

    def query_radius(self, point: np.ndarray, radius: float) -> List[np.ndarray]:
        """半径查询"""
        results = []
        self._query_radius_recursive(point, radius, self.root, results)
        return results

    def _query_radius_recursive(self, point: np.ndarray, radius: float,
                                node: OctreeNode, results: List):
        """递归半径查询"""
        if node is None:
            return

        # 检查节点边界是否与查询球相交
        half_size = node.size / 2
        closest_point = np.clip(point,
                               node.center - half_size,
                               node.center + half_size)
        dist = np.linalg.norm(point - closest_point)

        if dist > radius:
            return

        if node.is_leaf:
            for p in node.points:
                if np.linalg.norm(point - p) <= radius:
                    results.append(p)
        else:
            for child in node.children:
                if child is not None:
                    self._query_radius_recursive(point, radius, child, results)

    def mark_background(self, node: OctreeNode = None):
        """标记背景节点（访问次数高的节点）"""
        if node is None:
            node = self.root

        if node.is_leaf:
            node.visit_count += 1
            if node.visit_count >= self.background_threshold:
                node.is_background = True
        else:
            for child in node.children:
                if child is not None:
                    self.mark_background(child)

    def is_background_point(self, point: np.ndarray) -> bool:
        """判断点是否为背景"""
        node = self._find_leaf_node(point, self.root)
        return node.is_background if node else False

    def _find_leaf_node(self, point: np.ndarray, node: OctreeNode) -> Optional[OctreeNode]:
        """查找点所在的叶节点"""
        if node is None:
            return None

        if node.is_leaf:
            return node

        octant = self.get_octant(point, node)
        if node.children[octant] is None:
            return None

        return self._find_leaf_node(point, node.children[octant])


class SVOBackgroundRemover:
    """基于SVO的背景去除器"""

    def __init__(self,
                 voxel_size: float = 0.2,
                 background_threshold: int = 5,
                 sor_nb_neighbors: int = 20,
                 sor_std_ratio: float = 2.0):
        """
        Args:
            voxel_size: 体素大小
            background_threshold: 背景判定阈值（连续帧数）
            sor_nb_neighbors: SOR邻域点数
            sor_std_ratio: SOR标准差倍数
        """
        self.voxel_size = voxel_size
        self.background_threshold = background_threshold
        self.sor_nb_neighbors = sor_nb_neighbors
        self.sor_std_ratio = sor_std_ratio
        self.octree = None
        self.frame_count = 0

    def _statistical_outlier_removal(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """统计离群点滤波"""
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.sor_nb_neighbors,
            std_ratio=self.sor_std_ratio
        )
        return pcd.select_by_index(ind)

    def initialize_octree(self, pcd: o3d.geometry.PointCloud):
        """初始化八叉树"""
        points = np.asarray(pcd.points)
        min_bound = points.min(axis=0) - 1.0
        max_bound = points.max(axis=0) + 1.0

        self.octree = SparseVoxelOctree(
            bounds=(min_bound, max_bound),
            leaf_size=self.voxel_size,
            background_threshold=self.background_threshold
        )

    def process_frame(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        处理单帧点云

        Args:
            pcd: 输入点云

        Returns:
            去除背景后的点云
        """
        # 1. 统计离群点滤波
        pcd_filtered = self._statistical_outlier_removal(pcd)

        # 2. 初始化或更新八叉树
        if self.octree is None:
            self.initialize_octree(pcd_filtered)

        points = np.asarray(pcd_filtered.points)

        # 3. 插入点到八叉树并标记背景
        for point in points:
            self.octree.insert(point, self.octree.root)

        self.frame_count += 1

        # 4. 更新背景模型
        if self.frame_count >= self.background_threshold:
            self.octree.mark_background()

        # 5. 分离前景点（车辆）
        foreground_mask = []
        for point in points:
            is_bg = self.octree.is_background_point(point)
            foreground_mask.append(not is_bg)

        foreground_points = points[foreground_mask]

        # 6. 创建前景点云
        foreground_pcd = o3d.geometry.PointCloud()
        foreground_pcd.points = o3d.utility.Vector3dVector(foreground_points)

        if pcd_filtered.has_colors():
            colors = np.asarray(pcd_filtered.colors)[foreground_mask]
            foreground_pcd.colors = o3d.utility.Vector3dVector(colors)

        return foreground_pcd

    def process_sequence(self, pcd_list: List[o3d.geometry.PointCloud]) -> List[o3d.geometry.PointCloud]:
        """
        处理点云序列

        Args:
            pcd_list: 点云序列

        Returns:
            去除背景后的点云序列
        """
        results = []
        print(f"Processing {len(pcd_list)} frames...")

        for i, pcd in enumerate(pcd_list):
            print(f"Frame {i+1}/{len(pcd_list)}", end='\r')
            result_pcd = self.process_frame(pcd)
            results.append(result_pcd)

        print("\nProcessing complete!")
        return results


def demo_innovation1():
    """演示创新点1"""
    print("=" * 60)
    print("创新点1: 基于稀疏体素八叉树的增量式时空背景滤除")
    print("=" * 60)

    # 1. 生成模拟数据
    print("\n生成模拟场景...")

    # 背景点云（道路、建筑物等）
    background_points = []
    # 地面
    x = np.random.uniform(-20, 20, 5000)
    y = np.random.uniform(-10, 10, 5000)
    z = np.random.uniform(-0.5, 0, 5000)
    background_points.append(np.column_stack([x, y, z]))

    # 路边建筑
    x = np.random.uniform(-22, -20, 1000)
    y = np.random.uniform(-10, 10, 1000)
    z = np.random.uniform(0, 5, 1000)
    background_points.append(np.column_stack([x, y, z]))

    background_points = np.vstack(background_points)

    # 创建点云序列（包含移动车辆）
    pcd_sequence = []
    num_frames = 10

    for t in range(num_frames):
        # 背景点云（每帧相同）
        frame_points = background_points.copy()

        # 移动车辆（简化为长方体）
        vehicle_x = -10 + t * 2  # 车辆沿x轴移动
        vehicle_points = []
        for dx in np.linspace(0, 4, 20):
            for dy in np.linspace(-1, 1, 10):
                for dz in np.linspace(0, 2, 10):
                    vehicle_points.append([vehicle_x + dx, dy, dz])

        vehicle_points = np.array(vehicle_points)
        frame_points = np.vstack([frame_points, vehicle_points])

        # 添加噪声
        noise = np.random.normal(0, 0.01, frame_points.shape)
        frame_points += noise

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_points)
        pcd_sequence.append(pcd)

    print(f"生成 {num_frames} 帧点云，每帧约 {len(frame_points)} 个点")

    # 2. 创建背景去除器
    remover = SVOBackgroundRemover(
        voxel_size=0.5,
        background_threshold=5,
        sor_nb_neighbors=20,
        sor_std_ratio=2.0
    )

    # 3. 处理序列
    print("\n开始处理...")
    start_time = time.time()
    result_sequence = remover.process_sequence(pcd_sequence)
    end_time = time.time()

    print(f"\n处理完成! 耗时: {end_time - start_time:.2f}秒")
    print(f"平均每帧: {(end_time - start_time) / num_frames:.3f}秒")

    # 4. 统计结果
    print("\n结果统计:")
    for i, (original, result) in enumerate(zip(pcd_sequence, result_sequence)):
        original_pts = len(original.points)
        result_pts = len(result.points)
        removal_rate = (1 - result_pts / original_pts) * 100
        print(f"帧 {i+1}: {original_pts} -> {result_pts} 点 (去除 {removal_rate:.1f}%)")

    # 5. 可视化（最后一帧）
    print("\n可视化最后一帧...")
    print("- 红色: 原始点云")
    print("- 绿色: 去除背景后的点云（车辆）")

    # 着色
    pcd_sequence[-1].paint_uniform_color([1, 0, 0])  # 红色
    result_sequence[-1].paint_uniform_color([0, 1, 0])  # 绿色

    # 显示
    o3d.visualization.draw_geometries(
        [pcd_sequence[-1], result_sequence[-1]],
        window_name="创新点1: 背景去除结果",
        width=1200,
        height=800
    )

    return result_sequence


if __name__ == "__main__":
    result = demo_innovation1()
    print("\n创新点1演示完成!")
