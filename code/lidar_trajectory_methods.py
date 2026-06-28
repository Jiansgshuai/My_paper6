import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict


def distance_partition_normalize(points: np.ndarray, sensor_origin: np.ndarray, num_bins: int = 5, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize BEV point coordinates by distance bins to reduce scale differences."""
    assert points.shape[1] >= 2, "Points must include x,y coordinates"
    distances = np.linalg.norm(points[:, :2] - sensor_origin[:2], axis=1)
    if len(distances) == 0:
        return points.copy()
    bins = np.quantile(distances, np.linspace(0, 1, num_bins + 1))
    normalized = points.copy()
    for k in range(num_bins):
        mask = (distances >= bins[k]) & (distances <= bins[k + 1])
        if np.count_nonzero(mask) == 0:
            continue
        subset = points[mask, :2]
        mean = subset.mean(axis=0)
        std = subset.std(axis=0) + epsilon
        normalized[mask, 0] = (subset[:, 0] - mean[0]) / std[0]
        normalized[mask, 1] = (subset[:, 1] - mean[1]) / std[1]
    return normalized


def build_spatio_temporal_stack(frames: List[o3d.geometry.PointCloud], timestamps: List[float]) -> np.ndarray:
    """Build a spatio-temporal point stack from multi-frame foreground point clouds."""
    assert len(frames) == len(timestamps)
    stacked = []
    for frame, t in zip(frames, timestamps):
        points = np.asarray(frame.points)
        if points.shape[0] == 0:
            continue
        xyz = points[:, :3]
        t_column = np.full((xyz.shape[0], 1), t, dtype=np.float32)
        stacked.append(np.hstack([xyz, t_column]))
    if not stacked:
        return np.zeros((0, 4), dtype=np.float32)
    return np.vstack(stacked)


def cylindrical_dbscan(points: np.ndarray, times: np.ndarray, eps_space: float, eps_time: float, min_samples: int) -> np.ndarray:
    """Perform density clustering with a cylindrical neighborhood in space-time."""
    n = points.shape[0]
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def neighbors(idx: int) -> List[int]:
        delta_t = np.abs(times - times[idx])
        mask = delta_t <= eps_time
        spatial = np.linalg.norm(points[mask] - points[idx], axis=1)
        idxs = np.where(mask)[0]
        return idxs[spatial <= eps_space].tolist()

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = nbrs.copy()
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                j_nbrs = neighbors(j)
                if len(j_nbrs) >= min_samples:
                    queue.extend([x for x in j_nbrs if x not in queue])
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def incremental_spatio_temporal_clustering(frames: List[o3d.geometry.PointCloud], timestamps: List[float], eps_space: float = 1.0, eps_time: float = 0.5, min_samples: int = 5, window_frames: int = 3, overlap_frames: int = 1) -> Dict[int, np.ndarray]:
    """Run incremental clustering on a sequence of frames and maintain consistent labels across windows."""
    if len(frames) == 0:
        return {}
    assert len(frames) == len(timestamps), "Frame and timestamp counts must match"
    labels_by_frame: Dict[int, np.ndarray] = {}
    next_global_label = 0
    n_frames = len(frames)
    last_window_frame_indices: List[int] = []
    last_window_labels: List[np.ndarray] = []

    for start in range(0, n_frames, window_frames - overlap_frames):
        end = min(start + window_frames, n_frames)
        frame_indices = list(range(start, end))
        frame_point_arrays = [np.asarray(frames[i].points) for i in frame_indices]
        counts = [arr.shape[0] for arr in frame_point_arrays]
        nonempty = [arr for arr in frame_point_arrays if arr.shape[0] > 0]
        local_points = np.vstack([arr[:, :3] for arr in nonempty]) if nonempty else np.zeros((0, 3), dtype=np.float32)
        local_times = np.concatenate([np.full((arr.shape[0],), timestamps[i], dtype=np.float32) for i, arr in zip(frame_indices, frame_point_arrays) if arr.shape[0] > 0]) if nonempty else np.zeros((0,), dtype=np.float32)
        if local_points.size == 0:
            for i in frame_indices:
                labels_by_frame[i] = np.array([], dtype=int)
            last_window_frame_indices = frame_indices
            last_window_labels = [np.array([], dtype=int) for _ in frame_indices]
            continue
        local_labels = cylindrical_dbscan(local_points[:, :2], local_times, eps_space, eps_time, min_samples)
        label_segments = []
        idx = 0
        for cnt in counts:
            label_segments.append(local_labels[idx:idx + cnt] if cnt > 0 else np.array([], dtype=int))
            idx += cnt
        if start == 0:
            for frame_idx, seg in zip(frame_indices, label_segments):
                labels_by_frame[frame_idx] = seg
            next_global_label = np.max(local_labels[local_labels >= 0]) + 1 if np.any(local_labels >= 0) else 0
        else:
            mapping: Dict[int, int] = {}
            overlap_start = max(start, start - overlap_frames)
            overlap_indices = [fi for fi in frame_indices if fi < start + overlap_frames]
            for fi in overlap_indices:
                if fi in last_window_frame_indices:
                    old_seg = last_window_labels[last_window_frame_indices.index(fi)]
                    cur_seg = label_segments[frame_indices.index(fi)]
                    for old_lbl in np.unique(old_seg[old_seg >= 0]):
                        mask = old_seg == old_lbl
                        if np.count_nonzero(mask) == 0:
                            continue
                        candidates = cur_seg[mask]
                        candidates = candidates[candidates >= 0]
                        if candidates.size == 0:
                            continue
                        mapped = np.bincount(candidates).argmax()
                        if mapped >= 0:
                            mapping[mapped] = old_lbl
            for old_lbl in np.unique(np.concatenate(last_window_labels) if last_window_labels else np.array([], dtype=int)):
                if old_lbl < 0:
                    continue
                if old_lbl not in mapping.values():
                    pass
            for idx_lbl in range(local_labels.shape[0]):
                if local_labels[idx_lbl] >= 0 and local_labels[idx_lbl] in mapping:
                    local_labels[idx_lbl] = mapping[local_labels[idx_lbl]]
                elif local_labels[idx_lbl] >= 0:
                    local_labels[idx_lbl] = next_global_label
                    next_global_label += 1
            idx = 0
            for frame_idx, cnt in zip(frame_indices, counts):
                labels_by_frame[frame_idx] = local_labels[idx:idx + cnt] if cnt > 0 else np.array([], dtype=int)
                idx += cnt
        last_window_frame_indices = frame_indices
        last_window_labels = [labels_by_frame[i] for i in frame_indices]
    return labels_by_frame


def compute_centroid(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    return points.mean(axis=0)


def principal_direction(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cov = np.cov(points.T)
    w, v = np.linalg.eigh(cov)
    return v[:, np.argmax(w)]


typedef = Dict[str, np.ndarray]


def re_associate_fragments(disappeared: List[Dict], new_segments: List[Dict], angle_thresh_deg: float = 5.0) -> List[Tuple[int, int]]:
    """Greedily associate disappeared fragments with new segments based on motion direction."""
    matches = []
    if not disappeared or not new_segments:
        return matches
    cost = []
    for i, old in enumerate(disappeared):
        old_centroid = old['centroid']
        old_dir = old['direction'] / (np.linalg.norm(old['direction']) + 1e-6)
        for j, new in enumerate(new_segments):
            new_centroid = new['centroid']
            delta = new_centroid - old_centroid
            if np.linalg.norm(delta) < 1e-6:
                angle = 0.0
            else:
                delta_dir = delta / np.linalg.norm(delta)
                cos_val = np.clip(np.dot(old_dir, delta_dir), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_val))
            cost.append((angle, i, j))
    cost.sort(key=lambda x: x[0])
    used_old = set()
    used_new = set()
    for angle, i, j in cost:
        if angle > angle_thresh_deg:
            break
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        matches.append((i, j))
    return matches


def icp_register_sequence(point_clouds: List[o3d.geometry.PointCloud], max_correspondence_distance: float = 1.0, init_voxel_size: float = 0.5, aggregate_window: int = 3) -> Tuple[List[np.ndarray], List[o3d.geometry.PointCloud]]:
    """Sequentially register a vehicle point cloud sequence and aggregate clouds in a sliding window."""
    poses: List[np.ndarray] = []
    clouds_aligned: List[o3d.geometry.PointCloud] = []
    if not point_clouds:
        return poses, clouds_aligned
    prev_cloud = point_clouds[0].voxel_down_sample(init_voxel_size)
    poses.append(np.eye(4, dtype=np.float64))
    clouds_aligned.append(prev_cloud)
    aggregated = prev_cloud
    for idx in range(1, len(point_clouds)):
        source = point_clouds[idx].voxel_down_sample(init_voxel_size)
        target = aggregated
        icp = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            np.eye(4, dtype=np.float64),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        transform = icp.transformation
        poses.append(transform)
        aligned = source.transform(transform)
        clouds_aligned.append(aligned)
        if idx < aggregate_window:
            aggregated = aggregated + aligned
        else:
            keep = [clouds_aligned[i] for i in range(max(0, idx - aggregate_window + 1), idx + 1)]
            aggregated = keep[0]
            for c in keep[1:]:
                aggregated += c
        aggregated = aggregated.voxel_down_sample(init_voxel_size)
    return poses, clouds_aligned


def fit_oriented_bounding_box(cloud: o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
    return cloud.get_oriented_bounding_box()


def pose_error_metrics(estimated: List[np.ndarray], truth: List[np.ndarray]) -> Dict[str, float]:
    translations = []
    rotations = []
    for est, gt in zip(estimated, truth):
        dt = np.linalg.norm(est[:3, 3] - gt[:3, 3])
        translations.append(dt)
        R_est = est[:3, :3]
        R_gt = gt[:3, :3]
        dR = R_est @ R_gt.T
        trace = np.clip(np.trace(dR), -1.0, 3.0)
        angle = np.degrees(np.arccos((trace - 1) / 2))
        rotations.append(angle)
    return {
        'mean_translation_error': float(np.mean(translations)) if translations else 0.0,
        'mean_rotation_error_deg': float(np.mean(rotations)) if rotations else 0.0,
    }
