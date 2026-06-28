import numpy as np
import open3d as o3d
from lidar_trajectory_methods import (
    build_spatio_temporal_stack,
    distance_partition_normalize,
    incremental_spatio_temporal_clustering,
    compute_centroid,
    principal_direction,
    re_associate_fragments,
    icp_register_sequence,
    fit_oriented_bounding_box,
)


def create_synthetic_vehicle_point_cloud(position, heading, n_points=150, noise=0.05):
    center = np.array(position, dtype=np.float32)
    angle = heading
    length, width = 4.2, 1.8
    corners = np.array([
        [ length / 2,  width / 2, 0.0],
        [ length / 2, -width / 2, 0.0],
        [-length / 2, -width / 2, 0.0],
        [-length / 2,  width / 2, 0.0],
    ], dtype=np.float32)
    pts = np.random.uniform(-1.0, 1.0, size=(n_points, 2))
    pts[:, 0] *= length / 2
    pts[:, 1] *= width / 2
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    rotated = pts @ rot.T
    cloud = np.zeros((n_points, 3), dtype=np.float32)
    cloud[:, :2] = rotated + center[:2]
    cloud[:, 2] = center[2] + np.random.normal(0.0, 0.02, size=n_points)
    cloud += np.random.normal(0.0, noise, size=cloud.shape)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))


def run_demo():
    timestamps = [0.0, 0.2, 0.4, 0.6]
    headings = [0.0, 0.04, 0.08, 0.12]
    positions = [[0.0, 0.0, 0.0], [0.4, 0.02, 0.0], [0.8, 0.05, 0.0], [1.2, 0.1, 0.0]]
    frames = [create_synthetic_vehicle_point_cloud(pos, hdg) for pos, hdg in zip(positions, headings)]

    stack = build_spatio_temporal_stack(frames, timestamps)
    print('Spatio-temporal stack shape:', stack.shape)

    sensor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    normalized = distance_partition_normalize(stack[:, :3], sensor, num_bins=4)
    print('Normalized first 5 rows:\n', normalized[:5])

    labels_by_frame = incremental_spatio_temporal_clustering(frames, timestamps, eps_space=1.0, eps_time=0.3, min_samples=8, window_frames=3, overlap_frames=1)
    for idx, labels in labels_by_frame.items():
        print(f'Frame {idx} labels unique:', np.unique(labels))

    fragments_old = [
        {'centroid': np.array([0.8, 0.05, 0.0]), 'direction': np.array([1.0, 0.1, 0.0])},
    ]
    fragments_new = [
        {'centroid': np.array([1.2, 0.1, 0.0]), 'direction': np.array([1.0, 0.05, 0.0])},
    ]
    assoc = re_associate_fragments(fragments_old, fragments_new)
    print('Fragment re-association matches:', assoc)

    poses, aligned_clouds = icp_register_sequence(frames, max_correspondence_distance=1.0, init_voxel_size=0.1, aggregate_window=2)
    print('ICP poses count:', len(poses))
    for i, pose in enumerate(poses):
        print(f'Pose {i}: translation {pose[:3,3]}')

    merged_cloud = aligned_clouds[-1]
    bbox = fit_oriented_bounding_box(merged_cloud)
    print('Fitted bounding box center:', bbox.center)
    print('Bounding box extent:', bbox.extent)

    o3d.io.write_point_cloud('demo_aligned_vehicle.pcd', merged_cloud)
    print('Saved demo aligned vehicle cloud to demo_aligned_vehicle.pcd')

if __name__ == '__main__':
    run_demo()
