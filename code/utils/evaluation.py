"""
评估指标模块
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment


def calculate_trajectory_error(pred_traj: np.ndarray,
                               gt_traj: np.ndarray) -> Dict[str, float]:
    """
    计算轨迹误差

    Args:
        pred_traj: 预测轨迹 (N x 3)
        gt_traj: 真实轨迹 (M x 3)

    Returns:
        误差指标字典
    """
    # 对齐长度
    min_len = min(len(pred_traj), len(gt_traj))
    pred = pred_traj[:min_len]
    gt = gt_traj[:min_len]

    # 位置误差
    position_errors = np.linalg.norm(pred - gt, axis=1)

    metrics = {
        'ADE': position_errors.mean(),  # Average Displacement Error
        'FDE': position_errors[-1],     # Final Displacement Error
        'max_error': position_errors.max(),
        'min_error': position_errors.min(),
        'std_error': position_errors.std(),
    }

    return metrics


def calculate_mota(num_matches: int,
                  num_misses: int,
                  num_false_positives: int,
                  num_gt: int) -> float:
    """
    计算MOTA (Multiple Object Tracking Accuracy)

    Args:
        num_matches: 正确匹配数
        num_misses: 漏检数
        num_false_positives: 误检数
        num_gt: 真实目标总数

    Returns:
        MOTA分数
    """
    if num_gt == 0:
        return 0.0

    mota = 1 - (num_misses + num_false_positives) / num_gt
    return max(0.0, mota)


def calculate_motp(distance_sum: float, num_matches: int) -> float:
    """
    计算MOTP (Multiple Object Tracking Precision)

    Args:
        distance_sum: 所有匹配的距离总和
        num_matches: 匹配数

    Returns:
        MOTP分数
    """
    if num_matches == 0:
        return 0.0

    return distance_sum / num_matches


def calculate_idf1(num_true_positives: int,
                  num_false_positives: int,
                  num_false_negatives: int) -> float:
    """
    计算IDF1 (ID F1 Score)

    Args:
        num_true_positives: 真阳性数
        num_false_positives: 假阳性数
        num_false_negatives: 假阴性数

    Returns:
        IDF1分数
    """
    if num_true_positives == 0:
        return 0.0

    precision = num_true_positives / (num_true_positives + num_false_positives)
    recall = num_true_positives / (num_true_positives + num_false_negatives)

    if precision + recall == 0:
        return 0.0

    idf1 = 2 * precision * recall / (precision + recall)
    return idf1


def calculate_classification_metrics(y_true: List[str],
                                    y_pred: List[str]) -> Dict[str, float]:
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        指标字典
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics


def calculate_lane_assignment_accuracy(pred_lanes: Dict[int, int],
                                       gt_lanes: Dict[int, int]) -> float:
    """
    计算车道分配准确率

    Args:
        pred_lanes: 预测的车道分配 {track_id: lane_id}
        gt_lanes: 真实的车道分配 {track_id: lane_id}

    Returns:
        准确率
    """
    common_ids = set(pred_lanes.keys()) & set(gt_lanes.keys())

    if len(common_ids) == 0:
        return 0.0

    correct = sum(1 for tid in common_ids if pred_lanes[tid] == gt_lanes[tid])
    accuracy = correct / len(common_ids)

    return accuracy


def calculate_tracking_completeness(tracks: List,
                                    min_length: int = 5) -> Dict[str, float]:
    """
    计算轨迹完整性

    Args:
        tracks: 轨迹列表
        min_length: 最小有效长度

    Returns:
        完整性指标
    """
    if len(tracks) == 0:
        return {
            'num_tracks': 0,
            'avg_length': 0.0,
            'valid_ratio': 0.0,
            'max_length': 0,
            'min_length': 0
        }

    lengths = [len(track.observations) for track in tracks]
    valid_tracks = [t for t in tracks if len(t.observations) >= min_length]

    metrics = {
        'num_tracks': len(tracks),
        'num_valid_tracks': len(valid_tracks),
        'avg_length': np.mean(lengths),
        'valid_ratio': len(valid_tracks) / len(tracks),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'std_length': np.std(lengths)
    }

    return metrics


def evaluate_background_removal(original_points: int,
                                foreground_points: int,
                                gt_foreground_points: int = None) -> Dict[str, float]:
    """
    评估背景去除效果

    Args:
        original_points: 原始点数
        foreground_points: 前景点数
        gt_foreground_points: 真实前景点数（可选）

    Returns:
        评估指标
    """
    removal_rate = (1 - foreground_points / original_points) * 100

    metrics = {
        'original_points': original_points,
        'foreground_points': foreground_points,
        'removal_rate': removal_rate,
        'compression_ratio': original_points / foreground_points if foreground_points > 0 else 0
    }

    if gt_foreground_points is not None:
        error_rate = abs(foreground_points - gt_foreground_points) / gt_foreground_points * 100
        metrics['error_rate'] = error_rate

    return metrics


def calculate_processing_speed(num_frames: int,
                               total_time: float,
                               num_points_per_frame: int = None) -> Dict[str, float]:
    """
    计算处理速度

    Args:
        num_frames: 帧数
        total_time: 总时间（秒）
        num_points_per_frame: 每帧点数（可选）

    Returns:
        速度指标
    """
    fps = num_frames / total_time if total_time > 0 else 0
    time_per_frame = total_time / num_frames if num_frames > 0 else 0

    metrics = {
        'fps': fps,
        'time_per_frame': time_per_frame,
        'total_time': total_time,
        'num_frames': num_frames
    }

    if num_points_per_frame is not None:
        points_per_second = num_points_per_frame * fps
        metrics['points_per_second'] = points_per_second

    return metrics


def print_evaluation_report(metrics_dict: Dict[str, any],
                           title: str = "Evaluation Report"):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        elif isinstance(value, int):
            print(f"{key:.<40} {value}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"  {sub_key:.<38} {sub_value:.4f}")
                else:
                    print(f"  {sub_key:.<38} {sub_value}")
        else:
            print(f"{key:.<40} {value}")

    print("=" * 70 + "\n")
