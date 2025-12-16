import cv2
import numpy as np
from typing import Tuple, List, Dict

class HomographySmoother:
    def __init__(self, alpha: float = 0.9):
        """
        Initializes the homography smoother.

        Args:
            alpha (float): Smoothing factor, between 0 and 1. Higher values give more weight to the current homography.
        """
        self.alpha = alpha  # Smoothing factor
        self.smoothed_H = None  # Store the smoothed homography matrix

    def smooth(self, current_H: np.ndarray) -> np.ndarray:
        """
        Smooths the homography matrix using exponential smoothing.

        Args:
            current_H (np.ndarray): The current homography matrix of shape (3, 3).

        Returns:
            np.ndarray: The smoothed homography matrix of shape (3, 3).
        """
        if self.smoothed_H is None:
            # Initialize with the first homography matrix
            self.smoothed_H = current_H
        else:
            # Apply exponential smoothing
            self.smoothed_H = self.alpha * current_H + (1 - self.alpha) * self.smoothed_H

        return self.smoothed_H


def _filter_and_validate_keypoints(keypoints: dict, top_down_keypoints: np.ndarray) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    过滤和验证关键点，提高单应性矩阵计算的稳定性
    
    Args:
        keypoints: 检测到的关键点字典
        top_down_keypoints: 参考的俯视图关键点
        
    Returns:
        过滤后的关键点字典，格式为 {key: (detected_point, reference_point)}
    """
    filtered_keypoints = {}
    
    for key in keypoints.keys():
        # 检查关键点索引是否有效
        if key < 0 or key >= len(top_down_keypoints):
            print(f"[Warning] Key {key} out of range for top_down_keypoints (max: {len(top_down_keypoints)-1}).")
            continue
            
        detected_point = keypoints[key]
        reference_point = top_down_keypoints[key]
        
        # 验证检测到的关键点
        if not _is_valid_point(detected_point):
            print(f"[Warning] Invalid detected keypoint at index {key}: {detected_point}")
            continue
            
        # 验证参考关键点
        if not _is_valid_point(reference_point):
            print(f"[Warning] Invalid reference keypoint at index {key}: {reference_point}")
            continue
            
        # 检查关键点是否在合理范围内
        if not _is_reasonable_point(detected_point):
            print(f"[Warning] Detected keypoint {key} outside reasonable range: {detected_point}")
            continue
            
        filtered_keypoints[key] = (tuple(detected_point), tuple(reference_point))
    
    # 如果关键点太少，尝试放宽条件
    if len(filtered_keypoints) < 4:
        print(f"[Info] Strict filtering left {len(filtered_keypoints)} points, trying relaxed filtering...")
        filtered_keypoints = _relaxed_keypoint_filtering(keypoints, top_down_keypoints)
    
    # 关键点质量分析
    if len(filtered_keypoints) >= 4:
        quality_score = _assess_keypoint_quality(filtered_keypoints)
        print(f"[Info] Keypoint quality assessment: {quality_score:.2f}/1.0 ({len(filtered_keypoints)} points)")
        
        # 如果质量太低，尝试选择最好的关键点
        if quality_score < 0.3 and len(filtered_keypoints) > 4:
            filtered_keypoints = _select_best_keypoints(filtered_keypoints, min_count=4)
            print(f"[Info] Selected {len(filtered_keypoints)} best quality keypoints")
    
    return filtered_keypoints


def _is_valid_point(point) -> bool:
    """检查点是否有效（非None、长度为2、非NaN、非无穷大）"""
    try:
        if point is None or len(point) != 2:
            return False
        x, y = point
        return not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y))
    except (TypeError, ValueError):
        return False


def _is_reasonable_point(point, max_coord: float = 1e5) -> bool:
    """检查点坐标是否在合理范围内"""
    try:
        x, y = point
        return abs(x) < max_coord and abs(y) < max_coord
    except (TypeError, ValueError):
        return False


def _relaxed_keypoint_filtering(keypoints: dict, top_down_keypoints: np.ndarray) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """放宽条件的关键点过滤"""
    filtered_keypoints = {}
    
    for key in keypoints.keys():
        if key < 0 or key >= len(top_down_keypoints):
            continue
            
        detected_point = keypoints[key]
        reference_point = top_down_keypoints[key]
        
        # 只检查最基本的有效性
        if _is_valid_point(detected_point) and _is_valid_point(reference_point):
            filtered_keypoints[key] = (tuple(detected_point), tuple(reference_point))
    
    return filtered_keypoints


def _assess_keypoint_quality(keypoints: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]) -> float:
    """评估关键点质量"""
    if len(keypoints) < 4:
        return 0.0
    
    # 检查关键点分布是否合理（避免所有点都聚集在一个区域）
    detected_points = [kp[0] for kp in keypoints.values()]
    reference_points = [kp[1] for kp in keypoints.values()]
    
    # 计算点的分散程度
    detected_std = np.std(detected_points, axis=0)
    reference_std = np.std(reference_points, axis=0)
    
    # 分散程度评分（标准差越大越好，但有上限）
    spread_score = min(1.0, (detected_std[0] + detected_std[1] + reference_std[0] + reference_std[1]) / 1000.0)
    
    # 点数评分
    count_score = min(1.0, len(keypoints) / 8.0)  # 8个或更多关键点得满分
    
    return (spread_score + count_score) / 2.0


def _select_best_keypoints(keypoints: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]], min_count: int = 4) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """选择质量最好的关键点"""
    if len(keypoints) <= min_count:
        return keypoints
    
    # 计算每个关键点到其他点的平均距离（选择分散的点）
    points = list(keypoints.values())
    keys = list(keypoints.keys())
    scores = []
    
    for i, (detected, reference) in enumerate(points):
        # 计算到其他检测点的平均距离
        distances = []
        for j, (other_detected, _) in enumerate(points):
            if i != j:
                dist = np.sqrt((detected[0] - other_detected[0])**2 + (detected[1] - other_detected[1])**2)
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        scores.append((keys[i], avg_distance))
    
    # 选择距离最大的points（更分散）
    scores.sort(key=lambda x: x[1], reverse=True)
    selected_keys = [key for key, _ in scores[:max(min_count, len(keypoints)//2)]]
    
    return {key: keypoints[key] for key in selected_keys}


def get_homography_bak(keypoints: dict, top_down_keypoints: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix between detected keypoints and top-down keypoints.

    Args:
        keypoints (dict): A dictionary of detected keypoints, where keys are identifiers 
        and values are (x, y) coordinates.
        top_down_keypoints (np.ndarray): An array of shape (n, 2) containing the top-down keypoints.

    Returns:
        np.ndarray: A 3x3 homography matrix that maps the keypoints to the top-down view.
    """
    kps: List[Tuple[float, float]] = []
    proj_kps: List[Tuple[float, float]] = []

    for key in keypoints.keys():
        kps.append(keypoints[key])
        proj_kps.append(top_down_keypoints[key])

    def _compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """
        Compute a single homography matrix between source and destination points.

        Args:
            src_points (array): Source points coordinates of shape (n, 2).
            dst_points (array): Destination points coordinates of shape (n, 2).

        Returns:
            np.ndarray: The computed homography matrix of shape (3, 3).
        """
        if len(src_points) < 4 or len(dst_points) < 4:
            print(f"[Warning] Not enough keypoints for homography. Got {len(kps)}.")
            return None
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        h, _ = cv2.findHomography(src_points, dst_points)

        return h.astype(np.float32)

    H = _compute_homography(np.array(kps), np.array(proj_kps))

    return H


def get_homography(keypoints: dict, top_down_keypoints: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix between detected keypoints and top-down keypoints.

    Args:
        keypoints (dict): Detected keypoints, keys are ids and values are (x, y) coords.
        top_down_keypoints (np.ndarray): Array of shape (n, 2) for top-down view.

    Returns:
        Optional[np.ndarray]: A 3x3 homography matrix, or None if not computable.
    """
    # 关键点质量过滤和验证
    filtered_keypoints = _filter_and_validate_keypoints(keypoints, top_down_keypoints)
    
    if len(filtered_keypoints) < 4:
        print(f"[Warning] After filtering, only {len(filtered_keypoints)} keypoints remain, need at least 4.")
        return None
    
    kps: List[Tuple[float, float]] = []
    proj_kps: List[Tuple[float, float]] = []

    for key, (detected_point, reference_point) in filtered_keypoints.items():
        kps.append(detected_point)
        proj_kps.append(reference_point)

    def _compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        """
        Compute homography matrix with comprehensive error handling.
        
        Args:
            src_points: Source points array
            dst_points: Destination points array
            
        Returns:
            Homography matrix or None if computation fails
        """
        # 检查点数是否足够（至少4个点）
        if len(src_points) < 4 or len(dst_points) < 4:
            print(f"[Warning] Not enough keypoints for homography. Got {len(src_points)} points, need at least 4.")
            return None
            
        # 检查点数是否匹配
        if len(src_points) != len(dst_points):
            print(f"[Warning] Source and destination points count mismatch: {len(src_points)} vs {len(dst_points)}.")
            return None
            
        try:
            # 确保输入数据类型正确
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            
            # 检查点的有效性（非NaN, 非无穷大）
            if np.any(np.isnan(src_points)) or np.any(np.isnan(dst_points)):
                print("[Warning] NaN values detected in keypoints.")
                return None
                
            if np.any(np.isinf(src_points)) or np.any(np.isinf(dst_points)):
                print("[Warning] Infinite values detected in keypoints.")
                return None
            
            # 使用RANSAC方法计算单应性矩阵，提高鲁棒性
            h, mask = cv2.findHomography(
                src_points, 
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,  # RANSAC重投影阈值
                confidence=0.99,            # 置信度
                maxIters=2000              # 最大迭代次数
            )
            
            if h is None:
                print("[Warning] Homography computation returned None.")
                return None
                
            # 验证单应性矩阵的有效性
            if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                print("[Warning] Invalid homography matrix (contains NaN or Inf).")
                return None
                
            # 检查单应性矩阵的条件数，避免奇异矩阵
            try:
                cond_num = np.linalg.cond(h)
                if cond_num > 1e12:  # 条件数过大表示矩阵接近奇异
                    print(f"[Warning] Homography matrix is ill-conditioned (condition number: {cond_num:.2e}).")
                    return None
            except np.linalg.LinAlgError:
                print("[Warning] Failed to compute condition number of homography matrix.")
                return None
                
            return h.astype(np.float32)
            
        except cv2.error as e:
            print(f"[Error] OpenCV error in homography computation: {e}")
            return None
        except Exception as e:
            print(f"[Error] Unexpected error in homography computation: {e}")
            return None

    return _compute_homography(np.array(kps), np.array(proj_kps))


def apply_homography(pos: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    Apply a homography transformation to a 2D point with comprehensive error handling.

    Args:
        pos (Tuple[float, float]): The (x, y) coordinates of the point to be projected.
        H (np.ndarray): The homography matrix of shape (3, 3).

    Returns:
        Tuple[float, float]: The projected (x, y) coordinates in the destination space.
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If transformation fails
    """
    try:
        # 输入验证
        if pos is None or len(pos) != 2:
            raise ValueError("Position must be a tuple/list of 2 elements")
            
        if H is None:
            raise ValueError("Homography matrix cannot be None")
            
        if H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got shape {H.shape}")
        
        x, y = pos
        
        # 检查输入坐标的有效性
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            raise ValueError(f"Invalid input coordinates: ({x}, {y})")
        
        # 检查单应性矩阵的有效性
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            raise ValueError("Homography matrix contains NaN or infinite values")
        
        # 构建齐次坐标
        pos_homogeneous = np.array([float(x), float(y), 1.0], dtype=np.float64)
        
        # 应用单应性变换（使用更稳定的float64精度）
        projected_pos = np.dot(H.astype(np.float64), pos_homogeneous)
        
        # 检查齐次坐标的第三个分量
        if abs(projected_pos[2]) < 1e-10:  # 避免除零错误
            raise RuntimeError(f"Homography transformation resulted in near-zero homogeneous coordinate: {projected_pos[2]}")
        
        # 归一化齐次坐标
        projected_pos /= projected_pos[2]
        
        # 检查结果的有效性
        result_x, result_y = projected_pos[0], projected_pos[1]
        if np.isnan(result_x) or np.isnan(result_y) or np.isinf(result_x) or np.isinf(result_y):
            raise RuntimeError(f"Transformation resulted in invalid coordinates: ({result_x}, {result_y})")
        
        # 检查结果是否在合理范围内（可选的健壮性检查）
        if abs(result_x) > 1e6 or abs(result_y) > 1e6:
            print(f"[Warning] Large projection coordinates detected: ({result_x:.2f}, {result_y:.2f})")
        
        return float(result_x), float(result_y)
        
    except (ValueError, RuntimeError) as e:
        # 重新抛出已知的错误
        raise e
    except Exception as e:
        # 捕获其他未预期的错误
        raise RuntimeError(f"Unexpected error in homography transformation: {e}")


def get_opencv_version() -> str:
    """
    获取当前OpenCV版本信息
    
    Returns:
        OpenCV版本字符串
    """
    try:
        return cv2.__version__
    except:
        return "Unknown"


def is_opencv_compatible() -> bool:
    """
    检查OpenCV版本兼容性
    
    Returns:
        True if compatible, False otherwise
    """
    try:
        version = cv2.__version__
        major, minor = version.split('.')[:2]
        major, minor = int(major), int(minor)
        
        # 支持OpenCV 3.4+ 和 4.x+
        if major >= 4:
            return True
        elif major == 3 and minor >= 4:
            return True
        else:
            print(f"[Warning] OpenCV {version} may not be fully compatible. Recommended: 3.4+ or 4.x+")
            return False
    except Exception as e:
        print(f"[Warning] Could not determine OpenCV version: {e}")
        return False