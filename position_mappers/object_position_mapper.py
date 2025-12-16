from .abstract_mapper import AbstractMapper
from .homography import get_homography, apply_homography, HomographySmoother, is_opencv_compatible, get_opencv_version
from utils.bbox_utils import get_feet_pos

import numpy as np

class ObjectPositionMapper(AbstractMapper):
    """
    A class to map object positions from detected keypoints to a top-down view.

    This class implements the mapping of detected objects to their corresponding
    positions in a top-down representation based on the homography obtained from 
    detected keypoints.
    """

    def __init__(self, top_down_keypoints: np.ndarray, alpha: float = 0.9) -> None:
        """
        Initializes the ObjectPositionMapper.

        Args:
            top_down_keypoints (np.ndarray): An array of shape (n, 2) containing the top-down keypoints.
            alpha (float): Smoothing factor for homography smoothing.
        """
        super().__init__()
        self.top_down_keypoints = top_down_keypoints
        self.homography_smoother = HomographySmoother(alpha=alpha)
        
        # 版本兼容性检查
        opencv_version = get_opencv_version()
        if is_opencv_compatible():
            print(f"[ObjectPositionMapper] OpenCV版本兼容: {opencv_version}")
        else:
            print(f"[ObjectPositionMapper] OpenCV版本警告: {opencv_version} - 可能存在兼容性问题")
        
        # 统计信息
        self.homography_success_count = 0
        self.homography_failure_count = 0
        self.fallback_usage_count = 0

    def map(self, detection: dict) -> dict:
        """Maps the detection data to their positions in the top-down view.

        This method retrieves keypoints and object information from the detection data,
        computes the homography matrix, smooths it over frames, and projects the foot positions
        of detected objects.

        Args:
            detection (dict): The detection data containing keypoints and object information.

        Returns:
            dict: The detection data with projected positions added.
        """
        detection = detection.copy()
        
        keypoints = detection['keypoints']
        object_data = detection['object']

        if not keypoints or not object_data:
            # 即使没有关键点，也要为所有对象添加默认的projection字段
            #  使用边界框位置而不是固定中心点，确保每个对象有唯一位置
            for _, object_info in object_data.items():
                for track_id, track_info in object_info.items():
                    if 'projection' not in track_info:
                        bbox = track_info.get('bbox', [0, 0, 100, 100])
                        # 使用边界框中心作为投影位置，加上基于ID的小偏移
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        # 添加基于track_id的唯一偏移，确保不同对象有不同位置
                        offset_x = (hash(str(track_id)) % 200 - 100) * 0.1  # ±10像素偏移
                        offset_y = (hash(str(track_id)) % 200 - 100) * 0.1
                        track_info['projection'] = (float(center_x + offset_x), float(center_y + offset_y))
            return detection

        try:
            # 尝试计算单应性矩阵
            H = get_homography(keypoints, self.top_down_keypoints)
            
            if H is None:
                # 单应性矩阵计算失败，使用备用方案
                self.homography_failure_count += 1
                print(f"[ObjectPositionMapper] 单应性矩阵计算失败，使用备用投影方案 (失败次数: {self.homography_failure_count})")
                self._apply_fallback_projection(object_data)
                return detection
            
            self.homography_success_count += 1
            
            # 单应性矩阵有效，进行平滑处理
            try:
                smoothed_H = self.homography_smoother.smooth(H)
                if smoothed_H is None:
                    print("[ObjectPositionMapper] 单应性矩阵平滑失败，使用备用投影方案")
                    self._apply_fallback_projection(object_data)
                    return detection
            except Exception as e:
                print(f"[ObjectPositionMapper] 单应性矩阵平滑异常: {e}，使用备用投影方案")
                self._apply_fallback_projection(object_data)
                return detection

            # 应用单应性变换到所有对象
            projection_success_count = 0
            total_objects = 0
            
            for _, object_info in object_data.items():
                for track_id, track_info in object_info.items():
                    total_objects += 1
                    try:
                        bbox = track_info['bbox']
                        feet_pos = get_feet_pos(bbox)  # Get the foot position
                        
                        # 验证脚部位置的有效性
                        if feet_pos is None or len(feet_pos) != 2:
                            raise ValueError("Invalid feet position")
                            
                        if np.isnan(feet_pos[0]) or np.isnan(feet_pos[1]):
                            raise ValueError("NaN values in feet position")
                        
                        projected_pos = apply_homography(feet_pos, smoothed_H)
                        
                        # 验证投影结果的有效性
                        if projected_pos is None or len(projected_pos) != 2:
                            raise ValueError("Invalid projection result")
                            
                        if np.isnan(projected_pos[0]) or np.isnan(projected_pos[1]):
                            raise ValueError("NaN values in projection result")
                        
                        track_info['projection'] = projected_pos
                        projection_success_count += 1
                        
                    except Exception as e:
                        # 如果个别对象的投影失败，使用智能备用方案
                        self.fallback_usage_count += 1
                        print(f"[ObjectPositionMapper] 对象{track_id}投影失败: {e}，使用备用位置")
                        track_info['projection'] = self._get_fallback_position(track_info, track_id)
            
            # 检查投影成功率，如果太低则报告
            success_rate = projection_success_count / total_objects if total_objects > 0 else 0
            if success_rate < 0.5:  # 成功率低于50%
                print(f"[ObjectPositionMapper] 投影成功率较低: {success_rate:.1%} ({projection_success_count}/{total_objects})")
                
        except Exception as e:
            # 如果整体投影失败，为所有对象添加基于边界框的位置
            print(f"[ObjectPositionMapper] 整体投影失败: {e}，使用备用投影方案")
            self._apply_fallback_projection(object_data)

        return detection
    
    def _apply_fallback_projection(self, object_data: dict) -> None:
        """
        应用备用投影方案，为所有对象提供基于边界框的位置
        
        Args:
            object_data: 对象检测数据
        """
        for _, object_info in object_data.items():
            for track_id, track_info in object_info.items():
                if 'projection' not in track_info:
                    track_info['projection'] = self._get_fallback_position(track_info, track_id)
    
    def _get_fallback_position(self, track_info: dict, track_id: int) -> tuple:
        """
        获取单个对象的备用位置
        
        Args:
            track_info: 单个对象的跟踪信息
            track_id: 对象ID
            
        Returns:
            备用位置坐标 (x, y)
        """
        bbox = track_info.get('bbox', [0, 0, 100, 100])
        
        # 使用边界框底部中心作为脚部位置（更符合实际）
        center_x = (bbox[0] + bbox[2]) / 2
        bottom_y = bbox[3]  # 使用底部而不是中心
        
        # 添加基于track_id的唯一偏移，确保不同对象有不同位置
        # 使用更稳定的偏移算法
        hash_value = hash(str(track_id)) % 10000
        offset_x = (hash_value % 21 - 10) * 2.0  # ±20像素偏移
        offset_y = ((hash_value // 21) % 21 - 10) * 2.0  # ±20像素偏移
        
        fallback_x = float(center_x + offset_x)
        fallback_y = float(bottom_y + offset_y)
        
        return (fallback_x, fallback_y)
    
    def get_statistics(self) -> dict:
        """
        获取位置映射的统计信息
        
        Returns:
            统计信息字典
        """
        total_attempts = self.homography_success_count + self.homography_failure_count
        success_rate = self.homography_success_count / total_attempts if total_attempts > 0 else 0
        
        return {
            'homography_success_count': self.homography_success_count,
            'homography_failure_count': self.homography_failure_count,
            'homography_success_rate': success_rate,
            'fallback_usage_count': self.fallback_usage_count,
            'total_attempts': total_attempts
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("ObjectPositionMapper 统计信息:")
        print(f"  单应性矩阵计算成功: {stats['homography_success_count']}")
        print(f"  单应性矩阵计算失败: {stats['homography_failure_count']}")
        print(f"  成功率: {stats['homography_success_rate']:.1%}")
        print(f"  备用方案使用次数: {stats['fallback_usage_count']}")
        print(f"  总尝试次数: {stats['total_attempts']}")
        print("=" * 50)