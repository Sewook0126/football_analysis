"""
Point Detection Tracker - 使用目标检测模型检测球场点作为关键点

这个tracker将检测到的"unknown"目标转换为关键点输出，用于替代关键点检测模型
"""

from tracking.abstract_tracker import AbstractTracker
import cv2
import supervision as sv
from typing import List, Dict, Tuple
from ultralytics.engine.results import Results
import numpy as np


class PointDetectTracker(AbstractTracker):
    """使用目标检测模型检测球场点并模拟关键点输出"""

    def __init__(self, model_path: str, conf: float = 0.5,
                 input_size: int = 640, original_size: tuple = (1920, 1080),
                 min_points: int = 4) -> None:
        """
        Initialize PointDetectTracker for detecting field points.
        
        Args:
            model_path (str): Point detection model path.
            conf (float): Confidence threshold for point detection.
            input_size (int): Model input size (e.g., 640, 1280).
            original_size (tuple): Original video size (width, height).
            min_points (int): Minimum number of points required (default: 4).
        """
        super().__init__(model_path, conf)
        self.tracks = []
        self.cur_frame = 0
        
        # 使用传入的配置参数
        self.input_size = input_size
        self.original_size = original_size
        self.scale_x = original_size[0] / input_size
        self.scale_y = original_size[1] / input_size
        self.min_points = min_points
        
        print(f"✓ PointDetectTracker配置: {input_size}x{input_size} → {original_size[0]}x{original_size[1]}")
        print(f"   缩放因子: X={self.scale_x:.3f}, Y={self.scale_y:.3f}")
        print(f"   最小关键点数: {min_points}")
        print(f"   显示模式: 仅圆圈标记（无ID数字）")

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Perform point detection on multiple frames.

        Args:
            frames (List[np.ndarray]): List of frames for detection.
        
        Returns:
            List[Results]: Detected points for each frame
        """
        # 预处理：调整对比度
        contrast_adjusted_frames = [self._preprocess_frame(frame) for frame in frames]

        # 使用YOLO目标检测
        detections = self.model.predict(
            contrast_adjusted_frames, 
            conf=self.conf, 
            imgsz=self.input_size, 
            verbose=False  # 减少输出
        )
        return detections

    def track(self, detection: Results) -> Dict[int, Tuple[float, float]]:
        """
        将检测到的目标转换为关键点格式（简化版，不追踪ID）.
        
        Args:
            detection (Results): 检测结果
        
        Returns:
            dict: 关键点字典 {index: (x, y)}
        """
        # 转换为supervision格式
        detections_sv = sv.Detections.from_ultralytics(detection)
        
        if len(detections_sv) == 0:
            return {}
        
        # 提取边界框中心点作为关键点（简单分配ID）
        keypoints = {}
        for idx, (bbox, confidence, class_id) in enumerate(zip(
            detections_sv.xyxy, 
            detections_sv.confidence,
            detections_sv.class_id
        )):
            # 只处理高置信度的检测
            if confidence >= self.conf:
                # 计算边界框中心点
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                
                # 验证坐标在有效范围内
                if 0 <= x_center <= self.input_size and 0 <= y_center <= self.input_size:
                    # 缩放到原始尺寸
                    scaled_x = x_center * self.scale_x
                    scaled_y = y_center * self.scale_y
                    
                    keypoints[idx] = (scaled_x, scaled_y)
        
        self.tracks.append(detections_sv)
        self.cur_frame += 1
        
        # 输出检测信息（每10帧输出一次详细信息）
        if len(keypoints) > 0:
            if self.cur_frame % 10 == 0:
                print(f"[PointDetect] 帧 {self.cur_frame}: ✓ 检测到 {len(keypoints)} 个关键点")
                # 输出前3个关键点的位置作为示例
                sample_points = list(keypoints.items())[:3]
                for idx, (x, y) in sample_points:
                    print(f"           关键点 {idx}: ({x:.1f}, {y:.1f})")
        else:
            # 每帧都提醒（如果没有检测到），因为这是异常情况
            if self.cur_frame % 10 == 0:
                print(f"[PointDetect] ⚠️ 警告: 帧 {self.cur_frame} 未检测到关键点")
                print(f"           置信度阈值: {self.conf:.2f}, 建议降低到 0.2-0.3")
        
        return keypoints

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧：调整对比度并缩放到模型输入尺寸
        
        Args:
            frame (np.ndarray): 输入帧
        
        Returns:
            np.ndarray: 处理后的帧
        """
        # 调整对比度
        frame = self._adjust_contrast(frame)
        
        # 缩放到输入尺寸
        resized_frame = cv2.resize(frame, (self.input_size, self.input_size))
        
        return resized_frame
    
    def _adjust_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        使用直方图均衡化调整对比度
        
        Args:
            frame (np.ndarray): 输入帧
        
        Returns:
            np.ndarray: 对比度调整后的帧
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # 转换到YUV色彩空间
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # 对Y通道（亮度）应用直方图均衡化
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # 转换回BGR
            frame_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # 灰度图直接均衡化
            frame_equalized = cv2.equalizeHist(frame)
        
        return frame_equalized
    
    def get_statistics(self) -> dict:
        """获取跟踪统计信息"""
        return {
            'total_frames': self.cur_frame,
            'total_tracks': len(self.tracks),
            'average_points_per_frame': np.mean([len(t) for t in self.tracks]) if self.tracks else 0
        }

