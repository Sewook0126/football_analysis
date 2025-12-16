"""
清理版本的SpeedEstimator - 移除冗余函数，保留核心功能
"""

import math
import json
import os
import cv2
from collections import deque
from typing import Dict, Any, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def convert_numpy_types(obj):
    """递归转换NumPy数据类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

class SpeedEstimator:
    """精简版速度估算器 - 核心功能保留，移除冗余"""

    def __init__(self, field_width: int = 528, field_height: int = 352,
                 real_field_length: float = 105, real_field_width: float = 68, 
                 smoothing_window: int = 7, verbose_logging: bool = True,
                 scale_x: float = None, scale_y: float = None) -> None:
        """
        初始化速度估算器
        """
        # 场地参数
        self.field_width = field_width
        self.field_height = field_height
        self.real_field_length = real_field_length
        self.real_field_width = real_field_width
        
        # 比例尺设置与修正
        self.forced_scale_x = scale_x
        self.forced_scale_y = scale_y
        
        #  关键修正：应用比例尺修正因子
        scale_correction_factor = 0.15  # 修正过度估计的距离
        
        if scale_x is None:
            original_scale_x = real_field_length / field_width
            self.scale_x = original_scale_x * scale_correction_factor
        else:
            original_scale_x = scale_x
            self.scale_x = scale_x * scale_correction_factor
            
        if scale_y is None:
            original_scale_y = real_field_width / field_height
            self.scale_y = original_scale_y * scale_correction_factor
        else:
            original_scale_y = scale_y
            self.scale_y = scale_y * scale_correction_factor
            
        # 调试输出比例尺修正信息
        if verbose_logging:
            print(f" 比例尺修正详情:")
            print(f"   原始比例尺: X={original_scale_x:.6f}, Y={original_scale_y:.6f} m/pixel")
            print(f"   修正因子: {scale_correction_factor}")
            print(f"   修正后比例尺: X={self.scale_x:.6f}, Y={self.scale_y:.6f} m/pixel")
        
        # 核心数据结构
        self.previous_positions: Dict[Any, Tuple[Tuple[float, float], int]] = {}
        self.speed_history: Dict[Any, deque] = {}
        self.position_history: Dict[Any, deque] = {}
        self.player_states: Dict[Any, Dict] = {}
        
        # 速度计算参数
        self.smoothing_window = smoothing_window
        self.max_speed = 35.0  # km/h
        self.extreme_speed_threshold = 40.0  # 异常速度阈值 (更严格)
        
        #  优化的响应式速度参数 (增强版配置)
        self.speed_window_frames = 12        # 速度平均窗口: 12帧 (增强平滑)
        self.max_realistic_speed = 28.0      # 现实最大速度 km/h (适合足球)
        self.min_movement_speed = 1.0        # 最小运动速度阈值 km/h (更合理)
        self.stationary_threshold = 0.5      # 静止阈值 km/h (更实用)  
        self.position_smoothing = True       # 启用位置平滑
        self.speed_smoothing_factor = 0.85   # 速度平滑因子 (增强历史权重)
        
        # 时间窗口相关
        self.use_windowed_average = True     # 启用时间窗口平均
        self.min_window_size = 5             # 最小窗口大小 (更稳定启动)
        self.outlier_removal = True          # 启用异常值移除
        
        # 响应性优化
        self.quick_response_threshold = 5.0  # 快速响应阈值 km/h (降低敏感度)
        self.position_change_threshold = 0.5 # 位置变化阈值 (米)
        self.extreme_smoothing = False       # 极度平滑模式
        self.use_average_speed = True        # 使用平均速度而不是瞬时速度
        self.speed_update_interval = 3       # 速度更新间隔（帧数）- 减少间隔提高响应性
        
        #  新增跟踪质量控制
        self.min_tracking_records = 5        # 最少跟踪记录数（过滤误识别）- 进一步降低
        self.tracking_quality_threshold = 0.05 # 跟踪质量阈值 - 只过滤极低质量跟踪
        
        # 记录相关
        self.speed_records: List[Dict] = []
        self.player_max_speeds: Dict[Any, float] = {}
        self.player_avg_speeds: Dict[Any, List[float]] = {}
        self.verbose_logging = verbose_logging
        self.current_fps = None
        self.frame_count = 0
        
        # 平均速度计算相关
        self.last_speed_update_frame: Dict[Any, int] = {}  # 记录每个对象上次更新速度的帧数
        self.current_display_speed: Dict[Any, float] = {}  # 当前显示的速度
        
        # ROI收集（可选功能）
        self.player_frames: Dict[str, List[np.ndarray]] = {}
        self.player_bboxes: Dict[str, List[List[float]]] = {}
        self.roi_sample_frames = [30, 60, 90]  # 在这些帧采样ROI
        self.roi_margin = 20

    def calculate_speed(self, tracks: Dict[str, Any], frame_number: int, fps: float, 
                       keypoints: Dict[int, Tuple[float, float]] = None, 
                       original_frame: np.ndarray = None) -> Dict[str, Any]:
        """
        主要的速度计算函数
        """
        # 初始化帧率信息
        if self.current_fps is None:
            self.current_fps = fps
            if self.verbose_logging:
                print(f"[SpeedEstimator] 视频帧率: {fps:.2f} FPS")
                print(f"[SpeedEstimator] 比例尺: X={self.scale_x:.4f}, Y={self.scale_y:.4f} m/pixel")
        
        self.frame_count += 1
        
        # 处理所有跟踪对象
        for track_type, track_data in tracks.items():
            if track_type == 'keypoints':
                continue
                
            for track_id, track_info in track_data.items():
                if 'projection' not in track_info:
                    if self.verbose_logging and frame_number % 30 == 0:  # 每30帧打印一次
                        print(f"[DEBUG] {track_type} {track_id}: 没有投影坐标")
                    continue
                    
                current_position = track_info['projection']
                
                # 调试：检查投影坐标是否有效
                if self.verbose_logging and frame_number % 30 == 0:
                    print(f"[DEBUG] {track_type} {track_id}: 投影坐标 {current_position}")
                
                speed = self._calculate_object_speed(track_id, current_position, frame_number, fps)
                
                # 🚨 最终安全检查 - 确保速度绝对不超过限制
                final_speed = max(0.0, min(speed, self.max_speed))
                
                # 更新track信息
                track_info['speed'] = final_speed
                track_info['speed_kmh'] = final_speed
                        
                        # 记录速度数据
                self._record_speed_data(track_id, track_type, final_speed, current_position, frame_number, fps)
                
                # 收集ROI（如果需要）
                if original_frame is not None:
                    self._collect_roi_if_needed(track_id, track_type, track_info, original_frame, frame_number)
        
        return tracks

    def _calculate_object_speed(self, object_id: Any, current_position: Tuple[float, float], 
                               frame_number: int, fps: float) -> float:
        """ 优化的速度计算算法 - 增加跟踪质量验证"""
        
        # 初始化记录
        if object_id not in self.speed_history:
            self._init_object_history(object_id)
        if object_id not in self.last_speed_update_frame:
            self.last_speed_update_frame[object_id] = frame_number
            self.current_display_speed[object_id] = 0.0
        
        #  跟踪质量验证 - 检测突然出现的误识别（暂时禁用以诊断问题）
        tracking_quality = self._assess_tracking_quality(object_id, current_position, frame_number)
        
        # 暂时注释掉质量过滤，先让速度计算正常工作
        # if tracking_quality < 0.1:  # 质量太低，可能是误识别
        #     if self.verbose_logging and frame_number % 30 == 0:
        #         print(f"[DEBUG] 对象 {object_id}: 跟踪质量极低 ({tracking_quality:.2f})，跳过速度计算")
        #     return 0.0  # 返回0速度避免错误数据影响
        
        #  方法1: 减少更新频率
        frames_since_update = frame_number - self.last_speed_update_frame[object_id]
        
        # 只有达到更新间隔才重新计算速度，否则返回当前显示速度
        if frames_since_update < self.speed_update_interval:
            return self.current_display_speed[object_id]
        
        #  方法2: 使用更长跨度的平均速度
        if object_id in self.previous_positions:
            # 不是和上一帧比较，而是和更早的帧比较
            update_span = max(self.speed_update_interval, 5)  # 至少5帧跨度
            
            # 寻找合适的参考帧
            reference_position = None
            reference_frame = None
            
            # 查找历史位置记录中的合适参考点
            if len(self.position_history[object_id]) >= update_span:
                position_list = list(self.position_history[object_id])
                reference_position = position_list[-update_span]  # 取更早的位置
                reference_frame = frame_number - update_span
            else:
                # 如果历史不够长，使用最早的记录
                prev_position, prev_frame = self.previous_positions[object_id]
                reference_position = prev_position
                reference_frame = prev_frame
            
            if reference_position and reference_frame < frame_number:
                # 计算跨越多帧的平均速度
                distance = self._calculate_distance(reference_position, current_position)
                time_diff = (frame_number - reference_frame) / fps
                speed_ms = distance / time_diff if time_diff > 0 else 0
                speed_kmh = speed_ms * 3.6
                
                # 调试信息
                if self.verbose_logging and frame_number % 30 == 0:
                    print(f"[DEBUG] 对象 {object_id}: {update_span}帧平均速度={speed_kmh:.2f}km/h, 质量={tracking_quality:.2f}")
                
                #  增强的异常检测（暂时简化）
                # 基本异常检测
                if speed_kmh > self.extreme_speed_threshold:
                    if self.verbose_logging and frame_number % 30 == 0:
                        print(f"[DEBUG] 对象 {object_id}: 速度超出极限 ({speed_kmh:.2f} > {self.extreme_speed_threshold})")
                    speed_kmh = 0.0
                
                # 速度限制
                if speed_kmh > self.extreme_speed_threshold:
                    speed_kmh = 0.0
                elif speed_kmh > self.max_speed:
                    speed_kmh = self.max_speed
                
                # 与历史速度平滑
                if self.use_average_speed and object_id in self.current_display_speed:
                    prev_display = self.current_display_speed[object_id]
                    # 根据跟踪质量调整平滑权重
                    history_weight = 0.8 + (1.0 - tracking_quality) * 0.15  # 质量越低，越依赖历史
                    current_weight = 1.0 - history_weight
                    speed_kmh = history_weight * prev_display + current_weight * speed_kmh
                
                # 更新显示速度和记录
                self.current_display_speed[object_id] = max(0.0, min(speed_kmh, self.max_speed))
                self.last_speed_update_frame[object_id] = frame_number
                
                # 记录位置历史
                self.position_history[object_id].append(current_position)
                self.previous_positions[object_id] = (current_position, frame_number)
                
                return self.current_display_speed[object_id]
        
        # 首次出现
        self.previous_positions[object_id] = (current_position, frame_number)
        self.position_history[object_id].append(current_position)
        return 0.0
    
    def _assess_tracking_quality(self, object_id: Any, current_position: Tuple[float, float], 
                                frame_number: int) -> float:
        """评估跟踪质量，返回0-1的质量分数"""
        if object_id not in self.position_history or len(self.position_history[object_id]) == 0:
            return 0.5  # 新对象，中等质量
        
        quality_score = 1.0
        position_history = list(self.position_history[object_id])
        
        # 1. 检查位置连续性 - 突然的大距离跳跃表明跟踪错误
        if len(position_history) > 0:
            last_position = position_history[-1]
            distance = self._calculate_distance(last_position, current_position)
            
            # 如果距离超过合理范围（例如30米），降低质量
            if distance > 30.0:
                quality_score *= 0.1
            elif distance > 15.0:
                quality_score *= 0.3
            elif distance > 8.0:
                quality_score *= 0.6
        
        # 2. 检查历史跟踪长度 - 太短的跟踪历史表明可能是误识别（放宽条件）
        total_records = len(position_history)
        if total_records < 3:
            quality_score *= 0.1  # 记录太少，很可能是误识别
        elif total_records < 8:
            quality_score *= 0.4  # 记录较少，中等质量
        elif total_records < 20:
            quality_score *= 0.7  # 记录较多，较高质量
        
        # 3. 检查位置变化的一致性
        if len(position_history) >= 3:
            recent_positions = position_history[-3:]
            recent_distances = []
            for i in range(1, len(recent_positions)):
                dist = self._calculate_distance(recent_positions[i-1], recent_positions[i])
                recent_distances.append(dist)
            
            # 如果最近的位置变化过于剧烈，降低质量
            if recent_distances:
                max_recent_dist = max(recent_distances)
                if max_recent_dist > 20.0:
                    quality_score *= 0.2
                elif max_recent_dist > 10.0:
                    quality_score *= 0.5
        
        return max(0.0, min(1.0, quality_score))
    
    def _is_speed_anomaly(self, object_id: Any, speed_kmh: float, distance: float, time_diff: float) -> bool:
        """检测速度是否为异常值"""
        # 1. 基本物理限制检查
        if speed_kmh > self.extreme_speed_threshold:
            return True
        
        # 2. 距离合理性检查
        if distance > 50.0:  # 单次更新距离不应超过50米
            return True
        
        # 3. 时间合理性检查
        if time_diff < 0.1:  # 时间间隔太短
            return True
        
        # 4. 与历史速度对比检查
        if object_id in self.speed_history and len(self.speed_history[object_id]) >= 3:
            recent_speeds = list(self.speed_history[object_id])[-3:]
            avg_recent = np.mean(recent_speeds)
            
            # 如果当前速度是历史平均的3倍以上，可能是异常
            if speed_kmh > max(10.0, avg_recent * 3.0):
                return True
        
        return False
    
    def _init_object_history(self, object_id: Any) -> None:
        """初始化对象的历史记录 - 使用时间窗口大小"""
        self.speed_history[object_id] = deque(maxlen=self.speed_window_frames)
        self.position_history[object_id] = deque(maxlen=self.speed_window_frames)
        self.player_states[object_id] = {
            'recent_positions': deque(maxlen=10),  # 位置历史稍微长一些
            'stationary_count': 0,
            'is_stationary': False
        }

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点间的真实距离（米）"""
        dx = (pos2[0] - pos1[0]) * self.scale_x
        dy = (pos2[1] - pos1[1]) * self.scale_y
        return math.sqrt(dx**2 + dy**2)

    def _smooth_speed(self, object_id: Any, speed: float, position: Tuple[float, float]) -> float:
        """
        恢复并增强原有的平滑算法 - 包含突然停止检测等功能
        """
        # 确保历史记录已初始化
        if object_id not in self.speed_history:
            self._init_object_history(object_id)
        
        # 记录位置历史
        self.position_history[object_id].append(position)
        
        # 1. 异常值检测和过滤 (更严格)
        if speed > self.max_realistic_speed:
            speed = min(speed, self.max_realistic_speed)
        
        # 2. 使用增强的时间窗口平均算法
        if self.use_windowed_average:
            return self._calculate_windowed_average_speed(object_id, speed, position)
        else:
            # 简单平滑作为后备
            self.speed_history[object_id].append(speed)
            recent_speeds = list(self.speed_history[object_id])
            if len(recent_speeds) >= 3:
                # 使用更强的平滑
                return np.mean(recent_speeds) * 0.9 + speed * 0.1  # 90%历史，10%当前
            return speed
    
    def _calculate_windowed_average_speed(self, object_id: Any, current_speed: float, position: Tuple[float, float]) -> float:
        """
        优化的响应式时间窗口平均速度计算
        """
        # 添加当前速度到历史
        self.speed_history[object_id].append(current_speed)
        
        # 获取窗口内的速度数据
        recent_speeds = list(self.speed_history[object_id])
        
        # 如果数据不足，返回当前速度
        if len(recent_speeds) < self.min_window_size:
            return max(0.0, current_speed)
        
        #  极度平滑模式：禁用突然启停检测，始终使用平滑算法
        if hasattr(self, 'extreme_smoothing') and self.extreme_smoothing:
            # 极度平滑模式：直接使用平滑算法，不检测运动变化
            return self._apply_smooth_averaging(object_id, recent_speeds, current_speed)
        else:
            # 普通模式：保留原有的运动状态检测
            motion_change = self._detect_motion_change(object_id, current_speed, position)
            
            if motion_change == "sudden_stop":
                return self._apply_rapid_deceleration(object_id, current_speed)
            elif motion_change == "sudden_start":
                return self._apply_quick_acceleration(object_id, current_speed)
            else:
                return self._apply_smooth_averaging(object_id, recent_speeds, current_speed)
    
    def _detect_motion_change(self, object_id: Any, current_speed: float, position: Tuple[float, float]) -> str:
        """检测运动状态变化"""
        if len(self.speed_history[object_id]) < 3:
            return "normal"
        
        recent_speeds = list(self.speed_history[object_id])[-3:]  # 最近3帧
        avg_recent_speed = np.mean(recent_speeds)
        
        # 检查位置变化
        if len(self.position_history[object_id]) >= 2:
            prev_pos = list(self.position_history[object_id])[-1]
            pos_change = self._calculate_distance(prev_pos, position)
        else:
            pos_change = 0
        
        # 突然停止检测
        if (avg_recent_speed > self.quick_response_threshold and 
            current_speed < self.stationary_threshold and
            pos_change < self.position_change_threshold):
            return "sudden_stop"
        
        # 突然启动检测
        if (avg_recent_speed < self.stationary_threshold and 
            current_speed > self.quick_response_threshold and
            pos_change > self.position_change_threshold):
            return "sudden_start"
        
        return "normal"
    
    def _apply_rapid_deceleration(self, object_id: Any, current_speed: float) -> float:
        """应用快速减速"""
        if len(self.speed_history[object_id]) > 0:
            last_speed = list(self.speed_history[object_id])[-1]
            # 快速但平滑的减速
            result = max(0.0, last_speed * 0.3 + current_speed * 0.7)
            return min(result, self.max_speed)
        return min(current_speed, self.max_speed)
    
    def _apply_quick_acceleration(self, object_id: Any, current_speed: float) -> float:
        """应用快速加速响应"""
        if len(self.speed_history[object_id]) > 0:
            last_speed = list(self.speed_history[object_id])[-1]
            # 更重视当前速度
            result = last_speed * 0.2 + current_speed * 0.8
            return min(result, self.max_speed)
        return min(current_speed, self.max_speed)
    
    def _apply_smooth_averaging(self, object_id: Any, recent_speeds: List[float], current_speed: float) -> float:
        """增强的平滑平均算法 - 减少波动"""
        # 异常值移除 (使用四分位数方法)
        if self.outlier_removal and len(recent_speeds) >= 4:
            recent_speeds = self._remove_speed_outliers(recent_speeds)
        
        #  增强平滑：使用多种平均方法的组合
        # 1. 简单移动平均
        simple_avg = np.mean(recent_speeds)
        
        # 2. 加权移动平均 (更重视历史)
        weights = np.exp(np.linspace(-1, 0, len(recent_speeds)))  # 较缓的指数权重
        weighted_avg = np.average(recent_speeds, weights=weights)
        
        # 3. 中位数 (对异常值更鲁棒)
        median_speed = np.median(recent_speeds)
        
        # 4. 组合这些方法 (60%加权平均 + 30%简单平均 + 10%中位数)
        combined_speed = 0.6 * weighted_avg + 0.3 * simple_avg + 0.1 * median_speed
        
        # 5. 与历史速度进一步平滑 (根据模式调整权重)
        if len(self.speed_history[object_id]) > 1:
            previous_speed = list(self.speed_history[object_id])[-2]
            if hasattr(self, 'extreme_smoothing') and self.extreme_smoothing:
                # 极度平滑模式: 98%历史 + 2%当前 (几乎不变)
                combined_speed = 0.98 * previous_speed + 0.02 * combined_speed
            else:
                # 普通模式: 90%历史 + 10%当前
                combined_speed = 0.9 * previous_speed + 0.1 * combined_speed
        
        # 静止检测
        final_speed = self._apply_stationary_detection(object_id, combined_speed, 
                                                     list(self.position_history[object_id])[-1])
        
        #  极度平滑模式：限制单帧速度变化
        if hasattr(self, 'extreme_smoothing') and self.extreme_smoothing:
            if len(self.speed_history[object_id]) > 1:
                last_final_speed = list(self.speed_history[object_id])[-1]
                # 限制单帧最大变化为2 km/h
                max_change = 2.0
                if abs(final_speed - last_final_speed) > max_change:
                    if final_speed > last_final_speed:
                        final_speed = last_final_speed + max_change
                    else:
                        final_speed = max(0.0, last_final_speed - max_change)
        
        return max(0.0, min(final_speed, self.max_speed))
    
    def _remove_speed_outliers(self, speeds: List[float]) -> List[float]:
        """使用四分位数方法移除异常值"""
        if len(speeds) < 4:
            return speeds
            
        speeds_array = np.array(speeds)
        Q1 = np.percentile(speeds_array, 25)
        Q3 = np.percentile(speeds_array, 75)
        IQR = Q3 - Q1
        
        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 过滤异常值
        filtered_speeds = [s for s in speeds if lower_bound <= s <= upper_bound]
        
        # 如果过滤后数据太少，返回原始数据
        if len(filtered_speeds) < len(speeds) * 0.5:
            return speeds
        
        return filtered_speeds
    
    def _calculate_weighted_average(self, speeds: List[float]) -> float:
        """计算加权平均速度 (最近的帧权重更高)"""
        if not speeds:
            return 0.0
        
        # 生成权重 (线性递增)
        weights = np.arange(1, len(speeds) + 1, dtype=float)
        weights = weights / np.sum(weights)
        
        # 计算加权平均
        weighted_avg = np.average(speeds, weights=weights)
        return float(weighted_avg)
    
    def _apply_stationary_detection(self, object_id: Any, speed: float, position: Tuple[float, float]) -> float:
        """优化的静止检测逻辑"""
        # 检查位置变化
        if len(self.position_history[object_id]) >= 2:
            recent_positions = list(self.position_history[object_id])[-3:]  # 最近3个位置，更快响应
            position_changes = []
            
            for i in range(1, len(recent_positions)):
                dist = self._calculate_distance(recent_positions[i-1], recent_positions[i])
                position_changes.append(dist)
            
            avg_position_change = np.mean(position_changes) if position_changes else 0
            
            # 更敏感的静止判定
            if avg_position_change < 0.05 and speed < self.stationary_threshold:
                return 0.0
            
            # 如果位置有明显变化但速度很低，说明在缓慢移动
            if avg_position_change > 0.1 and speed < self.min_movement_speed:
                # 根据位置变化估算一个最小速度
                fps = self.current_fps if self.current_fps is not None else 25.0  # 默认25fps
                estimated_speed = avg_position_change * fps * 3.6  # 转换为km/h
                return min(estimated_speed, speed) if speed > 0 else estimated_speed
        
        # 最小运动速度过滤 - 更宽松
        if 0 < speed < self.min_movement_speed:
            # 不直接归零，而是保留一个小值
            return self.min_movement_speed * 0.8
        
        return speed
    
    def _is_stationary(self, object_id: Any, position: Tuple[float, float], speed: float) -> bool:
        """检测对象是否静止 - 更敏感的版本"""
        state = self.player_states[object_id]
        state['recent_positions'].append(position)
        
        # 速度检查 - 更严格
        speed_is_low = speed < self.stationary_threshold
        
        # 位置稳定性检查 - 更敏感
        position_stable = True
        if len(state['recent_positions']) >= 2:
            recent_positions = list(state['recent_positions'])
            # 只检查最近的几个位置变化
            check_length = min(3, len(recent_positions))
            movements = []
            for i in range(len(recent_positions) - check_length + 1, len(recent_positions)):
                if i > 0:
                    dist = self._calculate_distance(recent_positions[i-1], recent_positions[i])
                    movements.append(dist)
            avg_movement = np.mean(movements) if movements else 0
            position_stable = avg_movement < self.movement_noise_threshold
        
        # 如果检测到明显移动，立即取消静止状态
        if not speed_is_low or not position_stable:
            state['stationary_count'] = 0
            state['is_stationary'] = False
            return False
            
        # 更新静止计数
        if speed_is_low and position_stable:
            state['stationary_count'] += 1
        
        # 判定静止 - 需要更少帧数
        is_stationary = state['stationary_count'] >= self.stationary_frames_required
        state['is_stationary'] = is_stationary
        
        return is_stationary
    
    def _apply_gradual_decay(self, object_id: Any) -> float:
        """应用渐进式速度衰减"""
        if len(self.speed_history[object_id]) > 0:
            last_speed = list(self.speed_history[object_id])[-1]
            
            # 根据当前速度调整衰减速度
            if last_speed < 1.0:
                decay_factor = 0.3  # 快速衰减
            elif last_speed < 2.0:
                decay_factor = 0.5  # 中速衰减
            else:
                decay_factor = self.fast_decay_factor  # 正常衰减
            
            decayed = last_speed * decay_factor
            return 0.0 if decayed < self.stationary_threshold * 0.5 else decayed
        return 0.0
    
    def _filter_outliers(self, object_id: Any, speed: float) -> float:
        """温和的异常值过滤"""
        history = list(self.speed_history[object_id])
        if len(history) < 3:
            return speed
        
        # 使用最近几个值
        recent = history[-5:] if len(history) >= 5 else history
        median_speed = np.median(recent)
        mad = np.median(np.abs(np.array(recent) - median_speed))
        
        if mad > 0:
            threshold = median_speed + self.outlier_sensitivity * mad * 1.4826
            if speed > threshold:
                # 温和修正
                return speed * 0.3 + median_speed * 0.7
        
        return speed
    
    def _apply_trend_smoothing(self, object_id: Any, speed: float) -> float:
        """趋势感知平滑"""
        history = list(self.speed_history[object_id])
        if len(history) < 2:
            return speed
        
        # 计算趋势
        recent = history[-3:] if len(history) >= 3 else history
        if len(recent) >= 2:
            changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            avg_change = np.mean(changes)
            predicted = history[-1] + avg_change * 0.3  # 减弱趋势影响
            
            # 在测量值和预测值间加权
            return speed * 0.7 + predicted * 0.3
        
        return speed
    
    def _adaptive_averaging(self, object_id: Any, speed: float) -> float:
        """自适应加权平均"""
        history = list(self.speed_history[object_id])
        if len(history) < 2:
            return speed
        
        # 构建平滑窗口
        window_size = min(self.smooth_window_size, len(history) + 1)
        recent_speeds = history[-(window_size-1):] + [speed]
        
        # 根据稳定性调整权重
        if len(recent_speeds) >= 3:
            speed_std = np.std(recent_speeds)
            if speed_std < 1.0:  # 很稳定
                weights = np.ones(len(recent_speeds))
            elif speed_std < 3.0:  # 中等稳定
                weights = np.linspace(0.5, 1.5, len(recent_speeds))
            else:  # 不稳定
                weights = np.exp(np.linspace(-1, 0, len(recent_speeds)))
        else:
            weights = np.linspace(0.7, 1.3, len(recent_speeds))
        
        # 归一化并计算加权平均
        weights = weights / np.sum(weights)
        return np.sum(np.array(recent_speeds) * weights)
    
    def _record_speed_data(self, object_id: Any, track_type: str, speed: float, 
                          position: Tuple[float, float], frame_number: int, fps: float) -> None:
        """记录速度数据用于后续分析"""
        # 更新统计信息
        if object_id not in self.player_max_speeds:
            self.player_max_speeds[object_id] = 0.0
            self.player_avg_speeds[object_id] = []
        
        self.player_max_speeds[object_id] = max(self.player_max_speeds[object_id], speed)
        self.player_avg_speeds[object_id].append(speed)
        
        # 记录详细数据
        self.speed_records.append({
            'frame': frame_number,
            'time': frame_number / fps,
            'player_id': object_id,
            'track_type': track_type,
            'speed': speed,
            'position': position
        })
    
    def _collect_roi_if_needed(self, object_id: Any, track_type: str, track_info: Dict, 
                              frame: np.ndarray, frame_number: int) -> None:
        """收集ROI图像（可选功能）"""
        if frame_number not in self.roi_sample_frames:
            return
        
        if 'bbox' not in track_info:
            return
        
        player_key = f"{track_type}_{object_id}"
        
        if player_key not in self.player_frames:
            self.player_frames[player_key] = []
            self.player_bboxes[player_key] = []
        
        # 提取ROI
        bbox = track_info['bbox']
        roi = self._extract_roi(frame, bbox)
        
        if roi is not None:
            self.player_frames[player_key].append(roi)
            self.player_bboxes[player_key].append(bbox)
    
    def _extract_roi(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """提取ROI区域"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            h, w = frame.shape[:2]
            
            # 添加边距
            x1 = max(0, x1 - self.roi_margin)
            y1 = max(0, y1 - self.roi_margin)
            x2 = min(w, x2 + self.roi_margin)
            y2 = min(h, y2 + self.roi_margin)
            
            return frame[y1:y2, x1:x2]
        except:
            return None
    
    def set_responsiveness_parameters(self, stationary_threshold: float = 0.8,
                                    low_speed_threshold: float = 3.0,  # 兼容参数
                                    fast_decay_factor: float = 0.7,
                                    quick_response_mode: bool = True,
                                    smooth_window_size: int = 7,
                                    outlier_sensitivity: float = 2.5,
                                    trend_smoothing: bool = True) -> None:
        """设置响应性参数 - 兼容原始接口"""
        self.stationary_threshold = stationary_threshold
        self.low_speed_threshold = low_speed_threshold  # 保存但不直接使用
        self.fast_decay_factor = fast_decay_factor
        self.quick_response_mode = quick_response_mode
        self.smooth_window_size = smooth_window_size
        self.outlier_sensitivity = outlier_sensitivity
        self.trend_smoothing = trend_smoothing
        
        if self.verbose_logging:
            print(f"[SpeedEstimator] 速度计算参数已更新:")
            print(f"  静止阈值: {stationary_threshold} km/h")
            print(f"  低速阈值: {low_speed_threshold} km/h (兼容参数)")
            print(f"  快速衰减: {fast_decay_factor}")
            print(f"  快速响应: {quick_response_mode}")
            print(f"  平滑窗口: {smooth_window_size} 帧")
            print(f"  异常值敏感度: {outlier_sensitivity}")
            print(f"  趋势平滑: {trend_smoothing}")

    def save_speed_analysis(self, output_dir: str = "output_videos", filename: str = None) -> str:
        """保存速度分析结果"""
        if not self.speed_records:
            return ""
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"speed_analysis_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        # 准备数据
        analysis_data = {
            'metadata': {
                'total_frames': self.frame_count,
                'fps': self.current_fps,
                'scale_x': self.scale_x,
                'scale_y': self.scale_y,
                'max_speed_limit': self.max_speed
            },
            'player_stats': {},
            'speed_records': self.speed_records
        }
        
        # 计算每个球员的统计信息
        for player_id, speeds in self.player_avg_speeds.items():
            if speeds:
                analysis_data['player_stats'][str(player_id)] = {
                    'max_speed': float(self.player_max_speeds[player_id]),
                    'avg_speed': float(np.mean(speeds)),
                    'total_records': len(speeds)
                }
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
        
        if self.verbose_logging:
            print(f"[SpeedEstimator] 速度分析已保存: {filepath}")
        
        return filepath

    def print_speed_summary(self) -> None:
        """打印速度汇总信息 - 过滤低质量跟踪对象"""
        if not self.player_max_speeds:
            print("[SpeedEstimator] 没有速度数据")
            return
        
        #  过滤跟踪记录太少的对象（可能是误识别）
        min_records_threshold = 5   # 至少需要5条记录才被认为是有效跟踪
        
        filtered_players = {}
        filtered_count = 0
        
        for player_id, max_speed in self.player_max_speeds.items():
            if player_id in self.player_avg_speeds and self.player_avg_speeds[player_id]:
                record_count = len(self.player_avg_speeds[player_id])
                if record_count >= min_records_threshold:
                    filtered_players[player_id] = max_speed
                else:
                    filtered_count += 1
                    if self.verbose_logging:
                        print(f"[过滤] 对象 {player_id}: 记录数 {record_count} < {min_records_threshold}，可能是误识别")
        
        print(f"\n 速度分析汇总:")
        print(f"总处理帧数: {self.frame_count}")
        print(f"检测到 {len(self.player_max_speeds)} 个对象")
        print(f"过滤掉 {filtered_count} 个低质量跟踪对象（记录数 < {min_records_threshold}）")
        print(f"有效对象: {len(filtered_players)} 个")
        
        if not filtered_players:
            print("没有满足条件的有效跟踪对象")
            return
        
        # 排序显示
        sorted_players = sorted(filtered_players.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   {'对象ID':^8} | {'最高速度':^12} | {'平均速度':^12}")
        print("-" * 40)
        
        for player_id, max_speed in sorted_players:
            if player_id in self.player_avg_speeds and self.player_avg_speeds[player_id]:
                avg_speed = np.mean(self.player_avg_speeds[player_id])
                print(f"    {str(player_id):^8} | {max_speed:^12.1f} | {avg_speed:^12.1f}")

    def reset(self) -> None:
        """重置所有数据"""
        self.previous_positions = {}
        self.speed_history = {}
        self.position_history = {}
        self.player_states = {}
        self.speed_records = []
        self.player_max_speeds = {}
        self.player_avg_speeds = {}
        self.player_frames = {}
        self.player_bboxes = {}
        self.current_fps = None
        self.frame_count = 0

    # =============================================
    # 以下是可选的ROI和可视化功能，可以根据需要保留
    # =============================================
    
    def save_player_rois_and_curves(self, output_dir: str = "output_videos") -> None:
        """保存球员ROI和速度曲线（可选功能）"""
        if not self.player_frames:
            if self.verbose_logging:
                print("[SpeedEstimator] 没有收集到ROI数据")
            return
        
        print(f" 正在保存球员ROI和速度曲线...")
        
        for player_key in self.player_frames.keys():
            player_dir = os.path.join(output_dir, f"player_analysis", player_key)
            os.makedirs(player_dir, exist_ok=True)
            
            # 保存ROI图片
            self._save_roi_images(player_key, player_dir)
            
            # 保存速度曲线
            self._save_speed_curve(player_key, player_dir)
    
    def _save_roi_images(self, player_key: str, player_dir: str) -> None:
        """保存ROI图片"""
        if player_key not in self.player_frames:
            return
        
        for i, roi in enumerate(self.player_frames[player_key]):
            filename = f"roi_frame_{i+1}.jpg"
            filepath = os.path.join(player_dir, filename)
            cv2.imwrite(filepath, roi)
    
    def _save_speed_curve(self, player_key: str, player_dir: str) -> None:
        """保存速度曲线图"""
        # 提取球员ID
        parts = player_key.split('_')
        if len(parts) < 2:
            return
        
        track_type, player_id = parts[0], '_'.join(parts[1:])
        
        # 收集该球员的速度数据
        player_speeds = []
        for record in self.speed_records:
            if str(record['player_id']) == player_id and record['track_type'] == track_type:
                player_speeds.append({
                    'frame': record['frame'],
                    'time': record['time'],
                    'speed': record['speed']
                })
        
        if not player_speeds:
            return
        
        # 生成速度曲线图
        times = [s['time'] for s in player_speeds]
        speeds = [s['speed'] for s in player_speeds]
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, speeds, 'b-', linewidth=2, alpha=0.8, label='Speed')
        plt.axhline(y=self.stationary_threshold, color='r', linestyle='--', alpha=0.5, label='Stationary Threshold')
        
        plt.title(f'Player {player_id} Speed Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (km/h)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 保存图片
        filename = f"speed_curve_{player_id}.png"
        filepath = os.path.join(player_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose_logging:
            print(f" {player_id} 速度曲线已保存")

    def generate_speed_tables(self, save_dir: str = "output_videos", time_interval: float = 0.5,
                             file_format: str = 'excel', language: str = 'chinese') -> None:
        """
        生成速度表格，记录每个球员在指定时间间隔的速度
        
        Args:
            save_dir: 输出目录
            time_interval: 时间间隔（秒），可选0.1, 0.5等
            file_format: 文件格式，'csv' 或 'excel'
            language: 表格语言，'chinese' 或 'korean'
        """
        if not self.speed_records:
            if self.verbose_logging:
                print("[SpeedEstimator] 没有速度数据用于生成表格")
            return
        
        # 语言配置
        if language == 'english':
            headers = {
                'time_range': 'Time Range (sec)',
                'time_center': 'Time Center (sec)',
                'avg_speed': 'Average Speed (km/h)',
                'max_speed': 'Maximum Speed (km/h)',
                'min_speed': 'Minimum Speed (km/h)',
                'data_count': 'Data Points'
            }
            lang_display = 'English'
        else:  # chinese
            headers = {
                'time_range': '时间段(秒)',
                'time_center': '时间中点(秒)',
                'avg_speed': '平均速度(km/h)',
                'max_speed': '最大速度(km/h)',
                'min_speed': '最小速度(km/h)',
                'data_count': '数据点数'
            }
            lang_display = '中文'
        
        format_display = 'Excel' if file_format == 'excel' else 'CSV'
        print(f" 正在生成速度表格（格式: {format_display}, 语言: {lang_display}, 时间间隔: {time_interval}秒）...")
        
        # 按球员分组数据
        player_data = {}
        for record in self.speed_records:
            player_key = f"{record['track_type']}_{record['player_id']}"
            if player_key not in player_data:
                player_data[player_key] = []
            player_data[player_key].append(record)
        
        # 创建输出目录
        tables_dir = os.path.join(save_dir, "speed_tables")
        os.makedirs(tables_dir, exist_ok=True)
        
        import pandas as pd
        
        # 为每个球员生成表格
        generated_count = 0
        for player_key, data in player_data.items():
            if len(data) < 5:  # 数据点太少跳过
                continue
            
            # 按时间排序
            sorted_data = sorted(data, key=lambda x: x['time'])
            
            # 计算时间跨度
            max_time = sorted_data[-1]['time']
            num_intervals = int(max_time / time_interval) + 1
            
            # 为每个时间间隔提取速度数据
            table_data = []
            for i in range(num_intervals):
                interval_start = i * time_interval
                interval_end = (i + 1) * time_interval
                
                # 查找该时间间隔内的速度数据
                interval_speeds = [
                    d['speed'] for d in sorted_data 
                    if interval_start <= d['time'] < interval_end
                ]
                
                if interval_speeds:
                    avg_speed = np.mean(interval_speeds)
                    max_speed = np.max(interval_speeds)
                    min_speed = np.min(interval_speeds)
                    data_count = len(interval_speeds)
                else:
                    avg_speed = 0.0
                    max_speed = 0.0
                    min_speed = 0.0
                    data_count = 0
                
                table_data.append({
                    headers['time_range']: f'{interval_start:.2f}-{interval_end:.2f}',
                    headers['time_center']: f'{(interval_start + interval_end) / 2:.2f}',
                    headers['avg_speed']: f'{avg_speed:.2f}',
                    headers['max_speed']: f'{max_speed:.2f}',
                    headers['min_speed']: f'{min_speed:.2f}',
                    headers['data_count']: data_count
                })
            
            # 创建DataFrame
            df = pd.DataFrame(table_data)
            
            # 根据格式保存文件
            safe_filename = player_key.replace('/', '_').replace('\\', '_')
            
            try:
                if file_format == 'excel':
                    # 只保存Excel格式
                    excel_path = os.path.join(tables_dir, f"speed_table_{safe_filename}.xlsx")
                    df.to_excel(excel_path, index=False, sheet_name=f'{player_key}')
                    if self.verbose_logging:
                        print(f" {player_key} 速度表格已保存: {excel_path}")
                    generated_count += 1
                else:
                    # 只保存CSV格式
                    csv_path = os.path.join(tables_dir, f"speed_table_{safe_filename}.csv")
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    if self.verbose_logging:
                        print(f" {player_key} 速度表格已保存: {csv_path}")
                    generated_count += 1
            except Exception as e:
                print(f" ⚠️ {player_key} 表格保存失败: {e}")
                if file_format == 'excel':
                    print("     请确保已安装: pip install openpyxl")
        
        print(f" 速度表格已保存到: {tables_dir}")
        print(f" 共生成 {generated_count} 个球员的速度表格")

    def plot_speed_curves(self, save_dir: str = "output_videos") -> None:
        """
        绘制所有球员的速度曲线（兼容方法）
        """
        if not self.speed_records:
            if self.verbose_logging:
                print("[SpeedEstimator] 没有速度数据用于绘制曲线")
            return
        
        print(f" 正在生成速度变化曲线...")
        
        # 按球员分组数据
        player_data = {}
        for record in self.speed_records:
            player_key = f"{record['track_type']}_{record['player_id']}"
            if player_key not in player_data:
                player_data[player_key] = []
            player_data[player_key].append(record)
        
        # 创建输出目录
        curves_dir = os.path.join(save_dir, "speed_curves")
        os.makedirs(curves_dir, exist_ok=True)
        
        # 为每个球员生成曲线
        for player_key, data in player_data.items():
            if len(data) < 5:  # 数据点太少跳过
                continue
            
            times = [d['time'] for d in data]
            speeds = [d['speed'] for d in data]
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=(12, 6))
            plt.plot(times, speeds, 'b-', linewidth=2, alpha=0.8, label='Speed')
            plt.axhline(y=self.stationary_threshold, color='r', linestyle='--', 
                       alpha=0.5, label=f'Stationary Threshold ({self.stationary_threshold} km/h)')
            
            # 标记高速时刻
            high_speeds = [(t, s) for t, s in zip(times, speeds) if s > 15.0]
            if high_speeds:
                high_times, high_vals = zip(*high_speeds)
                plt.scatter(high_times, high_vals, color='red', s=30, alpha=0.7, 
                           label='Sprint (>15 km/h)', zorder=5)
            
            plt.title(f'Player {player_key} Speed Curve', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Speed (km/h)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 保存图片
            safe_filename = player_key.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(curves_dir, f"speed_curve_{safe_filename}.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f" 速度曲线已保存到: {curves_dir}")
        print(f" 共生成 {len(player_data)} 个球员的速度曲线")
    
    def calculate_opponent_distances(self, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
        """
        计算球员到最近对手的距离（兼容方法 - 简化版本）
        """
        try:
            # 收集所有球员的位置信息
            team1_players = []  # 队伍1
            team2_players = []  # 队伍2
            
            for track_type in ['player', 'goalkeeper']:
                if track_type in tracks:
                    for player_id, track_info in tracks[track_type].items():
                        if 'projection' not in track_info or 'club' not in track_info:
                            continue
                        
                        projection = track_info['projection']
                        club = track_info['club']
                        
                        player_data = (player_id, projection, track_type, club)
                        
                        if club == 'Club1':
                            team1_players.append(player_data)
                        elif club == 'Club2':
                            team2_players.append(player_data)
            
            # 如果没有足够的球员数据，直接返回
            if len(team1_players) == 0 or len(team2_players) == 0:
                return tracks
            
            # 为每个球员计算到最近对手的距离
            def calculate_min_distance(player_pos, opponent_team):
                """计算到对手队伍最近球员的距离"""
                if not opponent_team:
                    return float('inf')
                
                min_dist = float('inf')
                for _, opp_pos, _, _ in opponent_team:
                    dist = self._calculate_distance(player_pos, opp_pos)
                    min_dist = min(min_dist, dist)
                return min_dist
            
            # 为队伍1的球员计算距离
            for player_id, projection, track_type, _ in team1_players:
                if track_type in tracks and player_id in tracks[track_type]:
                    min_dist = calculate_min_distance(projection, team2_players)
                    tracks[track_type][player_id]['nearest_opponent_distance'] = min_dist
            
            # 为队伍2的球员计算距离
            for player_id, projection, track_type, _ in team2_players:
                if track_type in tracks and player_id in tracks[track_type]:
                    min_dist = calculate_min_distance(projection, team1_players)
                    tracks[track_type][player_id]['nearest_opponent_distance'] = min_dist
            
        except Exception as e:
            if self.verbose_logging:
                print(f"[SpeedEstimator] 计算对手距离时出错: {e}")
            # 出错时不影响主流程，继续返回tracks
        
        return tracks
