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
    """
    递归转换NumPy数据类型为Python原生类型，用于JSON序列化
    """
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
    """Estimates the speed of objects (km/h)."""

    def __init__(self, field_width: int = 528, field_height: int = 352,
                 real_field_length: float = 105, real_field_width: float = 68, 
                 smoothing_window: int = 5, verbose_logging: bool = True,
                 scale_x: float = None, scale_y: float = None) -> None:
        """
        Initialize the SpeedEstimator with the field dimensions and real-world measurements.

        Args:
            field_width (int): Width of the field in pixels (projection image).
            field_height (int): Height of the field in pixels (projection image).
            real_field_length (float): Real-world length of the field in meters (default: 105m).
            real_field_width (float): Real-world width of the field in meters (default: 68m).
            smoothing_window (int): Number of frames to consider for speed smoothing.
            verbose_logging (bool): Whether to print detailed speed logs for each frame.
            scale_x (float, optional): Force X-axis scale factor (meters per pixel). 
                                     If None, will be calculated dynamically from keypoints.
            scale_y (float, optional): Force Y-axis scale factor (meters per pixel). 
                                     If None, will be calculated dynamically from keypoints.
        """
        self.field_width = field_width
        self.field_height = field_height
        self.real_field_length = real_field_length  # 标准足球场长度
        self.real_field_width = real_field_width    # 标准足球场宽度
        self.previous_positions: Dict[Any, Tuple[Tuple[float, float], int]] = {}
        self.speed_history: Dict[Any, deque] = {}
        self.position_history: Dict[Any, deque] = {}  # 位置历史记录
        self.velocity_history: Dict[Any, deque] = {}  # 速度向量历史记录
        self.smoothing_window = smoothing_window
        
        # 高级平滑参数
        self.adaptive_smoothing = True  # 自适应平滑
        self.kalman_filters: Dict[Any, Dict] = {}  # 简单卡尔曼滤波器
        self.outlier_threshold = 3.0  # 异常值阈值（标准差倍数）
        
        # 动态比例尺相关
        self.forced_scale_x = scale_x  # 强制指定的X轴比例
        self.forced_scale_y = scale_y  # 强制指定的Y轴比例
        self.scale_x = None  # 当前使用的X轴比例
        self.scale_y = None  # 当前使用的Y轴比例
        self.scale_history = []  # 比例尺历史记录用于平滑
        self.scale_confidence = 0.0  # 比例尺置信度
        self.fallback_scale_x = real_field_length / field_width  # 备用比例尺
        self.fallback_scale_y = real_field_width / field_height
        
        # 关键点相关
        self.keypoints_history = []  # 关键点历史记录
        self.last_valid_keypoints = None  # 最后一次有效的关键点
        
        # Maximum realistic speed (km/h)
        self.max_speed = 35.0
        
        #  快速响应和静止检测参数 (优化版)
        self.stationary_threshold = 0.8  # km/h - 低于此速度视为静止
        self.low_speed_threshold = 3.0   # km/h - 低速阈值
        self.fast_decay_factor = 0.7     # 快速衰减因子 (0.7 = 每帧保留70%速度，更平滑)
        self.movement_noise_threshold = 0.15  # 位置噪声阈值(米)，提高以减少误判
        self.stationary_frames_required = 4  # 连续4帧静止才判定为停止，更稳定
        self.quick_response_mode = True  # 启用快速响应模式
        
        #  平滑优化参数
        self.smooth_window_size = 7      # 增大平滑窗口，提高稳定性
        self.outlier_sensitivity = 2.5   # 异常值检测敏感度 (标准差倍数)
        self.trend_smoothing = True      # 启用趋势平滑
        
        # 球员状态跟踪
        self.player_states: Dict[Any, Dict] = {}  # 每个球员的状态信息
        
        # 速度记录相关
        self.speed_records: List[Dict] = []  # 存储所有帧的速度记录
        self.player_max_speeds: Dict[Any, float] = {}  # 每个球员的最高速度
        self.player_avg_speeds: Dict[Any, List[float]] = {}  # 每个球员的速度历史用于计算平均值
        self.verbose_logging = verbose_logging  # 是否打印详细日志
        
        # 帧率相关
        self.current_fps = None  # 当前视频的实际帧率
        self.frame_count = 0     # 处理的帧数计数
        
        # ROI截取相关
        self.player_frames: Dict[Any, List[np.ndarray]] = {}  # 存储每个球员的帧图像
        self.player_bboxes: Dict[Any, List[List]] = {}        # 存储每个球员的边界框
        self.roi_sample_frames = [10, 50, 100]  # 在这些帧数截取ROI（可调整）
        self.roi_margin = 20     # ROI边距（像素）
        
        # 初始化比例尺
        self._initialize_scale()

    def _initialize_scale(self) -> None:
        """
        初始化比例尺设置
        """
        if self.forced_scale_x is not None:
            self.scale_x = self.forced_scale_x
            print(f"[SpeedEstimator] 使用强制指定的X轴比例: {self.scale_x:.4f} m/pixel")
        else:
            self.scale_x = self.fallback_scale_x
            print(f"[SpeedEstimator] 初始化X轴比例(备用): {self.scale_x:.4f} m/pixel")
        
        if self.forced_scale_y is not None:
            self.scale_y = self.forced_scale_y
            print(f"[SpeedEstimator] 使用强制指定的Y轴比例: {self.scale_y:.4f} m/pixel")
        else:
            self.scale_y = self.fallback_scale_y
            print(f"[SpeedEstimator] 初始化Y轴比例(备用): {self.scale_y:.4f} m/pixel")

    def _calculate_scale_from_keypoints(self, keypoints: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        基于检测到的关键点动态计算比例尺
        
        Args:
            keypoints: 检测到的关键点字典 {id: (x, y)}
            
        Returns:
            Tuple[float, float, float]: (scale_x, scale_y, confidence)
        """
        if not keypoints or len(keypoints) < 4:
            return self.scale_x, self.scale_y, 0.0
        
        try:
            # 定义已知的实际距离（米）用于计算比例尺
            known_distances = []
            
            # 球门线长度 (约16.5米)
            if 6 in keypoints and 7 in keypoints:  # 左球门区角点
                goal_area_width = self._distance_between_points(keypoints[6], keypoints[7])
                known_distances.append((goal_area_width, 18.32))  # 球门区宽度18.32米
            
            if 22 in keypoints and 23 in keypoints:  # 右球门区角点
                goal_area_width = self._distance_between_points(keypoints[22], keypoints[23])
                known_distances.append((goal_area_width, 18.32))
            
            # 禁区宽度 (约40.32米)
            if 9 in keypoints and 12 in keypoints:  # 左禁区
                penalty_area_width = self._distance_between_points(keypoints[9], keypoints[12])
                known_distances.append((penalty_area_width, 40.32))  # 禁区宽度40.32米
            
            if 17 in keypoints and 20 in keypoints:  # 右禁区
                penalty_area_width = self._distance_between_points(keypoints[17], keypoints[20])
                known_distances.append((penalty_area_width, 40.32))
            
            # 场地长度
            if 0 in keypoints and 24 in keypoints:  # 左右球门线中点
                field_length = self._distance_between_points(keypoints[0], keypoints[24])
                known_distances.append((field_length, self.real_field_length))
            
            # 场地宽度
            if 13 in keypoints and 16 in keypoints:  # 中线两端
                field_width = self._distance_between_points(keypoints[13], keypoints[16])
                known_distances.append((field_width, self.real_field_width))
            
            # 中圆直径 (约18.3米)
            if 30 in keypoints and 31 in keypoints:  # 中圆左右端点
                center_circle_diameter = self._distance_between_points(keypoints[30], keypoints[31])
                known_distances.append((center_circle_diameter, 18.30))  # 中圆直径18.30米
            
            if not known_distances:
                return self.scale_x, self.scale_y, 0.0
            
            # 计算平均比例尺
            scales = []
            for pixel_dist, real_dist in known_distances:
                if pixel_dist > 0:
                    scale = real_dist / pixel_dist
                    scales.append(scale)
            
            if not scales:
                return self.scale_x, self.scale_y, 0.0
            
            # 使用中位数来避免异常值影响
            avg_scale = np.median(scales)
            confidence = min(1.0, len(scales) / 5.0)  # 基于可用距离数量的置信度
            
            # 检查比例尺是否合理 (0.05-0.5 m/pixel 是合理范围)
            if 0.05 <= avg_scale <= 0.5:
                return avg_scale, avg_scale, confidence
            else:
                return self.scale_x, self.scale_y, 0.0
                
        except Exception as e:
            if self.verbose_logging:
                print(f"[SpeedEstimator] 关键点比例尺计算失败: {e}")
            return self.scale_x, self.scale_y, 0.0

    def _distance_between_points(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        计算两点之间的像素距离
        """
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _update_scale_with_keypoints(self, keypoints: Dict[int, Tuple[float, float]]) -> None:
        """
        使用关键点更新比例尺
        """
        # 如果强制指定了比例尺，则不进行动态更新
        if self.forced_scale_x is not None and self.forced_scale_y is not None:
            return
        
        new_scale_x, new_scale_y, confidence = self._calculate_scale_from_keypoints(keypoints)
        
        # 如果置信度足够高，更新比例尺
        if confidence > 0.3:  # 至少需要30%的置信度
            # 使用历史记录进行平滑
            self.scale_history.append((new_scale_x, new_scale_y, confidence))
            
            # 保留最近10帧的比例尺记录
            if len(self.scale_history) > 10:
                self.scale_history.pop(0)
            
            # 计算加权平均比例尺
            total_weight = sum(conf for _, _, conf in self.scale_history)
            if total_weight > 0:
                weighted_scale_x = sum(sx * conf for sx, _, conf in self.scale_history) / total_weight
                weighted_scale_y = sum(sy * conf for _, sy, conf in self.scale_history) / total_weight
                
                # 更新比例尺（如果没有强制指定）
                if self.forced_scale_x is None:
                    self.scale_x = weighted_scale_x
                if self.forced_scale_y is None:
                    self.scale_y = weighted_scale_y
                
                self.scale_confidence = min(1.0, total_weight / len(self.scale_history))
                
                if self.verbose_logging and self.frame_count % 30 == 0:  # 每30帧打印一次
                    print(f"[SpeedEstimator] 动态比例尺更新: "
                          f"X={self.scale_x:.4f} m/pixel, Y={self.scale_y:.4f} m/pixel, "
                          f"置信度={self.scale_confidence:.2f}")
        
        # 保存最后一次有效的关键点
        if len(keypoints) >= 4:
            self.last_valid_keypoints = keypoints.copy()

    def _estimate_scale_from_player_movements(self, tracks: Dict[str, Any]) -> Tuple[float, float]:
        """
        基于球员移动模式估算比例尺（备用方法）
        当关键点检测完全失败时使用
        """
        if not self.previous_positions or len(self.previous_positions) < 3:
            return self.scale_x, self.scale_y
        
        try:
            # 分析球员移动速度的统计分布
            movement_distances = []
            for player_id, (prev_pos, prev_frame) in self.previous_positions.items():
                if player_id in tracks.get('player', {}):
                    current_track = tracks['player'][player_id]
                    if 'projection' in current_track:
                        current_pos = current_track['projection']
                        pixel_dist = self._distance_between_points(prev_pos, current_pos)
                        if pixel_dist > 0:
                            movement_distances.append(pixel_dist)
            
            if len(movement_distances) >= 3:
                # 使用中位数移动距离来估算比例尺
                median_movement = np.median(movement_distances)
                # 假设合理的球员移动速度范围是2-15 km/h
                # 在一帧内的移动距离应该在合理范围内
                if median_movement > 0:
                    frame_time = 1.0 / self.current_fps if self.current_fps else 1.0/25.0
                    # 假设中位数速度为8 km/h (2.22 m/s)
                    expected_distance_meters = 2.22 * frame_time
                    estimated_scale = expected_distance_meters / median_movement
                    
                    # 检查估算结果是否合理
                    if 0.05 <= estimated_scale <= 0.5:
                        return estimated_scale, estimated_scale
            
        except Exception as e:
            if self.verbose_logging:
                print(f"[SpeedEstimator] 基于球员移动的比例尺估算失败: {e}")
        
        return self.scale_x, self.scale_y

    def _init_player_history(self, player_id: Any) -> None:
        """
        为新球员初始化所有历史记录
        """
        if player_id not in self.speed_history:
            self.speed_history[player_id] = deque([0.0] * self.smoothing_window, maxlen=self.smoothing_window * 2)
        if player_id not in self.position_history:
            self.position_history[player_id] = deque(maxlen=self.smoothing_window)
        if player_id not in self.velocity_history:
            self.velocity_history[player_id] = deque(maxlen=self.smoothing_window)
        if player_id not in self.kalman_filters:
            self.kalman_filters[player_id] = self._init_kalman_filter()
        if player_id not in self.player_states:
            self.player_states[player_id] = {
                'recent_positions': deque(maxlen=self.stationary_frames_required + 2),
                'recent_speeds': deque(maxlen=self.stationary_frames_required + 2),
                'stationary_count': 0,
                'last_movement_frame': 0,
                'is_stationary': False
            }

    def get_scale_info(self) -> Dict[str, Any]:
        """
        获取当前比例尺信息（用于调试和监控）
        """
        return {
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'scale_confidence': self.scale_confidence,
            'forced_scale_x': self.forced_scale_x,
            'forced_scale_y': self.forced_scale_y,
            'fallback_scale_x': self.fallback_scale_x,
            'fallback_scale_y': self.fallback_scale_y,
            'scale_history_length': len(self.scale_history),
            'has_valid_keypoints': self.last_valid_keypoints is not None
        }

    def calculate_opponent_distances(self, tracks: Dict[str, Dict[int, Any]]) -> Dict[str, Dict[int, Any]]:
        """
        Calculate distance to the nearest opponent player for each player.

        Args:
            tracks (Dict[str, Dict[int, Any]]): The tracking data with club information.

        Returns:
            Dict[str, Dict[int, Any]]: The updated tracking data with distance to nearest opponent.
        """
        # 组织为列表便于处理
        attack_team = []
        defend_team = []

        for track_type in ['goalkeeper', 'player']:
            for player_id, track in tracks[track_type].items():
                # 必须包含 projection（投影坐标）和 club（所属队伍）信息
                if 'projection' in track and 'club' in track:
                    player_data = (player_id, track['projection'], track_type, track['club'])
                    if track['club'] == 'Club1':
                        attack_team.append(player_data)
                    else:
                        defend_team.append(player_data)

        # 计算最近对手距离
        def find_min_dist(player_proj, opponents):
            min_dist = float('inf')
            for _, opp_proj, _, _ in opponents:
                dist = self._calculate_distance(player_proj, opp_proj)
                min_dist = min(min_dist, dist)
            return min_dist if min_dist != float('inf') else 0.0

        # 对每个进攻方球员找最近防守方
        for player_id, proj, track_type, _ in attack_team:
            nearest_dist = find_min_dist(proj, defend_team)
            tracks[track_type][player_id]['nearest_opponent_distance'] = nearest_dist

        # 对每个防守方球员找最近进攻方
        for player_id, proj, track_type, _ in defend_team:
            nearest_dist = find_min_dist(proj, attack_team)
            tracks[track_type][player_id]['nearest_opponent_distance'] = nearest_dist

        return tracks
        

    def calculate_speed(self, tracks: Dict[str, Any], frame_number: int, fps: float, keypoints: Dict[int, Tuple[float, float]] = None, original_frame: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate the speed of players based on their projections and update the track information.

        Args:
            tracks (Dict[str, Any]): A dictionary containing tracking information for players.
            frame_number (int): The current frame number of the video.
            fps (float): Frames per second of the video.
            keypoints (Dict[int, Tuple[float, float]], optional): Detected keypoints for dynamic scale calculation.

        Returns:
            Dict[str, Any]: Updated tracks with calculated speeds.
        """
        # 更新当前帧率信息
        if self.current_fps is None:
            self.current_fps = fps
            print(f"[SpeedEstimator] 视频帧率: {fps:.2f} FPS")
            print(f"[SpeedEstimator] 场地尺寸: {self.field_width}x{self.field_height} pixels")
            print(f"[SpeedEstimator] 实际场地: {self.real_field_length}x{self.real_field_width} meters")
            print(f"[SpeedEstimator] 初始缩放因子: X={self.scale_x:.4f} m/pixel, Y={self.scale_y:.4f} m/pixel")
        
        self.frame_count += 1
        
        # 动态更新比例尺（基于关键点）
        if keypoints:
            self._update_scale_with_keypoints(keypoints)
        elif self.last_valid_keypoints and self.frame_count % 10 == 0:
            # 每10帧使用最后一次有效的关键点尝试更新
            self._update_scale_with_keypoints(self.last_valid_keypoints)
        
        # 检查坐标范围（仅第一帧）
        self.check_coordinate_range(tracks)
        
        frame_speed_data = {
            'frame': frame_number,
            'timestamp': frame_number / fps,
            'fps': fps,
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'scale_confidence': self.scale_confidence,
            'players': {}
        }
        
        for track_type in tracks:
            for player_id, track in tracks[track_type].items():
                if 'projection' in track:
                    current_position = track['projection']
                    
                    if player_id in self.previous_positions:
                        prev_position, prev_frame = self.previous_positions[player_id]
                        
                        # Calculate distance in meters
                        distance = self._calculate_distance(prev_position, current_position)
                        
                        # Calculate time difference in seconds (使用实际帧差和帧率)
                        frame_diff = frame_number - prev_frame
                        time_diff = frame_diff / fps
                        
                        # Calculate speed in km/h
                        speed = (distance / time_diff) * 3.6 if time_diff > 0 else 0.0
                        
                        # 检查异常速度
                        is_abnormal_speed = speed > self.max_speed
                        
                        # Apply maximum speed check
                        capped_speed = min(speed, self.max_speed)
                        
                        # Apply advanced smoothing with position information
                        smoothed_speed = self._smooth_speed(player_id, capped_speed, current_position)
                        
                        # Add speed to track
                        tracks[track_type][player_id]['speed'] = smoothed_speed
                        
                        # 打印每帧的速度计算日志
                        if self.verbose_logging and (smoothed_speed > 0.5 or is_abnormal_speed):
                            status = "CAPPED" if is_abnormal_speed else "✓"
                            print(f"Frame {frame_number:4d} | {track_type}_{player_id:2d} | "
                                  f"Speed: {smoothed_speed:5.1f} km/h ({speed:5.1f}) {status} | "
                                  f"Dist: {distance:4.2f}m | "
                                  f"Time: {time_diff:.4f}s ({frame_diff}f) | "
                                  f"FPS: {fps:.1f} | "
                                  f"Pos: ({current_position[0]:6.1f}, {current_position[1]:6.1f})")
                        
                        # 记录速度数据
                        self._record_speed(player_id, track_type, smoothed_speed, current_position, frame_number, fps)
                        
                        # 添加到当前帧的速度数据
                        frame_speed_data['players'][f"{track_type}_{player_id}"] = {
                            'type': track_type,
                            'id': player_id,
                            'speed': round(smoothed_speed, 2),
                            'position': current_position,
                            'club': track.get('club', 'Unknown')
                        }
                        
                        # 收集ROI数据
                        if original_frame is not None:
                            self._collect_player_roi(player_id, track_type, track, original_frame, frame_number)
                    else:
                        # If it's the first time we're seeing this player, set speed to 0
                        tracks[track_type][player_id]['speed'] = 0.0
                        # 初始化所有历史记录
                        self._init_player_history(player_id)
                    
                    # Update previous position
                    self.previous_positions[player_id] = (current_position, frame_number)
                else:
                    # If there's no projection, set speed to 0
                    tracks[track_type][player_id]['speed'] = 0.0
        
        # 将当前帧的速度数据添加到记录中
        if frame_speed_data['players']:
            self.speed_records.append(frame_speed_data)
        
        return tracks

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate the Euclidean distance between two positions.

        Args:
            pos1 (Tuple[float, float]): The first position (x, y).
            pos2 (Tuple[float, float]): The second position (x, y).

        Returns:
            float: The distance in meters.
        """
        dx = (pos2[0] - pos1[0]) * self.scale_x
        dy = (pos2[1] - pos1[1]) * self.scale_y
        return math.sqrt(dx**2 + dy**2)

    def _smooth_speed(self, player_id: Any, speed: float, position: Tuple[float, float] = None) -> float:
        """
         优化的速度平滑算法 - 平衡响应性和稳定性

        Args:
            player_id (Any): The identifier for the player.
            speed (float): The calculated speed to be smoothed.
            position (Tuple[float, float], optional): 当前位置，用于运动预测

        Returns:
            float: The smoothed speed value.
        """
        # 初始化历史记录
        if player_id not in self.speed_history:
            self._init_player_history(player_id)
        
        # 记录位置历史
        if position and player_id in self.position_history:
            self.position_history[player_id].append(position)
        
        #  静止状态检测 (优先处理)
        if position:
            is_stationary = self._detect_stationary_state(player_id, position, speed, self.frame_count)
            if is_stationary:
                # 应用渐进式衰减，避免突变
                decayed_speed = self._apply_gradual_decay(player_id, speed)
                self.speed_history[player_id].append(decayed_speed)
                return decayed_speed
        
        #  多层平滑策略
        smoothed_speed = self._apply_multi_layer_smoothing(player_id, speed)
        
        # 更新历史记录
        self.speed_history[player_id].append(smoothed_speed)
        
        return max(0.0, smoothed_speed)
    
    def _apply_gradual_decay(self, player_id: Any, current_speed: float) -> float:
        """
        🔄 应用渐进式衰减，避免速度突然跳变
        """
        if player_id in self.speed_history and len(self.speed_history[player_id]) > 0:
            last_speed = list(self.speed_history[player_id])[-1]
            
            # 渐进式衰减：如果上一帧速度已经很低，加速衰减
            if last_speed < 1.0:
                decay_factor = 0.5  # 快速衰减
            elif last_speed < 2.0:
                decay_factor = 0.6  # 中速衰减
            else:
                decay_factor = self.fast_decay_factor  # 正常衰减
            
            decayed_speed = last_speed * decay_factor
            
            # 如果衰减后的速度很低，直接设为0
            if decayed_speed < self.stationary_threshold * 0.5:
                decayed_speed = 0.0
                
            return decayed_speed
        else:
            return 0.0
    
    def _apply_multi_layer_smoothing(self, player_id: Any, speed: float) -> float:
        """
         多层平滑策略：异常值过滤 → 趋势平滑 → 加权平均
        """
        # 第1层：异常值检测和过滤
        filtered_speed = self._gentle_outlier_filter(player_id, speed)
        
        # 第2层：趋势平滑
        if self.trend_smoothing:
            trend_smoothed_speed = self._apply_trend_smoothing(player_id, filtered_speed)
        else:
            trend_smoothed_speed = filtered_speed
        
        # 第3层：自适应加权平均
        final_speed = self._apply_adaptive_averaging(player_id, trend_smoothed_speed)
        
        return final_speed
    
    def _gentle_outlier_filter(self, player_id: Any, speed: float) -> float:
        """
        🛡️ 温和的异常值过滤，避免过度矫正
        """
        history = list(self.speed_history[player_id])
        if len(history) < 3:
            return speed
        
        # 使用最近5个值计算统计量，更稳定
        recent_history = history[-5:] if len(history) >= 5 else history
        median_speed = np.median(recent_history)
        mad = np.median(np.abs(np.array(recent_history) - median_speed))  # 中位数绝对偏差
        
        # 使用MAD代替标准差，更稳健
        if mad > 0:
            outlier_threshold = median_speed + self.outlier_sensitivity * mad * 1.4826  # MAD转标准差系数
            if speed > outlier_threshold:
                # 温和修正：向中位数靠拢，而不是完全替换
                correction_factor = 0.7
                filtered_speed = speed * (1 - correction_factor) + median_speed * correction_factor
                return filtered_speed
        
        return speed
    
    def _apply_trend_smoothing(self, player_id: Any, speed: float) -> float:
        """
         趋势平滑：考虑速度变化的趋势
        """
        history = list(self.speed_history[player_id])
        if len(history) < 2:
            return speed
        
        # 计算趋势
        recent_history = history[-3:] if len(history) >= 3 else history
        if len(recent_history) >= 2:
            # 计算平均变化率
            changes = []
            for i in range(1, len(recent_history)):
                changes.append(recent_history[i] - recent_history[i-1])
            avg_change = np.mean(changes) if changes else 0
            
            # 预测值：基于趋势的预期速度
            predicted_speed = history[-1] + avg_change * 0.5  # 减弱趋势影响
            
            # 在测量值和预测值之间加权
            trend_weight = 0.3  # 趋势权重
            trend_smoothed = speed * (1 - trend_weight) + predicted_speed * trend_weight
            
            # 确保在合理范围内
            return max(0, min(trend_smoothed, self.max_speed))
        
        return speed
    
    def _apply_adaptive_averaging(self, player_id: Any, speed: float) -> float:
        """
        🎛️ 自适应加权平均：根据速度稳定性调整平滑强度
        """
        history = list(self.speed_history[player_id])
        if len(history) < 2:
            return speed
        
        # 使用更大的窗口进行平滑
        window_size = min(self.smooth_window_size, len(history) + 1)
        recent_speeds = history[-(window_size-1):] + [speed]
        
        # 计算速度稳定性
        if len(recent_speeds) >= 3:
            speed_variance = np.var(recent_speeds)
            speed_std = np.sqrt(speed_variance)
            
            # 根据稳定性调整权重分布
            if speed_std < 1.0:  # 很稳定
                # 使用均匀权重
                weights = np.ones(len(recent_speeds))
            elif speed_std < 3.0:  # 中等稳定
                # 更多权重给最近的值
                weights = np.linspace(0.5, 1.5, len(recent_speeds))
            else:  # 不稳定
                # 大部分权重给最近的值，但仍然平滑
                weights = np.exp(np.linspace(-1, 0, len(recent_speeds)))
        else:
            # 历史数据不足，使用线性权重
            weights = np.linspace(0.7, 1.3, len(recent_speeds))
        
        # 归一化权重
        weights = weights / np.sum(weights)
        
        # 加权平均
        weighted_speed = np.sum(np.array(recent_speeds) * weights)
        
        return weighted_speed

    def _init_kalman_filter(self) -> Dict:
        """
        初始化简单的卡尔曼滤波器参数
        """
        return {
            'x': 0.0,           # 状态 (速度)
            'P': 1.0,           # 估计误差协方差
            'Q': 0.1,           # 过程噪声协方差
            'R': 0.5,           # 测量噪声协方差
            'K': 0.0            # 卡尔曼增益
        }

    def _kalman_filter_speed(self, player_id: Any, measured_speed: float) -> float:
        """
        使用卡尔曼滤波器平滑速度
        """
        kf = self.kalman_filters[player_id]
        
        # 预测步骤
        # x_pred = x_prev (假设速度变化缓慢)
        # P_pred = P_prev + Q
        P_pred = kf['P'] + kf['Q']
        
        # 更新步骤
        # K = P_pred / (P_pred + R)
        kf['K'] = P_pred / (P_pred + kf['R'])
        
        # x = x_pred + K * (z - x_pred)
        kf['x'] = kf['x'] + kf['K'] * (measured_speed - kf['x'])
        
        # P = (1 - K) * P_pred
        kf['P'] = (1 - kf['K']) * P_pred
        
        return kf['x']

    def _detect_and_filter_outliers(self, player_id: Any, speed: float) -> float:
        """
        检测和过滤异常速度值
        """
        history = list(self.speed_history[player_id])
        if len(history) < 3:
            return speed
        
        # 计算历史速度的均值和标准差
        mean_speed = np.mean(history)
        std_speed = np.std(history)
        
        # 如果当前速度偏离均值超过阈值，则进行调整
        if std_speed > 0 and abs(speed - mean_speed) > self.outlier_sensitivity * std_speed:
            # 使用历史趋势预测合理速度
            if len(history) >= 2:
                # 简单线性预测
                trend = history[-1] - history[-2] if len(history) >= 2 else 0
                predicted_speed = history[-1] + trend * 0.5  # 减弱趋势影响
                
                # 在预测值和测量值之间加权
                weight = 0.7  # 更信任预测值
                filtered_speed = weight * predicted_speed + (1 - weight) * speed
                
                # 确保不超过合理范围
                filtered_speed = max(0, min(filtered_speed, self.max_speed))
                
                if self.verbose_logging:
                    print(f"[SpeedEstimator] 球员{player_id}异常速度过滤: "
                          f"{speed:.1f} -> {filtered_speed:.1f} km/h")
                
                return filtered_speed
        
        return speed

    def _adaptive_smooth(self, player_id: Any, speed: float) -> float:
        """
        自适应加权平滑算法
        """
        history = self.speed_history[player_id]
        history.append(speed)
        
        if len(history) < 2:
            return speed
        
        # 根据速度变化程度调整平滑强度
        recent_speeds = list(history)[-min(5, len(history)):]
        speed_variance = np.var(recent_speeds) if len(recent_speeds) > 1 else 0
        
        # 自适应权重：变化大时更多平滑，变化小时更少平滑
        if speed_variance > 25:  # 高变化
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # 更重视历史
        elif speed_variance > 10:  # 中等变化
            weights = np.array([0.15, 0.2, 0.25, 0.2, 0.2])  # 平衡
        else:  # 低变化
            weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])    # 均匀权重
        
        # 只使用可用的历史数据
        available_count = min(len(recent_speeds), len(weights))
        used_speeds = recent_speeds[-available_count:]
        used_weights = weights[-available_count:]
        used_weights = used_weights / np.sum(used_weights)  # 归一化
        
        smoothed_speed = np.sum([s * w for s, w in zip(used_speeds, used_weights)])
        
        return smoothed_speed

    def set_smoothing_parameters(self, adaptive_smoothing: bool = True, outlier_threshold: float = 3.0, 
                                kalman_q: float = 0.1, kalman_r: float = 0.5) -> None:
        """
        设置高级平滑参数
        
        Args:
            adaptive_smoothing: 是否启用自适应平滑
            outlier_threshold: 异常值检测阈值（标准差倍数）
            kalman_q: 卡尔曼滤波过程噪声
            kalman_r: 卡尔曼滤波测量噪声
        """
        self.adaptive_smoothing = adaptive_smoothing
        self.outlier_threshold = outlier_threshold
        
        # 更新所有现有的卡尔曼滤波器参数
        for player_id, kf in self.kalman_filters.items():
            kf['Q'] = kalman_q
            kf['R'] = kalman_r
        
        print(f"[SpeedEstimator] 平滑参数已更新:")
        print(f"  自适应平滑: {adaptive_smoothing}")
        print(f"  异常值阈值: {outlier_threshold}")
        print(f"  卡尔曼Q: {kalman_q}, R: {kalman_r}")

    def _collect_player_roi(self, player_id: Any, track_type: str, track: Dict[str, Any], frame: np.ndarray, frame_number: int) -> None:
        """
        收集球员的ROI图像数据
        
        Args:
            player_id: 球员ID
            track_type: 球员类型 (player/goalkeeper)  
            track: 球员跟踪数据
            frame: 原始帧图像
            frame_number: 帧号
        """
        player_key = f"{track_type}_{player_id}"
        
        # 检查是否需要在这一帧收集ROI
        if frame_number in self.roi_sample_frames or len(self.player_frames.get(player_key, [])) < 3:
            if 'bbox' in track:
                bbox = track['bbox']
                
                # 初始化球员数据
                if player_key not in self.player_frames:
                    self.player_frames[player_key] = []
                    self.player_bboxes[player_key] = []
                
                # 如果已经有3张图片，替换最旧的
                if len(self.player_frames[player_key]) >= 3:
                    self.player_frames[player_key].pop(0)
                    self.player_bboxes[player_key].pop(0)
                
                # 存储完整帧和bbox信息
                self.player_frames[player_key].append(frame.copy())
                self.player_bboxes[player_key].append(bbox.copy())
                
                if self.verbose_logging:
                    print(f"[SpeedEstimator] 收集{player_key}的ROI数据 (帧{frame_number})")

    def _extract_player_roi(self, frame: np.ndarray, bbox: List[float], margin: int = None) -> np.ndarray:
        """
        从帧中提取球员ROI
        
        Args:
            frame: 原始帧图像
            bbox: 边界框 [x1, y1, x2, y2]
            margin: 边距（像素）
            
        Returns:
            np.ndarray: ROI图像
        """
        if margin is None:
            margin = self.roi_margin
            
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 添加边距并确保在图像范围内
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # 提取ROI
        roi = frame[y1:y2, x1:x2]
        
        # 确保ROI不为空
        if roi.size == 0:
            # 返回一个小的默认图像
            roi = np.zeros((50, 50, 3), dtype=np.uint8)
        
        return roi

    def save_player_rois_and_curves(self, output_dir: str = "output_videos") -> None:
        """
        保存所有球员的ROI图像和速度曲线到各自的文件夹中
        
        Args:
            output_dir: 输出目录
        """
        if not self.player_frames:
            print("没有收集到球员ROI数据")
            return
        
        print(f"正在保存 {len(self.player_frames)} 个球员的ROI和速度曲线...")
        
        for player_key in self.player_frames:
            player_dir = os.path.join(output_dir, player_key)
            os.makedirs(player_dir, exist_ok=True)
            
            # 保存ROI图像
            self._save_player_roi_images(player_key, player_dir)
            
            # 保存速度曲线到球员文件夹
            self._save_player_speed_curve(player_key, player_dir)
        
        print(f" 所有球员数据已保存到各自的文件夹中！")

    def _save_player_roi_images(self, player_key: str, player_dir: str) -> None:
        """
        保存单个球员的ROI图像
        """
        frames = self.player_frames[player_key]
        bboxes = self.player_bboxes[player_key]
        
        for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
            # 提取ROI
            roi = self._extract_player_roi(frame, bbox)
            
            # 保存ROI图像
            roi_filename = f"roi_{i+1}.png"
            roi_path = os.path.join(player_dir, roi_filename)
            cv2.imwrite(roi_path, roi)
            
            # 保存带标注的原图区域（可选）
            annotated_frame = frame.copy()
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, player_key, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            annotated_filename = f"annotated_{i+1}.png"
            annotated_path = os.path.join(player_dir, annotated_filename)
            cv2.imwrite(annotated_path, annotated_frame)
        
        print(f" {player_key}: 保存了 {len(frames)} 张ROI图像")

    def _save_player_speed_curve(self, player_key: str, player_dir: str) -> None:
        """
        为单个球员保存速度曲线
        """
        if not self.speed_records:
            return
        
        # 提取该球员的速度数据
        player_speeds = []
        for record in self.speed_records:
            if player_key in record['players']:
                player_info = record['players'][player_key]
                player_speeds.append({
                    'frame': record['frame'],
                    'timestamp': record['timestamp'],
                    'speed': player_info['speed'],
                    'club': player_info.get('club', 'Unknown')
                })
        
        if len(player_speeds) < 2:
            print(f" {player_key}: 速度数据不足，跳过曲线生成")
            return
        
        # 绘制速度曲线
        self._plot_single_player_curve(player_key, player_speeds, player_dir)

    def _plot_single_player_curve(self, player_key: str, speeds_data: List[Dict], save_dir: str) -> None:
        """
        绘制单个球员的速度曲线
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 提取数据
        timestamps = [d['timestamp'] for d in speeds_data]
        speeds = [d['speed'] for d in speeds_data]
        club = speeds_data[0]['club']
        
        # 根据俱乐部设置颜色
        if club == 'Club1':
            color = '#FF6B6B'  # 红色
            club_name = '红队'
        elif club == 'Club2':
            color = '#4ECDC4'  # 蓝绿色
            club_name = '蓝队'
        else:
            color = '#95A5A6'  # 灰色
            club_name = '其他'
        
        # 绘制速度曲线
        ax.plot(timestamps, speeds, color=color, linewidth=3, marker='o', 
               markersize=4, alpha=0.9, label='速度曲线')
        ax.fill_between(timestamps, speeds, alpha=0.3, color=color)
        
        # 标记高速时刻（>30 km/h）
        high_speed_indices = [i for i, s in enumerate(speeds) if s > 30]
        if high_speed_indices:
            high_timestamps = [timestamps[i] for i in high_speed_indices]
            high_speeds = [speeds[i] for i in high_speed_indices]
            ax.scatter(high_timestamps, high_speeds, color='red', s=50, 
                      alpha=0.8, zorder=5, label='高速冲刺 (>30 km/h)')
        
        # 计算统计数据
        avg_speed = np.mean(speeds)
        max_speed = max(speeds)
        min_speed = min(speeds)
        
        # 添加统计线
        ax.axhline(y=avg_speed, color='orange', linestyle='--', alpha=0.8, 
                  linewidth=2, label=f'平均速度: {avg_speed:.1f} km/h')
        ax.axhline(y=35, color='red', linestyle='--', alpha=0.6, 
                  linewidth=2, label='最大速度限制: 35 km/h')
        
        # 设置标题和标签
        ax.set_title(f'{player_key} 速度变化曲线\n'
                    f'{club_name} | 最高: {max_speed:.1f} km/h | '
                    f'平均: {avg_speed:.1f} km/h | 最低: {min_speed:.1f} km/h', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlabel('比赛时间 (秒)', fontsize=12)
        ax.set_ylabel('速度 (km/h)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 40)
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=10)
        
        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # 添加统计信息文本框
        stats_text = f'数据点数: {len(speeds)}\n高速冲刺次数: {len(high_speed_indices)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"speed_curve.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f" {player_key}: 速度曲线已保存")
        
        # 清理内存
        plt.close()

    def _record_speed(self, player_id: Any, track_type: str, speed: float, position: Tuple[float, float], frame_number: int, fps: float) -> None:
        """
        记录球员的速度数据用于统计分析
        
        Args:
            player_id: 球员ID
            track_type: 球员类型 (player/goalkeeper)
            speed: 当前速度 (km/h)
            position: 当前位置
            frame_number: 帧号
            fps: 帧率
        """
        # 更新最高速度记录
        key = f"{track_type}_{player_id}"
        if key not in self.player_max_speeds or speed > self.player_max_speeds[key]:
            self.player_max_speeds[key] = speed
        
        # 记录速度历史用于计算平均值
        if key not in self.player_avg_speeds:
            self.player_avg_speeds[key] = []
        self.player_avg_speeds[key].append(speed)

    def save_speed_analysis(self, output_dir: str = "output_videos", filename: str = None) -> str:
        """
        保存速度分析结果到文件
        
        Args:
            output_dir: 输出目录
            filename: 文件名（可选）
            
        Returns:
            str: 保存的文件路径
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"speed_analysis_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        # 计算统计数据
        analysis_data = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "total_frames": len(self.speed_records),
                "total_players": len(self.player_max_speeds)
            },
            "player_statistics": {},
            "frame_by_frame_data": self.speed_records
        }
        
        # 为每个球员计算统计数据
        for player_key, speeds in self.player_avg_speeds.items():
            if speeds:  # 确保有速度数据
                analysis_data["player_statistics"][player_key] = {
                    "max_speed": round(self.player_max_speeds.get(player_key, 0), 2),
                    "avg_speed": round(sum(speeds) / len(speeds), 2),
                    "min_speed": round(min(speeds), 2),
                    "speed_count": len(speeds),
                    "speeds_above_20": len([s for s in speeds if s > 20]),
                    "speeds_above_25": len([s for s in speeds if s > 25]),
                    "speeds_above_30": len([s for s in speeds if s > 30])
                }
        
        # 转换NumPy数据类型并保存到JSON文件
        analysis_data_converted = convert_numpy_types(analysis_data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_data_converted, f, indent=2, ensure_ascii=False)
        
        print(f"速度分析结果已保存到: {filepath}")
        return filepath

    def print_speed_summary(self) -> None:
        """
        打印速度统计摘要
        """
        print("\n=== 球员速度统计摘要 ===")
        print(f"总帧数: {len(self.speed_records)}")
        print(f"跟踪球员数: {len(self.player_max_speeds)}")
        print("\n球员速度排行:")
        
        # 按最高速度排序
        sorted_players = sorted(self.player_max_speeds.items(), key=lambda x: x[1], reverse=True)
        
        for i, (player_key, max_speed) in enumerate(sorted_players[:10], 1):  # 显示前10名
            avg_speed = 0
            if player_key in self.player_avg_speeds and self.player_avg_speeds[player_key]:
                avg_speed = sum(self.player_avg_speeds[player_key]) / len(self.player_avg_speeds[player_key])
            
            print(f"{i:2d}. {player_key:15s} - 最高: {max_speed:5.1f} km/h, 平均: {avg_speed:5.1f} km/h")

    def reset(self) -> None:
        """
        Reset the previous positions and speed history. 
        Call this at the start of a new video or when needed.
        """
        self.previous_positions = {}
        self.speed_history = {}
        self.position_history = {}
        self.velocity_history = {}
        self.kalman_filters = {}
        self.speed_records = []
        self.player_max_speeds = {}
        self.player_avg_speeds = {}
        self.player_frames = {}
        self.player_bboxes = {}
        self.player_states = {}  # 重置球员状态
        self.current_fps = None
        self.frame_count = 0

    def check_coordinate_range(self, tracks: Dict[str, Any]) -> None:
        """
        检查坐标范围，用于诊断问题
        """
        if self.frame_count == 1:  # 只在第一帧检查
            all_positions = []
            for track_type in tracks:
                for player_id, track in tracks[track_type].items():
                    if 'projection' in track:
                        pos = track['projection']
                        all_positions.append(pos)
            
            if all_positions:
                x_coords = [pos[0] for pos in all_positions]
                y_coords = [pos[1] for pos in all_positions]
                
                print(f"[SpeedEstimator] 坐标范围检查:")
                print(f"  X坐标范围: {min(x_coords):.1f} ~ {max(x_coords):.1f}")
                print(f"  Y坐标范围: {min(y_coords):.1f} ~ {max(y_coords):.1f}")
                print(f"  预期场地范围: 0 ~ {self.field_width} (X), 0 ~ {self.field_height} (Y)")
                
                # 检查坐标是否超出预期范围
                if max(x_coords) > self.field_width * 2 or max(y_coords) > self.field_height * 2:
                    print(f"   警告: 坐标范围超出预期，可能影响速度计算准确性")

    def plot_speed_curves(self, save_dir: str = "output_videos") -> None:
        """
        为每个球员单独绘制速度变化曲线图
        """
        if not self.speed_records:
            print("没有速度数据可以绘制")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取每个球员的速度数据
        from collections import defaultdict
        player_data = defaultdict(list)
        
        for record in self.speed_records:
            frame = record['frame']
            timestamp = record['timestamp']
            for player_key, player_info in record['players'].items():
                player_id = f"{player_info['type']}_{player_info['id']}"
                speed = player_info['speed']
                club = player_info.get('club', 'Unknown')
                
                player_data[player_id].append({
                    'frame': frame,
                    'timestamp': timestamp,
                    'speed': speed,
                    'club': club
                })
        
        # 过滤出有足够数据的球员（至少5个数据点）
        valid_players = {k: v for k, v in player_data.items() if len(v) >= 5}
        
        if not valid_players:
            print("没有足够的数据来绘制速度曲线")
            return
        
        print(f"正在为 {len(valid_players)} 个球员分别绘制速度曲线...")
        
        # 为每个球员单独创建图片
        for player_id, speeds_data in valid_players.items():
            # 创建单独的图形
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # 提取数据
            timestamps = [d['timestamp'] for d in speeds_data]
            speeds = [d['speed'] for d in speeds_data]
            club = speeds_data[0]['club']
            
            # 根据俱乐部设置颜色
            if club == 'Club1':
                color = '#FF6B6B'  # 红色
                club_name = '红队'
            elif club == 'Club2':
                color = '#4ECDC4'  # 蓝绿色
                club_name = '蓝队'
            else:
                color = '#95A5A6'  # 灰色
                club_name = '其他'
            
            # 绘制速度曲线
            ax.plot(timestamps, speeds, color=color, linewidth=3, marker='o', 
                   markersize=4, alpha=0.9, label='速度曲线')
            ax.fill_between(timestamps, speeds, alpha=0.3, color=color)
            
            # 标记高速时刻（>30 km/h）
            high_speed_indices = [i for i, s in enumerate(speeds) if s > 30]
            if high_speed_indices:
                high_timestamps = [timestamps[i] for i in high_speed_indices]
                high_speeds = [speeds[i] for i in high_speed_indices]
                ax.scatter(high_timestamps, high_speeds, color='red', s=50, 
                          alpha=0.8, zorder=5, label='高速冲刺 (>30 km/h)')
            
            # 计算统计数据
            avg_speed = np.mean(speeds)
            max_speed = max(speeds)
            min_speed = min(speeds)
            
            # 添加统计线
            ax.axhline(y=avg_speed, color='orange', linestyle='--', alpha=0.8, 
                      linewidth=2, label=f'平均速度: {avg_speed:.1f} km/h')
            ax.axhline(y=35, color='red', linestyle='--', alpha=0.6, 
                      linewidth=2, label='最大速度限制: 35 km/h')
            
            # 设置标题和标签
            ax.set_title(f'{player_id} 速度变化曲线\n'
                        f'{club_name} | 最高: {max_speed:.1f} km/h | '
                        f'平均: {avg_speed:.1f} km/h | 最低: {min_speed:.1f} km/h', 
                        fontsize=14, fontweight='bold', pad=20)
            
            ax.set_xlabel('比赛时间 (秒)', fontsize=12)
            ax.set_ylabel('速度 (km/h)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 40)
            
            # 添加图例
            ax.legend(loc='upper right', fontsize=10)
            
            # 设置刻度字体大小
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # 添加统计信息文本框
            stats_text = f'数据点数: {len(speeds)}\n高速冲刺次数: {len(high_speed_indices)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存单独的图片
            safe_player_id = player_id.replace('/', '_').replace('\\', '_')  # 处理文件名中的特殊字符
            filename = f"speed_curves_{safe_player_id}.png"
            filepath = os.path.join(save_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" {player_id} 速度曲线已保存: {filename}")
            
            # 清理内存
            plt.close()
        
        print(f"\n 所有 {len(valid_players)} 个球员的速度曲线图已生成完成！")
    
    def _detect_stationary_state(self, player_id: Any, current_position: Tuple[float, float], 
                               current_speed: float, frame_number: int) -> bool:
        """
         检测球员是否处于静止状态
        
        Args:
            player_id: 球员ID
            current_position: 当前位置
            current_speed: 当前计算的速度
            frame_number: 当前帧号
            
        Returns:
            bool: 是否静止
        """
        if player_id not in self.player_states:
            return False
        
        state = self.player_states[player_id]
        state['recent_positions'].append(current_position)
        state['recent_speeds'].append(current_speed)
        
        # 1. 速度检查：速度低于静止阈值
        speed_is_low = current_speed < self.stationary_threshold
        
        # 2. 位置稳定性检查
        position_is_stable = False
        if len(state['recent_positions']) >= 2:
            position_is_stable = self._calculate_position_stability(player_id, state['recent_positions'])
        
        # 3. 更新静止计数
        if speed_is_low and position_is_stable:
            state['stationary_count'] += 1
        else:
            state['stationary_count'] = 0
            state['last_movement_frame'] = frame_number
        
        # 4. 判定是否静止
        is_stationary = state['stationary_count'] >= self.stationary_frames_required
        state['is_stationary'] = is_stationary
        
        if self.verbose_logging and is_stationary and state['stationary_count'] == self.stationary_frames_required:
            print(f"[SpeedEstimator] 球员 {player_id} 检测为静止状态")
        
        return is_stationary
    
    def _calculate_position_stability(self, player_id: Any, positions: deque) -> bool:
        """
         计算位置稳定性
        
        Args:
            player_id: 球员ID
            positions: 最近的位置历史
            
        Returns:
            bool: 位置是否稳定
        """
        if len(positions) < 2:
            return True
        
        # 计算最近几个位置的移动距离
        distances = []
        for i in range(1, len(positions)):
            pos1, pos2 = positions[i-1], positions[i]
            # 使用当前的比例尺计算实际距离
            dx = (pos2[0] - pos1[0]) * self.scale_x
            dy = (pos2[1] - pos1[1]) * self.scale_y
            distance = math.sqrt(dx**2 + dy**2)
            distances.append(distance)
        
        # 如果平均移动距离小于噪声阈值，认为位置稳定
        avg_movement = sum(distances) / len(distances) if distances else 0
        return avg_movement < self.movement_noise_threshold
    
    def _apply_fast_decay(self, player_id: Any, current_speed: float) -> float:
        """
         应用快速衰减，让球员停止时速度快速降为0
        
        Args:
            player_id: 球员ID
            current_speed: 当前速度
            
        Returns:
            float: 衰减后的速度
        """
        if player_id in self.speed_history and len(self.speed_history[player_id]) > 0:
            last_speed = list(self.speed_history[player_id])[-1]
            # 应用快速衰减
            decayed_speed = last_speed * self.fast_decay_factor
            
            # 如果衰减后的速度很低，直接设为0
            if decayed_speed < self.stationary_threshold:
                decayed_speed = 0.0
                
            return decayed_speed
        else:
            return 0.0
    
    def _apply_quick_response(self, player_id: Any, current_speed: float) -> float:
        """
         优化的快速响应机制 - 更平滑的响应
        
        Args:
            player_id: 球员ID
            current_speed: 当前速度
            
        Returns:
            float: 快速响应后的速度
        """
        if not self.quick_response_mode or player_id not in self.speed_history:
            return current_speed
        
        recent_speeds = list(self.speed_history[player_id])
        if len(recent_speeds) < 3:  # 需要更多历史数据
            return current_speed
        
        # 计算速度变化的稳定性
        recent_window = recent_speeds[-3:]
        speed_changes = []
        for i in range(1, len(recent_window)):
            change = abs(recent_window[i] - recent_window[i-1])
            speed_changes.append(change)
        
        current_change = abs(current_speed - recent_speeds[-1])
        avg_change = np.mean(speed_changes) if speed_changes else 0
        std_change = np.std(speed_changes) if len(speed_changes) > 1 else 0
        
        # 更保守的快速响应条件
        significant_change_threshold = avg_change + 1.5 * std_change if std_change > 0 else avg_change * 2.0
        
        if current_change > significant_change_threshold and current_change > 1.0:
            # 应用温和的快速响应
            response_intensity = min(0.4, current_change / (significant_change_threshold + 1.0))
            quick_response_speed = current_speed * response_intensity + recent_speeds[-1] * (1 - response_intensity)
            
            if self.verbose_logging:
                print(f"[SpeedEstimator] 球员{player_id}快速响应: {current_speed:.1f} -> {quick_response_speed:.1f}")
            
            return quick_response_speed
        
        return current_speed
    
    def set_responsiveness_parameters(self, stationary_threshold: float = 0.8,
                                    low_speed_threshold: float = 3.0,
                                    fast_decay_factor: float = 0.7,
                                    quick_response_mode: bool = True,
                                    smooth_window_size: int = 7,
                                    outlier_sensitivity: float = 2.5,
                                    trend_smoothing: bool = True) -> None:
        """
        🎛️ 设置响应性和平滑参数
        
        Args:
            stationary_threshold: 静止速度阈值 (km/h)
            low_speed_threshold: 低速阈值 (km/h)
            fast_decay_factor: 快速衰减因子 (0-1)
            quick_response_mode: 是否启用快速响应模式
            smooth_window_size: 平滑窗口大小
            outlier_sensitivity: 异常值检测敏感度 (标准差倍数)
            trend_smoothing: 是否启用趋势平滑
        """
        self.stationary_threshold = stationary_threshold
        self.low_speed_threshold = low_speed_threshold
        self.fast_decay_factor = fast_decay_factor
        self.quick_response_mode = quick_response_mode
        self.smooth_window_size = smooth_window_size
        self.outlier_sensitivity = outlier_sensitivity
        self.trend_smoothing = trend_smoothing
        
        print(f"[SpeedEstimator] 速度计算参数已更新:")
        print(f"  静止阈值: {stationary_threshold} km/h")
        print(f"  低速阈值: {low_speed_threshold} km/h")
        print(f"  快速衰减: {fast_decay_factor}")
        print(f"  快速响应: {quick_response_mode}")
        print(f"  平滑窗口: {smooth_window_size} 帧")
        print(f"  异常值敏感度: {outlier_sensitivity}")
        print(f"  趋势平滑: {trend_smoothing}")
