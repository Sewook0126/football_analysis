from tracking.abstract_tracker import AbstractTracker

import supervision as sv
import cv2
from typing import List
import numpy as np
from ultralytics.engine.results import Results

class ObjectTracker(AbstractTracker):

    def __init__(self, model_path: str, conf: float = 0.5, ball_conf: float = 0.3, 
                 input_size: int = 640, original_size: tuple = (1920, 1080)) -> None:
        """
        Initialize ObjectTracker with detection and tracking.

        Args:
            model_path (str): Model Path.
            conf (float): Confidence threshold for detection.
            ball_conf (float): Ball detection confidence threshold.
            input_size (int): Model input size (e.g., 640, 1280).
            original_size (tuple): Original video size (width, height).
        """
        super().__init__(model_path, conf)  # Call the Tracker base class constructor

        self.ball_conf = ball_conf
        self.classes = ['ball', 'goalkeeper', 'player', 'referee']
        # 调整ByteTracker参数提高跟踪稳定性
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.6,      # 提高跟踪激活阈值 (默认0.25)
            lost_track_buffer=15,                # 增加丢失轨迹缓冲 (默认30，设为15)
            minimum_matching_threshold=0.8,      # 提高最小匹配阈值 (默认0.8，保持)
            frame_rate=25,                       # 设置帧率 (默认30，设为25)
            minimum_consecutive_frames=3         # 增加最小连续帧数 (默认1，设为3)
        )
        self.tracker.reset()
        self.all_tracks = {class_name: {} for class_name in self.classes}  # Initialize tracks
        self.cur_frame = 0  # Frame counter initialization
        
        # 使用传入的配置参数
        self.input_size = input_size
        self.original_size = original_size
        self.scale_x = original_size[0] / input_size  # 缩放因子X
        self.scale_y = original_size[1] / input_size  # 缩放因子Y
        
        print(f" ObjectTracker配置: {input_size}x{input_size} → {original_size[0]}x{original_size[1]}")
        print(f"   缩放因子: X={self.scale_x:.3f}, Y={self.scale_y:.3f}")

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        """
        Perform object detection on multiple frames.

        Args:
            frames (List[np.ndarray]): List of frames to perform object detection on.

        Returns:
            List[Results]: Detection results for each frame.
        """
        # Preprocess: Resize frames to configured input size
        resized_frames = [self._preprocess_frame(frame) for frame in frames]

        # Use YOLOv8's predict method to handle batch inference
        # 显式指定输入尺寸以确保使用正确的推理尺寸
        detections = self.model.predict(resized_frames, conf=self.conf, imgsz=self.input_size, verbose=True)

        return detections  # Batch of detections

    def track(self, detection: Results) -> dict:
        """
        Perform object tracking on detection.

        Args:
            detection (Results): Detected objects for a single frame.

        Returns:
            dict: Dictionary containing tracks of the frame.
        """
        # Convert Ultralytics detections to supervision
        detection_sv = sv.Detections.from_ultralytics(detection)

        # Perform ByteTracker object tracking on the detections
        tracks = self.tracker.update_with_detections(detection_sv)

        self.current_frame_tracks = self._tracks_mapper(tracks, self.classes)
        
        # Store the current frame's tracking information in all_tracks
        self.all_tracks[self.cur_frame] = self.current_frame_tracks.copy()

        # Increment the current frame counter
        self.cur_frame += 1

        # Return only the last frame's data
        return self.current_frame_tracks
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame by resizing it to the configured input size.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            np.ndarray: The resized frame.
        """
        # Resize the frame to configured input size
        resized_frame = cv2.resize(frame, (self.input_size, self.input_size))
        return resized_frame
    
    def _tracks_mapper(self, tracks: sv.Detections, class_names: List[str]) -> dict:
        """
        Maps tracks to a dictionary by class and tracker ID. Also, adjusts bounding boxes to 1920x1080 resolution.

        Args:
            tracks (sv.Detections): Tracks from the frame.
            class_names (List[str]): List of class names.

        Returns:
            dict: Mapped detections for the frame.
        """
        # Initialize the dictionary
        result = {class_name: {} for class_name in class_names}

        # Extract relevant data from tracks
        xyxy = tracks.xyxy  # Bounding boxes
        class_ids = tracks.class_id  # Class IDs
        tracker_ids = tracks.tracker_id  # Tracker IDs
        confs = tracks.confidence

        # Iterate over all tracks
        for bbox, class_id, track_id, conf in zip(xyxy, class_ids, tracker_ids, confs):
            class_name = class_names[class_id]

            # Skip balls with confidence lower than ball_conf
            if class_name == "ball" and conf < self.ball_conf:
                continue  # Skip low-confidence ball detections

            # Create class_name entry if not already present
            if class_name not in result:
                result[class_name] = {}

            # Scale the bounding box back to the original resolution (1920x1080)
            scaled_bbox = [
                bbox[0] * self.scale_x,  # x1
                bbox[1] * self.scale_y,  # y1
                bbox[2] * self.scale_x,  # x2
                bbox[3] * self.scale_y   # y2
            ]

            # Add track_id entry if not already present
            if track_id not in result[class_name]:
                result[class_name][track_id] = {'bbox': scaled_bbox}

        return result
