from .abstract_annotator import AbstractAnnotator

import cv2
import numpy as np
from typing import Dict

class KeypointsAnnotator(AbstractAnnotator):
    """Annotates frames with keypoints, drawing points at the keypoints' locations."""

    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        """
        Annotates the frame with keypoints.

        Args:
            frame (np.ndarray): The current frame to be annotated.
            tracks (Dict): A dictionary containing keypoints, where the key is 
                           the keypoint ID and the value is a tuple (x, y) of coordinates.
        
        Returns:
            np.ndarray: The frame with keypoints annotated on it.
        """
         
        frame = frame.copy()
        
        # 调试信息：确认收到的关键点数量
        if len(tracks) == 0:
            # 如果没有关键点，在左上角显示警告
            cv2.putText(frame, "No keypoints detected", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 显示关键点数量
            cv2.putText(frame, f"Keypoints: {len(tracks)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for kp_id, (x, y) in tracks.items():
            # Draw a larger circle for better visibility (radius 8 instead of 5)
            # Outer circle (white outline for better visibility)
            cv2.circle(frame, (int(x), int(y)), radius=8, color=(255, 255, 255), thickness=2)
            # Inner circle (green fill)
            cv2.circle(frame, (int(x), int(y)), radius=6, color=(0, 255, 0), thickness=-1)
            
            # 不显示关键点ID数字，避免因视角变动导致的混乱
            # 只保留绿色圆圈标记即可

        return frame
