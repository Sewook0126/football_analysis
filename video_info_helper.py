"""
视频信息获取辅助函数
用于动态获取视频的实际尺寸信息
"""

import cv2
from typing import Tuple, Optional

def get_video_info(video_path: str) -> Optional[Tuple[int, int, float]]:
    """
    获取视频的基本信息
    
    Args:
        video_path (str): 视频文件路径
        
    Returns:
        Optional[Tuple[int, int, float]]: (width, height, fps) 或 None
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f" 无法打开视频文件: {video_path}")
        return None
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f" 视频信息:")
        print(f"   文件: {video_path}")
        print(f"   尺寸: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        
        return width, height, fps
        
    except Exception as e:
        print(f" 获取视频信息失败: {e}")
        return None
        
    finally:
        cap.release()

def get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """
    仅获取视频尺寸 (简化版本)
    
    Args:
        video_path (str): 视频文件路径
        
    Returns:
        Tuple[int, int]: (width, height)，失败时返回默认值 (1920, 1080)
    """
    info = get_video_info(video_path)
    if info:
        return info[0], info[1]  # width, height
    else:
        print("  使用默认视频尺寸: 1920x1080")
        return 1920, 1080

def calculate_optimal_input_size(video_width: int, video_height: int, 
                                target_sizes: list = [320, 480, 640, 800, 960, 1280]) -> int:
    """
    根据视频尺寸推荐最优的推理输入尺寸
    
    Args:
        video_width (int): 视频宽度
        video_height (int): 视频高度
        target_sizes (list): 可选的推理尺寸列表
        
    Returns:
        int: 推荐的推理尺寸
    """
    video_max_dim = max(video_width, video_height)
    
    # 推荐规则: 推理尺寸为视频最大尺寸的 1/2 到 1/3
    recommended_range = (video_max_dim // 3, video_max_dim // 2)
    
    # 找到最接近推荐范围的尺寸
    best_size = target_sizes[2]  # 默认640
    min_diff = float('inf')
    
    for size in target_sizes:
        # 计算与推荐范围中点的差距
        range_center = sum(recommended_range) // 2
        diff = abs(size - range_center)
        
        if diff < min_diff:
            min_diff = diff
            best_size = size
    
    print(f" 推理尺寸推荐:")
    print(f"   视频尺寸: {video_width}x{video_height}")
    print(f"   推荐范围: {recommended_range[0]}-{recommended_range[1]}")
    print(f"   建议使用: {best_size}x{best_size}")
    
    return best_size

def create_dynamic_config(video_path: str, custom_input_size: Optional[int] = None) -> dict:
    """
    基于视频创建动态配置
    
    Args:
        video_path (str): 视频文件路径
        custom_input_size (Optional[int]): 自定义推理尺寸，None时自动推荐
        
    Returns:
        dict: 配置字典
    """
    # 获取视频信息
    video_info = get_video_info(video_path)
    if not video_info:
        # 使用默认配置
        return {
            "original_width": 1920,
            "original_height": 1080,
            "input_size": custom_input_size or 640,
            "fps": 25.0,
            "scale_x": 1920 / (custom_input_size or 640),
            "scale_y": 1080 / (custom_input_size or 640)
        }
    
    width, height, fps = video_info
    
    # 确定推理尺寸
    if custom_input_size:
        input_size = custom_input_size
        print(f" 使用自定义推理尺寸: {input_size}")
    else:
        input_size = calculate_optimal_input_size(width, height)
        print(f" 使用推荐推理尺寸: {input_size}")
    
    # 计算缩放因子
    scale_x = width / input_size
    scale_y = height / input_size
    
    config = {
        "original_width": width,
        "original_height": height,
        "input_size": input_size,
        "fps": fps,
        "scale_x": scale_x,
        "scale_y": scale_y
    }
    
    print(f" 动态配置生成:")
    print(f"   原始尺寸: {width}x{height}")
    print(f"   推理尺寸: {input_size}x{input_size}")
    print(f"   缩放因子: X={scale_x:.3f}, Y={scale_y:.3f}")
    print(f"   帧率: {fps:.1f} FPS")
    
    return config

# 使用示例
if __name__ == "__main__":
    import sys
    
    # 测试视频信息获取
    test_video = "input_videos/liverpoor vs norwich city (fourth round goal3)9(2).mp4"
    
    print(" 视频信息获取测试")
    print("=" * 50)
    
    # 获取视频信息
    info = get_video_info(test_video)
    if info:
        width, height, fps = info
        print(f" 成功获取视频信息")
        
        # 推荐推理尺寸
        optimal_size = calculate_optimal_input_size(width, height)
        print(f" 推荐推理尺寸: {optimal_size}")
        
        # 创建动态配置
        config = create_dynamic_config(test_video)
        print(f" 动态配置创建完成")
        
    else:
        print(" 视频信息获取失败")
    
    print("\n 使用方法:")
    print("from video_info_helper import create_dynamic_config")
    print("config = create_dynamic_config('your_video.mp4')")
    print("MODEL_INPUT_SIZE = config['input_size']")
    print("ORIGINAL_WIDTH = config['original_width']")
    print("ORIGINAL_HEIGHT = config['original_height']")
