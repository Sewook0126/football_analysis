'''
Descripttion: 
version: 
Author: akabulaka
Date: 2025-09-19 14:17:19
LastEditors: akabulaka
LastEditTime: 2025-10-07 12:23:44
'''
from utils.video_utils_simple import process_video
from tracking import ObjectTracker, KeypointsTracker, PointDetectTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
from video_info_helper import create_dynamic_config

import numpy as np
import os

def main():
    """
    Main function to demonstrate how to use the football analysis project.
    This script will walk you through loading models, assigning clubs, tracking objects and players, and processing the video.
    """
    
    # ===========================================
    # 推理配置 - 在这里修改推理尺寸和视频路径
    # ===========================================
    # 视频文件路径
    VIDEO_PATH = 'input_videos/wigan athletic 1-0 man  복사본.mp4'
    
    # 推理尺寸设置 (可选: 320, 480, 640, 800, 960, 1280, 或 None 自动推荐)
    CUSTOM_INPUT_SIZE = 640      # 设置为 None 可自动根据视频尺寸推荐最优值
    
    # ===========================================
    # 关键点检测配置
    # ===========================================
    # 是否使用point_detect模型替代关键点检测模型
    # True: 使用runs/point_detect/train/weights/best.pt (默认，推荐)
    # False: 使用传统关键点检测模型 models/weights/keypoints-detection.pt
    USE_POINT_DETECT_MODEL = True
    POINT_DETECT_MODEL_PATH = 'runs/point_detect/train/weights/best.pt'
    POINT_DETECT_CONF = 0.3  # Point detection confidence threshold (降低以检测更多关键点)
    
    # ===========================================
    # 速度表格生成配置
    # ===========================================
    # 是否生成速度表格
    GENERATE_SPEED_TABLES = True
    # 速度表格时间间隔（秒）- 可选0.1, 0.5, 1.0等
    SPEED_TABLE_INTERVAL = 0.5  # 0.5秒一个时间点
    # 表格格式：'csv' 或 'excel'
    SPEED_TABLE_FORMAT = 'excel'  # 'csv' 或 'excel'
    # 表格语言：'chinese' 或 'english'
    SPEED_TABLE_LANGUAGE = 'english'  # 'chinese' 或 'english'
    
    # 动态获取视频配置
    print("正在分析视频并生成配置...")
    print("=" * 50)
    
    video_config = create_dynamic_config(VIDEO_PATH, CUSTOM_INPUT_SIZE)
    
    # 从配置中提取参数
    MODEL_INPUT_SIZE = video_config['input_size']
    ORIGINAL_WIDTH = video_config['original_width']
    ORIGINAL_HEIGHT = video_config['original_height']
    VIDEO_FPS = video_config['fps']
    SCALE_X = video_config['scale_x']
    SCALE_Y = video_config['scale_y']
    
    print(f"配置完成:")
    print(f"   视频文件: {VIDEO_PATH}")
    print(f"   原始尺寸: {ORIGINAL_WIDTH}x{ORIGINAL_HEIGHT}")
    print(f"   推理尺寸: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
    print(f"   缩放因子: X={SCALE_X:.3f}, Y={SCALE_Y:.3f}")
    print(f"   预期性能提升: {(1280/MODEL_INPUT_SIZE)**2:.1f}x")
    print("=" * 50)

    # 1. Load the object detection model
    # Adjust the 'conf' value as per your requirements.
    obj_tracker = ObjectTracker(
        model_path='models/weights/object-detection.pt',    # Object Detection Model Weights Path
        conf=.7,                                            # Object Detection confidence threshold (提高到0.7)
        ball_conf=.3,                                       # Ball Detection confidence threshold (提高到0.3)
        input_size=MODEL_INPUT_SIZE,                        # 传入推理尺寸
        original_size=(ORIGINAL_WIDTH, ORIGINAL_HEIGHT)     # 传入原始尺寸
    )

    # 2. Load the keypoints detection model OR point detection model
    # 根据配置选择使用哪个模型
    if USE_POINT_DETECT_MODEL:
        print("=" * 50)
        print("使用 Point Detection 模型 (推荐)")
        print(f"模型路径: {POINT_DETECT_MODEL_PATH}")
        print("=" * 50)
        
        # 检查模型文件是否存在
        if not os.path.exists(POINT_DETECT_MODEL_PATH):
            print(f"⚠️  警告: Point Detection 模型未找到: {POINT_DETECT_MODEL_PATH}")
            print("   切换到传统关键点检测模型...")
            USE_POINT_DETECT_MODEL = False
        else:
            kp_tracker = PointDetectTracker(
                model_path=POINT_DETECT_MODEL_PATH,             # Point Detection Model Path
                conf=POINT_DETECT_CONF,                         # Detection confidence threshold
                input_size=MODEL_INPUT_SIZE,                    # 传入推理尺寸
                original_size=(ORIGINAL_WIDTH, ORIGINAL_HEIGHT) # 传入原始尺寸
            )
    
    if not USE_POINT_DETECT_MODEL:
        print("=" * 50)
        print("使用传统关键点检测模型")
        print("=" * 50)
        kp_tracker = KeypointsTracker(
            model_path='models/weights/keypoints-detection.pt', # Keypoints Model Weights Path
            conf=.8,                                            # Field Detection confidence threshold
            kp_conf=.8,                                         # Keypoint confidence threshold
            input_size=MODEL_INPUT_SIZE,                        # 传入推理尺寸
            original_size=(ORIGINAL_WIDTH, ORIGINAL_HEIGHT)     # 传入原始尺寸
        )
    
    # 3. Assign clubs to players based on their uniforms' colors
    # Create 'Club' objects - Needed for Player Club Assignment
    # Replace the RGB values with the actual colors of the clubs.
    # club1 = Club('Club1',                       # club name
    #              (232, 247, 248),               # player jersey color
    #              (6, 25, 21)                    # goalkeeper jersey color
    #              )
    # club2 = Club('Club2',                       # club name
    #              (172, 251, 145),               # player jersey color
    #              (239, 156, 132)                # goalkeeper jersey color
    #              )

    club1 = Club('Club1',  # 红色队伍
                 (200,0,0),  # 球员球衣颜色：红色（200，0，0）
                 (30,30,30)  # 守门员球衣颜色：接近黑色（30，30，30）
                 )

    club2 = Club('Club2',  # 蓝色队伍
                 (173, 216, 230),  # 球员球衣颜色：亮蓝（0，100，255）
                 (0, 50, 100)  # 守门员球衣颜色：深蓝（0，50，150）
                 )

    # club1 = Club('Club1',  # 浅蓝色队伍
    #              (173, 216, 230),  # 球员球衣颜色：浅蓝（类似于 light blue / sky blue）
    #              (50, 50, 50)  # 守门员球衣颜色：深灰偏黑
    #              )
    #
    # club2 = Club('Club2',  # 深蓝色队伍
    #              (0, 0, 139),  # 球员球衣颜色：深蓝（类似于 dark blue / navy）
    #              (0, 50, 100)  # 守门员球衣颜色：深蓝变体
    #              )

    # # 设置 Club 对象颜色
    # club1 = Club('Club1',  # 黄色队伍
    #              (255, 255, 0),  # 球员球衣颜色：亮黄（Yellow）
    #              (50, 50, 50)  # 守门员球衣颜色：偏黑
    #              )
    #
    # club2 = Club('Club2',  # 浅蓝色队伍
    #              (173, 216, 230),  # 球员球衣颜色：浅蓝（Light Blue / Sky Blue）
    #              (0, 50, 100)  # 守门员球衣颜色：深蓝
    #              )

    # club1 = Club('Club1',  # 蓝褐色队伍
    #              (70, 90, 140),  # 球员球衣颜色：蓝褐色（Blue-Brown, 可理解为深蓝带棕调）
    #              (30, 30, 30)  # 守门员球衣颜色：深灰黑
    #              )
    #
    # club2 = Club('Club2',  # 白色队伍
    #              (240, 240, 240),  # 球员球衣颜色：白色
    #              (80, 80, 80)  # 守门员球衣颜色：中灰
    #              )

    # club1 = Club('Club1',  # 深橙偏红队伍
    #              (200, 70, 30),  # 球员球衣颜色：深橙偏红
    #              (30, 30, 30)  # 守门员球衣颜色：深灰/黑色
    #              )
    #
    # club2 = Club('Club2',  # 白色队伍
    #              (245, 245, 245),  # 球员球衣颜色：白色
    #              (100, 100, 100)  # 守门员球衣颜色：中灰色
    #              )

    # club1 = Club('Club1',  # 褐色队伍
    #              (139, 69, 19),  # 球员球衣颜色：深褐色（类似巧克力棕）
    #              (40, 40, 40)  # 守门员球衣颜色：深灰色或黑色
    #              )
    #
    # club2 = Club('Club2',  # 白色队伍
    #              (245, 245, 245),  # 球员球衣颜色：白色
    #              (100, 100, 100)  # 守门员球衣颜色：中灰色
    #              )


    # Create a ClubAssigner Object to automatically assign players and goalkeepers 
    # to their respective clubs based on jersey colors.
    club_assigner = ClubAssigner(club1, club2)

    # 4. Initialize the BallToPlayerAssigner object
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    # 5. Define the keypoints for a top-down view of the football field (from left to right and top to bottom)
    # These are used to transform the perspective of the field.
    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],                        # 0-5 (left goal line)
        [32, 122], [32, 229],                                                           # 6-7 (left goal box corners)
        [64, 176],                                                                      # 8 (left penalty dot)
        [96, 57], [96, 122], [96, 229], [96, 293],                                      # 9-12 (left penalty box)
        [263, 0], [263, 122], [263, 229], [263, 351],                                   # 13-16 (halfway line)
        [431, 57], [431, 122], [431, 229], [431, 293],                                  # 17-20 (right penalty box)
        [463, 176],                                                                     # 21 (right penalty dot)
        [495, 122], [495, 229],                                                         # 22-23 (right goal box corners)
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351],            # 24-29 (right goal line)
        [210, 176], [317, 176]                                                          # 30-31 (center circle leftmost and rightmost points)
    ])

    # 6. Initialize the video processor
    # This processor will handle every task needed for analysis.
    processor = FootballVideoProcessor(obj_tracker,                                   # Created ObjectTracker object
                                       kp_tracker,                                    # Created KeypointsTracker object
                                       club_assigner,                                 # Created ClubAssigner object
                                       ball_player_assigner,                          # Created BallToPlayerAssigner object
                                       top_down_keypoints,                            # Created Top-Down keypoints numpy array
                                       field_img_path='input_videos/field_2d_v2.png', # Top-Down field image path
                                       save_tracks_dir='output_videos',               # Directory to save tracking information.
                                       draw_frame_num=True                            # Whether or not to draw current frame number on 
                                                                                      #the output video.
                                       )
    

    
    # 优化配置 - 解决0速度问题，提高检测准确性
    processor.speed_estimator.speed_window_frames = 15        # 时间窗口: 10帧 (提高响应性)
    processor.speed_estimator.max_realistic_speed = 30.0      # 现实最大速度: 30 km/h (提高限制)
    processor.speed_estimator.min_movement_speed = 1.0        # 最小运动速度: 1.0 km/h
    processor.speed_estimator.stationary_threshold = 0.5      # 静止阈值: 0.5 km/h
    processor.speed_estimator.speed_smoothing_factor = 0.8    # 平滑因子: 0.8 (减少过度平滑)
    processor.speed_estimator.min_window_size = 4             # 最小窗口: 4帧 (更快启动)
    processor.speed_estimator.quick_response_threshold = 5.0  # 快速响应阈值: 5.0 km/h
    processor.speed_estimator.position_change_threshold = 0.5 # 位置变化阈值: 0.5米
    processor.speed_estimator.use_windowed_average = True     # 启用时间窗口平均
    processor.speed_estimator.outlier_removal = True          # 启用异常值移除 
    
    # 关键优化参数（解决0速度率12%的问题）
    processor.speed_estimator.use_average_speed = True           # 启用平均速度算法
    processor.speed_estimator.speed_update_interval = 5          # 2帧更新间隔（提高更新频率）
    processor.speed_estimator.min_tracking_records = 3           # 最少3条记录（降低过滤门槛）
    processor.speed_estimator.tracking_quality_threshold = 0.02  # 质量阈值0.02（更宽松要求）
    
    
    # 7. Process the video
    # Specify the input video path and the output video path. 
    # The batch_size determines how many frames are processed in one go.
    
    # 根据视频路径生成输出路径
    video_name = VIDEO_PATH.split('/')[-1].rsplit('.', 1)[0]
    output_path = f'output_videos/{video_name}_processed.mp4'
    
    print(f"开始处理视频:")
    print(f"   输入: {VIDEO_PATH}")
    print(f"   输出: {output_path}")
    print(f"   批处理大小: 10")
    print("=" * 50)
    
    # 控制是否显示实时检测窗口
    SHOW_PREVIEW = False  # 设置为False可以关闭实时检测窗口，提高处理速度
    
    process_video(processor,                                 # Created FootballVideoProcessor object
                  video_source=VIDEO_PATH,                   # 使用动态配置的视频路径
                  output_video=output_path,                  # 动态生成的输出路径
                  batch_size=10,                             # Number of frames to process at once
                  target_resolution=(ORIGINAL_WIDTH, ORIGINAL_HEIGHT),  # 传递实际的视频尺寸
                  show_preview=SHOW_PREVIEW                  # 控制是否显示实时检测窗口
                  )
    
    # 8. 生成速度表格（如果启用）
    if GENERATE_SPEED_TABLES:
        print("\n" + "=" * 50)
        print("生成速度分析表格...")
        print("=" * 50)
        try:
            processor.speed_estimator.generate_speed_tables(
                save_dir='output_videos',
                time_interval=SPEED_TABLE_INTERVAL,
                file_format=SPEED_TABLE_FORMAT,
                language=SPEED_TABLE_LANGUAGE
            )
            format_name = 'Excel' if SPEED_TABLE_FORMAT == 'excel' else 'CSV'
            lang_name = 'English' if SPEED_TABLE_LANGUAGE == 'english' else '中文'
            print(f"✓ 速度表格生成完成 (格式: {format_name}, 语言: {lang_name})")
            print(f"✓ 时间间隔: {SPEED_TABLE_INTERVAL}秒")
            print(f"✓ 表格保存位置: output_videos/speed_tables/")
        except Exception as e:
            print(f"⚠️  速度表格生成失败: {e}")
            if SPEED_TABLE_FORMAT == 'excel':
                print("   如果缺少openpyxl库，请运行: pip install openpyxl")
            else:
                print("   如果缺少pandas库，请运行: pip install pandas")


if __name__ == '__main__':
    main()
