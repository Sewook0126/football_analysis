"""
批量测试所有视频的主程序
用于测试不同视频下的速度计算参数，寻找最优阈值
"""

import os
import glob
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path

# 导入必要的模块
from utils.video_utils_simple import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
from video_info_helper import create_dynamic_config

def get_all_videos(input_dir: str) -> list:
    """获取输入目录下的所有视频文件"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    return sorted(video_files)

def create_video_processor(model_input_size: int, original_width: int, original_height: int, 
                          video_output_dir: str = 'output_videos'):
    """创建视频处理器（复用main.py的逻辑）"""
    
    # 1. 加载目标检测模型
    obj_tracker = ObjectTracker(
        model_path='models/weights/object-detection.pt',
        conf=.7,
        ball_conf=.3,
        input_size=model_input_size,
        original_size=(original_width, original_height)
    )

    # 2. 加载关键点检测模型
    kp_tracker = KeypointsTracker(
        model_path='models/weights/keypoints-detection.pt',
        conf=.8,
        kp_conf=.8,
        input_size=model_input_size,
        original_size=(original_width, original_height)
    )
    
    # 3. 设置球队颜色（使用通用设置）
    club1 = Club('Club1',
                 (200, 0, 0),    # 红色队伍
                 (30, 30, 30)    # 守门员
                 )

    club2 = Club('Club2',
                 (255, 255, 0),  # 黄色队伍
                 (50, 50, 50)    # 守门员
                 )

    club_assigner = ClubAssigner(club1, club2)
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    # 4. 定义关键点（复用main.py的设置）
    import numpy as np
    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],
        [32, 122], [32, 229],
        [64, 176],
        [96, 57], [96, 122], [96, 229], [96, 293],
        [263, 0], [263, 122], [263, 229], [263, 351],
        [431, 57], [431, 122], [431, 229], [431, 293],
        [463, 176],
        [495, 122], [495, 229],
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351],
        [210, 176], [317, 176]
    ])

    # 5. 初始化视频处理器
    processor = FootballVideoProcessor(
        obj_tracker, kp_tracker, club_assigner, ball_player_assigner,
        top_down_keypoints,
        field_img_path='input_videos/field_2d_v2.png',
        save_tracks_dir=video_output_dir,  # 使用单独的视频输出目录
        draw_frame_num=True
    )
    
    # 6. 配置速度计算参数（基于39个视频测试的优化参数）
    processor.speed_estimator.speed_window_frames = 10
    processor.speed_estimator.max_realistic_speed = 30.0
    processor.speed_estimator.min_movement_speed = 1.0
    processor.speed_estimator.stationary_threshold = 0.5
    processor.speed_estimator.speed_smoothing_factor = 0.8
    processor.speed_estimator.min_window_size = 4
    processor.speed_estimator.quick_response_threshold = 5.0
    processor.speed_estimator.position_change_threshold = 0.5
    processor.speed_estimator.use_windowed_average = True
    processor.speed_estimator.outlier_removal = True
    processor.speed_estimator.use_average_speed = True
    processor.speed_estimator.speed_update_interval = 2
    processor.speed_estimator.min_tracking_records = 3
    processor.speed_estimator.tracking_quality_threshold = 0.02
    
    return processor

def process_single_video(video_path: str, output_base_dir: str) -> dict:
    """处理单个视频并返回统计结果"""
    
    print(f"\n{'='*80}")
    print(f"开始处理视频: {os.path.basename(video_path)}")
    print(f"{'='*80}")
    
    try:
        # 动态获取视频配置
        video_config = create_dynamic_config(video_path, 640)  # 使用640作为默认推理尺寸
        
        model_input_size = video_config['input_size']
        original_width = video_config['original_width']
        original_height = video_config['original_height']
        video_fps = video_config['fps']
        
        print(f"视频配置:")
        print(f"   原始尺寸: {original_width}x{original_height}")
        print(f"   推理尺寸: {model_input_size}x{model_input_size}")
        print(f"   帧率: {video_fps} FPS")
        
        # 为每个视频创建独立的输出目录
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 创建视频处理器
        processor = create_video_processor(model_input_size, original_width, original_height, video_output_dir)
        
        # 设置输出路径
        output_video_path = os.path.join(video_output_dir, f"{video_name}_processed.mp4")
        
        # 处理视频（不显示预览窗口）
        process_video(
            processor=processor,
            video_source=video_path,
            output_video=output_video_path,
            batch_size=10,
            target_resolution=(original_width, original_height),
            show_preview=False  #  关闭预览窗口
        )
        
        # 处理完成后的额外操作
        if hasattr(processor, 'speed_estimator'):
            # 1. 保存速度分析到视频目录
            speed_file = processor.speed_estimator.save_speed_analysis(video_output_dir, f"{video_name}_speed_analysis.json")
            
            # 2. 生成并保存速度曲线到视频目录
            processor.speed_estimator.plot_speed_curves(video_output_dir)
            
            # 3. 移动或复制JSON文件到视频目录（避免覆盖）
            
            # JSON文件已经直接保存在video_output_dir中，无需复制
            # 检查并确认文件是否存在
            json_files = ['object_tracks.json', 'keypoint_tracks.json']
            for json_file in json_files:
                json_path = os.path.join(video_output_dir, json_file)
                if os.path.exists(json_path):
                    print(f"   确认文件: {json_file}")
                else:
                    print(f"   警告: 未找到 {json_file}")
            
            print(f"   文件检查完成")
        
        # 收集统计信息
        stats = {
            'video_name': video_name,
            'video_path': video_path,
            'output_dir': video_output_dir,
            'original_size': f"{original_width}x{original_height}",
            'fps': video_fps,
            'total_frames': processor.speed_estimator.frame_count if hasattr(processor, 'speed_estimator') else 0,
            'total_objects': len(processor.speed_estimator.player_max_speeds) if hasattr(processor, 'speed_estimator') else 0,
            'max_speeds': {},
            'avg_speeds': {},
            'zero_speed_objects': 0,
            'valid_objects': 0
        }
        
        # 分析速度数据
        if hasattr(processor, 'speed_estimator') and processor.speed_estimator.player_max_speeds:
            for player_id, max_speed in processor.speed_estimator.player_max_speeds.items():
                stats['max_speeds'][str(player_id)] = max_speed
                
                if player_id in processor.speed_estimator.player_avg_speeds:
                    avg_speeds = processor.speed_estimator.player_avg_speeds[player_id]
                    if avg_speeds:
                        avg_speed = sum(avg_speeds) / len(avg_speeds)
                        stats['avg_speeds'][str(player_id)] = avg_speed
                        
                        # 统计有效和0速度对象
                        if avg_speed == 0.0:
                            stats['zero_speed_objects'] += 1
                        elif len(avg_speeds) >= processor.speed_estimator.min_tracking_records:
                            stats['valid_objects'] += 1
        
        print(f" 视频处理完成!")
        print(f"   输出目录: {video_output_dir}")
        print(f"   总对象数: {stats['total_objects']}")
        print(f"   有效对象: {stats['valid_objects']}")
        print(f"   0速度对象: {stats['zero_speed_objects']}")
        
        return stats
        
    except Exception as e:
        print(f" 处理视频时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'video_name': Path(video_path).stem,
            'video_path': video_path,
            'error': str(e),
            'status': 'failed'
        }

def generate_summary_report(all_stats: list, output_dir: str):
    """生成批量处理的汇总报告"""
    
    report_path = os.path.join(output_dir, f"batch_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(" 批量视频测试汇总报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试视频数量: {len(all_stats)}\n\n")
        
        # 统计总览
        total_videos = len(all_stats)
        successful_videos = len([s for s in all_stats if 'error' not in s])
        failed_videos = total_videos - successful_videos
        
        f.write(" 处理统计:\n")
        f.write(f"  总视频数: {total_videos}\n")
        f.write(f"  成功处理: {successful_videos}\n")
        f.write(f"  处理失败: {failed_videos}\n\n")
        
        # 成功处理的视频详情
        if successful_videos > 0:
            f.write(" 成功处理的视频:\n")
            f.write("-" * 60 + "\n")
            
            total_objects = 0
            total_valid_objects = 0
            total_zero_speed = 0
            
            for stats in all_stats:
                if 'error' not in stats:
                    f.write(f"\n {stats['video_name']}\n")
                    f.write(f"   尺寸: {stats['original_size']}, FPS: {stats['fps']}\n")
                    f.write(f"   总帧数: {stats['total_frames']}\n")
                    f.write(f"   检测对象: {stats['total_objects']}\n")
                    f.write(f"   有效对象: {stats['valid_objects']}\n")
                    f.write(f"   0速度对象: {stats['zero_speed_objects']}\n")
                    f.write(f"   输出目录: {stats['output_dir']}\n")
                    
                    # 显示前5个最高速度
                    if stats['max_speeds']:
                        sorted_speeds = sorted(stats['max_speeds'].items(), 
                                             key=lambda x: x[1], reverse=True)[:5]
                        f.write(f"   前5最高速度: {sorted_speeds}\n")
                    
                    total_objects += stats['total_objects']
                    total_valid_objects += stats['valid_objects']
                    total_zero_speed += stats['zero_speed_objects']
            
            f.write(f"\n 总体统计:\n")
            f.write(f"   总检测对象: {total_objects}\n")
            f.write(f"   总有效对象: {total_valid_objects}\n")
            f.write(f"   总0速度对象: {total_zero_speed}\n")
            if total_objects > 0:
                f.write(f"   有效率: {total_valid_objects/total_objects*100:.1f}%\n")
                f.write(f"   0速度率: {total_zero_speed/total_objects*100:.1f}%\n")
        
        # 失败的视频
        if failed_videos > 0:
            f.write(f"\n 处理失败的视频:\n")
            f.write("-" * 60 + "\n")
            for stats in all_stats:
                if 'error' in stats:
                    f.write(f" {stats['video_name']}: {stats['error']}\n")
        
        f.write(f"\n 当前参数配置:\n")
        f.write(f"   speed_window_frames: 12\n")
        f.write(f"   speed_update_interval: 3\n")
        f.write(f"   min_tracking_records: 5\n")
        f.write(f"   tracking_quality_threshold: 0.05\n")
        f.write(f"   max_realistic_speed: 28.0 km/h\n")
    
    print(f" 汇总报告已保存: {report_path}")
    return report_path

def main():
    """主函数 - 批量测试所有视频"""
    
    print(" 开始批量视频测试")
    print("=" * 80)
    
    # 配置路径
    input_videos_dir = "input_videos"
    output_base_dir = "output_all_videos"
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = get_all_videos(input_videos_dir)
    
    if not video_files:
        print(f" 在 {input_videos_dir} 目录下没有找到视频文件")
        return
    
    print(f" 找到 {len(video_files)} 个视频文件:")
    for i, video_path in enumerate(video_files, 1):
        print(f"   {i}. {os.path.basename(video_path)}")
    
    # 开始批量处理
    all_stats = []
    start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n 进度: {i}/{len(video_files)}")
        
        stats = process_single_video(video_path, output_base_dir)
        all_stats.append(stats)
        
        # 显示进度
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = (len(video_files) - i) * avg_time
        print(f"  已用时: {elapsed/60:.1f}分钟, 预计剩余: {remaining/60:.1f}分钟")
    
    # 生成汇总报告
    total_time = time.time() - start_time
    print(f"\n 批量处理完成!")
    print(f"  总耗时: {total_time/60:.1f}分钟")
    
    report_path = generate_summary_report(all_stats, output_base_dir)
    
    print(f"\n 所有结果已保存到: {output_base_dir}")
    print(f" 查看汇总报告: {report_path}")
    print(f"\n 提示: 可以通过分析报告来调整速度计算参数，寻找最优阈值")

if __name__ == '__main__':
    main()
