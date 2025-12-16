import cv2
import os
import time
from typing import Optional

def process_video(processor=None, video_source: str = 0, output_video: Optional[str] = "output.mp4", 
                  batch_size: int = 30, skip_seconds: int = 0, target_resolution: tuple = None,
                  show_preview: bool = True) -> None:
    """
    简化的视频处理函数 - 直接保存视频，不使用临时目录
    
    Args:
        processor: 视频处理器
        video_source (str): 视频源路径
        output_video (Optional[str]): 输出视频路径，None则不保存
        batch_size (int): 批处理大小
        skip_seconds (int): 跳过开始的秒数
        target_resolution (tuple): 目标分辨率 (width, height)
        show_preview (bool): 是否显示实时检测窗口，默认True
    """
    from annotation import AbstractVideoProcessor  # Lazy import

    if processor is not None and not isinstance(processor, AbstractVideoProcessor):
        raise ValueError("The processor must be an instance of AbstractVideoProcessor.")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_skip = int(skip_seconds * fps)
    
    # 获取原始视频尺寸
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 确定最终输出尺寸
    if target_resolution:
        target_width, target_height = target_resolution
        print(f"视频将调整为: {target_width}x{target_height}")
    else:
        target_width, target_height = original_width, original_height
        print(f"使用原始视频尺寸: {target_width}x{target_height}")
    
    print(f"视频信息: {fps} FPS, 原始尺寸 {original_width}x{original_height}")
    
    # 初始化视频写入器
    video_writer = None
    if output_video is not None:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        
        # 使用更兼容的编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者用 'XVID'
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (target_width, target_height))
        
        if not video_writer.isOpened():
            print(f"无法创建视频写入器: {output_video}")
            cap.release()
            return
        else:
            print(f"视频写入器初始化成功: {output_video}")
    
    # 跳过开始的帧
    if frames_to_skip > 0:
        print(f"跳过前 {skip_seconds} 秒 ({frames_to_skip} 帧)")
        for _ in range(frames_to_skip):
            cap.read()
    
    frame_count = 0
    batch_frames = []
    
    print("开始处理视频 (按 'q' 退出)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频读取完成")
                break
            
            # 调整帧大小（如果需要）
            if target_resolution and (frame.shape[1] != target_width or frame.shape[0] != target_height):
                frame = cv2.resize(frame, (target_width, target_height))
            
            batch_frames.append(frame)
            
            # 当达到批次大小时处理
            if len(batch_frames) >= batch_size:
                processed_frames = process_batch(processor, batch_frames, fps)
                
                # 保存和显示处理后的帧
                for processed_frame in processed_frames:
                    frame_count += 1
                    
                    # 写入视频文件
                    if video_writer is not None:
                        video_writer.write(processed_frame)
                    
                    # 显示帧（如果启用预览）
                    if show_preview:
                        cv2.imshow('Football Analysis', processed_frame)
                        
                        # 检查退出键
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("用户按下 'q'，停止处理")
                            raise KeyboardInterrupt
                
                # 清空批次
                batch_frames = []
                
                # 显示进度
                if frame_count % (fps * 5) == 0:  # 每5秒显示一次
                    print(f"已处理 {frame_count} 帧 ({frame_count/fps:.1f}秒)")
        
        # 处理剩余的帧
        if batch_frames:
            processed_frames = process_batch(processor, batch_frames, fps)
            for processed_frame in processed_frames:
                frame_count += 1
                if video_writer is not None:
                    video_writer.write(processed_frame)
                if show_preview:
                    cv2.imshow('Football Analysis', processed_frame)
        
        print(f"视频处理完成！总共处理了 {frame_count} 帧")
        
        # 保存速度分析结果
        if hasattr(processor, 'speed_estimator'):
            processor.speed_estimator.print_speed_summary()
            speed_file = processor.speed_estimator.save_speed_analysis()
            print(f"速度分析已保存: {speed_file}")
            
            # 自动绘制速度曲线
            print("正在生成速度变化曲线...")
            processor.speed_estimator.plot_speed_curves()
    
    except KeyboardInterrupt:
        print("\n处理被用户中断")
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"视频已保存: {output_video}")
        if show_preview:
            cv2.destroyAllWindows()
    
    print("视频处理完成！")


def process_batch(processor, frames, fps):
    """处理一批帧"""
    if processor is None:
        return frames
    
    try:
        return processor.process(frames, fps)
    except Exception as e:
        print(f"批处理出错: {e}")
        return frames
