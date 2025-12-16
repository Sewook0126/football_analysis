import cv2
import os
import numpy as np
from pathlib import Path
import mediapipe as mp

def extract_frames_from_video(video_path, output_dir, num_frames=10):
    """
    从视频中均匀截取指定数量的帧
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return []
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"视频信息：")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 帧率: {fps:.2f} FPS")
    print(f"  - 时长: {duration:.2f} 秒")
    
    # 计算要截取的帧的索引
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    extracted_files = []
    
    for i, frame_idx in enumerate(frame_indices):
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # 保存帧
            timestamp = frame_idx / fps
            filename = f"frame_{i+1:02d}_time_{timestamp:.2f}s.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            extracted_files.append(filepath)
            print(f"已保存帧 {i+1}/{num_frames}: {filename}")
        else:
            print(f"警告：无法读取第 {frame_idx} 帧")
    
    cap.release()
    print(f"\n成功从视频中提取了 {len(extracted_files)} 张图片到: {output_dir}")
    return extracted_files

def test_keypoint_detection(image_paths, output_dir):
    """
    使用MediaPipe检测关键点并绘制结果
    """
    # 初始化MediaPipe姿态检测
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # 创建关键点检测输出目录
    keypoint_output_dir = os.path.join(output_dir, "keypoint_results")
    os.makedirs(keypoint_output_dir, exist_ok=True)
    
    print(f"\n开始关键点检测测试...")
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:
        
        for i, image_path in enumerate(image_paths):
            print(f"处理图片 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"错误：无法读取图片 {image_path}")
                continue
            
            # 转换BGR到RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 进行姿态检测
            results = pose.process(image_rgb)
            
            # 在图片上绘制关键点
            annotated_image = image.copy()
            
            if results.pose_landmarks:
                # 绘制姿态关键点
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                print(f"  - 检测到姿态关键点")
                
                # 计算关键点数量
                landmarks = results.pose_landmarks.landmark
                visible_points = sum(1 for lm in landmarks if lm.visibility > 0.5)
                print(f"  - 可见关键点数量: {visible_points}/{len(landmarks)}")
                
            else:
                print(f"  - 未检测到姿态关键点")
                # 在图片上添加文字说明
                cv2.putText(annotated_image, "No pose detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存带关键点的图片
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(keypoint_output_dir, f"{base_name}_keypoints.jpg")
            cv2.imwrite(output_path, annotated_image)
            
    print(f"\n关键点检测结果已保存到: {keypoint_output_dir}")

def main():
    # 设置路径
    video_path = "D:/desktop/xy/xy_football/football_analysis/input_videos/liverpoor vs norwich city (fourth round goal3)9(2).mp4"
    output_base_dir = "D:/desktop/xy/xy_football/football_analysis/extracted_frames"
    
    print("=== 足球视频帧提取和关键点检测测试 ===\n")
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 {video_path}")
        return
    
    # 提取视频帧
    print("1. 开始提取视频帧...")
    extracted_files = extract_frames_from_video(video_path, output_base_dir, num_frames=10)
    
    if not extracted_files:
        print("错误：未能提取任何帧")
        return
    
    # 进行关键点检测测试
    print("\n2. 开始关键点检测测试...")
    test_keypoint_detection(extracted_files, output_base_dir)
    
    print("\n=== 任务完成 ===")
    print(f"提取的原始帧位置: {output_base_dir}")
    print(f"关键点检测结果位置: {output_base_dir}/keypoint_results")
    print("\n您可以查看关键点检测结果来评估模型的准确性！")

if __name__ == "__main__":
    main()
