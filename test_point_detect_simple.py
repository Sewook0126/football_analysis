"""
简单快速测试 Point Detection 模型
从视频中提取第一帧进行测试
"""

import cv2
import os
import sys
from ultralytics import YOLO
import supervision as sv

def simple_test():
    """简单快速测试"""
    
    print("\n" + "=" * 60)
    print("Point Detection 快速测试")
    print("=" * 60)
    
    # 配置
    MODEL_PATH = 'runs/point_detect/train/weights/best.pt'
    VIDEO_PATH = 'input_videos/liverpoor vs norwich city (fourth round goal3)9(2).mp4'
    
    # 1. 检查文件
    print("\n检查文件...")
    if not os.path.exists(MODEL_PATH):
        print(f"✗ 模型不存在: {MODEL_PATH}")
        return False
    print(f"✓ 模型: {MODEL_PATH}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ 视频不存在: {VIDEO_PATH}")
        return False
    print(f"✓ 视频: {VIDEO_PATH}")
    
    # 2. 加载模型
    print("\n加载模型...")
    try:
        model = YOLO(MODEL_PATH)
        print("✓ 模型加载成功")
        
        # 显示模型信息
        if hasattr(model, 'names'):
            print(f"  模型类别: {model.names}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    # 3. 读取第一帧
    print("\n读取视频第一帧...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("✗ 无法读取视频帧")
        return False
    
    h, w = frame.shape[:2]
    print(f"✓ 帧尺寸: {w}x{h}")
    
    # 4. 测试不同置信度
    print("\n" + "=" * 60)
    print("测试不同置信度阈值:")
    print("=" * 60)
    
    confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_conf = None
    best_count = 0
    
    for conf in confidence_levels:
        try:
            results = model.predict(
                frame,
                conf=conf,
                imgsz=640,
                verbose=False
            )
            
            detections = sv.Detections.from_ultralytics(results[0])
            num_detections = len(detections)
            
            status = "✓" if num_detections > 0 else "✗"
            print(f"{status} 置信度 {conf:.1f}: 检测到 {num_detections:2d} 个目标")
            
            if num_detections > best_count:
                best_count = num_detections
                best_conf = conf
                
        except Exception as e:
            print(f"✗ 置信度 {conf:.1f}: 检测失败 - {e}")
    
    # 5. 使用最佳置信度生成可视化
    print("\n" + "=" * 60)
    print("生成可视化结果...")
    print("=" * 60)
    
    if best_count > 0:
        print(f"\n最佳置信度: {best_conf:.1f} (检测到 {best_count} 个目标)")
        
        # 重新检测
        results = model.predict(
            frame,
            conf=best_conf,
            imgsz=640,
            verbose=False
        )
        
        detections = sv.Detections.from_ultralytics(results[0])
        
        # 绘制结果
        result_frame = frame.copy()
        
        print("\n检测详情:")
        for idx, (bbox, confidence, class_id) in enumerate(zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id
        )):
            # 中心点
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_center = int((bbox[1] + bbox[3]) / 2)
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # 绘制关键点
            cv2.circle(result_frame, (x_center, y_center), 12, (255, 255, 255), 2)
            cv2.circle(result_frame, (x_center, y_center), 10, (0, 255, 0), -1)
            
            # 标签
            label = f"#{idx}"
            cv2.putText(result_frame, label, (x_center + 15, y_center - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 类别名称
            class_name = model.names.get(int(class_id), f"Class_{class_id}") if hasattr(model, 'names') else f"Class_{class_id}"
            
            print(f"  #{idx}: 类别={class_name}, 置信度={confidence:.3f}, 位置=({x_center}, {y_center})")
        
        # 保存结果
        output_path = 'test_point_detect_simple_result.jpg'
        cv2.imwrite(output_path, result_frame)
        print(f"\n✓ 可视化结果已保存: {output_path}")
        
        # 总结
        print("\n" + "=" * 60)
        print("✓ 测试成功!")
        print("=" * 60)
        print(f"模型工作正常，检测到 {best_count} 个目标")
        print(f"建议在 main.py 中设置:")
        print(f"  POINT_DETECT_CONF = {best_conf}")
        print("\n下一步:")
        print("  1. 查看结果图片确认检测效果")
        print("  2. 运行 test_point_detect_on_video.py 测试完整视频")
        print("  3. 运行 main.py 进行完整分析")
        
        return True
        
    else:
        print("\n✗ 所有置信度下都未检测到目标!")
        print("\n可能原因:")
        print("  1. 模型训练的目标类别与视频内容不匹配")
        print("  2. 模型需要重新训练")
        print("  3. 视频质量问题")
        print("\n建议:")
        print("  1. 检查模型训练数据和标签")
        print("  2. 尝试其他测试视频")
        print("  3. 检查模型训练日志")
        
        return False

if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1)

