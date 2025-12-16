"""
快速测试 Point Detection 模型是否能正常检测关键点
"""

import cv2
import os
import sys
from tracking import PointDetectTracker

def test_point_detection():
    """测试 Point Detection 模型"""
    
    print("=" * 60)
    print("Point Detection 模型测试")
    print("=" * 60)
    
    # 检查模型文件
    model_path = 'runs/point_detect/train/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        return False
    
    print(f"✓ 模型文件存在: {model_path}")
    
    # 检查测试图片
    test_images = [
        'extracted_frames/frame_01_time_0.00s.jpg',
        'extracted_frames/frame_02_time_0.77s.jpg',
        'extracted_frames/a9f16c_2_10_png.rf.1f66b9ae700c46168ffe92b38d4bb646.jpg'
    ]
    
    test_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if test_image is None:
        print("✗ 未找到测试图片")
        print("   请确保 extracted_frames/ 目录下有图片")
        return False
    
    print(f"✓ 使用测试图片: {test_image}")
    
    # 加载图片
    frame = cv2.imread(test_image)
    if frame is None:
        print(f"✗ 无法读取图片: {test_image}")
        return False
    
    h, w = frame.shape[:2]
    print(f"✓ 图片尺寸: {w}x{h}")
    
    # 初始化 PointDetectTracker
    try:
        print("\n初始化 PointDetectTracker...")
        tracker = PointDetectTracker(
            model_path=model_path,
            conf=0.3,  # 降低置信度以便检测更多点
            input_size=640,
            original_size=(w, h)
        )
        print("✓ PointDetectTracker 初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False
    
    # 执行检测
    try:
        print("\n执行关键点检测...")
        detections = tracker.detect([frame])
        keypoints = tracker.track(detections[0])
        
        print(f"\n检测结果:")
        print(f"  检测到关键点数量: {len(keypoints)}")
        
        if len(keypoints) == 0:
            print("\n⚠️  未检测到任何关键点!")
            print("  可能原因:")
            print("  1. 模型置信度阈值过高（当前: 0.3）")
            print("  2. 图片中没有训练的目标类别")
            print("  3. 模型需要重新训练")
            print("\n建议:")
            print("  1. 尝试降低 POINT_DETECT_CONF 到 0.2 或更低")
            print("  2. 检查模型训练时使用的类别")
            return False
        
        # 显示检测到的关键点
        print("\n关键点详情:")
        for idx, (x, y) in list(keypoints.items())[:10]:
            print(f"  关键点 {idx}: ({x:.1f}, {y:.1f})")
        
        if len(keypoints) > 10:
            print(f"  ... 还有 {len(keypoints) - 10} 个关键点")
        
        # 可视化结果
        print("\n生成可视化结果...")
        result_frame = frame.copy()
        
        for idx, (x, y) in keypoints.items():
            # 绘制关键点
            cv2.circle(result_frame, (int(x), int(y)), 8, (255, 255, 255), 2)
            cv2.circle(result_frame, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(result_frame, str(idx), (int(x) + 10, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果
        output_path = 'test_point_detection_result.jpg'
        cv2.imwrite(output_path, result_frame)
        print(f"✓ 结果已保存: {output_path}")
        
        print("\n" + "=" * 60)
        print("✓ 测试成功!")
        print(f"✓ Point Detection 模型工作正常，检测到 {len(keypoints)} 个关键点")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 检测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_point_detection()
    sys.exit(0 if success else 1)

