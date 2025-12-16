"""
分析比例尺问题 - 为什么走路球员速度和前锋差不多
"""

def analyze_current_scale():
    """分析当前的比例尺设置"""
    print(" 比例尺问题分析")
    print("=" * 50)
    
    # 当前配置
    field_width = 528      # 俯视图像素宽度
    field_height = 352     # 俯视图像素高度
    real_field_length = 105  # 实际场地长度(米)
    real_field_width = 68    # 实际场地宽度(米)
    
    # 计算比例尺
    scale_x = real_field_length / field_width
    scale_y = real_field_width / field_height
    
    print(f"📐 当前比例尺配置:")
    print(f"   俯视图尺寸: {field_width}x{field_height} 像素")
    print(f"   实际场地: {real_field_length}x{real_field_width} 米")
    print(f"   X轴比例尺: {scale_x:.6f} m/pixel")
    print(f"   Y轴比例尺: {scale_y:.6f} m/pixel")
    
    # 分析问题
    print(f"\n 潜在问题分析:")
    
    # 1. 比例尺是否合理
    print(f"1. 比例尺合理性:")
    print(f"   1像素 = {scale_x:.3f}米 (X轴)")
    print(f"   1像素 = {scale_y:.3f}米 (Y轴)")
    
    if scale_x > 0.3:
        print(f"    X轴比例尺偏大，可能导致速度被高估")
    if scale_y > 0.3:
        print(f"    Y轴比例尺偏大，可能导致速度被高估")
    
    # 2. 测试不同的像素移动对应的速度
    print(f"\n2. 像素移动对应的速度 (假设25fps):")
    fps = 25
    test_movements = [1, 2, 5, 10, 20, 50]
    
    print(f"   {'像素/帧':<8} | {'距离(米)':<8} | {'速度(km/h)':<10}")
    print(f"   {'-'*30}")
    
    for pixels in test_movements:
        distance_m = pixels * scale_x
        speed_ms = distance_m * fps  # m/s
        speed_kmh = speed_ms * 3.6   # km/h
        print(f"   {pixels:<8} | {distance_m:<8.3f} | {speed_kmh:<10.1f}")
    
    # 3. 分析问题
    print(f"\n 问题分析:")
    print(f"如果走路球员和前锋速度差不多，可能原因:")
    print(f"1. 比例尺计算错误 - 像素到米的转换不准确")
    print(f"2. 投影坐标系统问题 - 俯视图坐标与实际场地不匹配")
    print(f"3. 边界框位置不准确 - 备用投影使用边界框可能有偏差")
    
    # 4. 建议的修正方案
    print(f"\n 建议的修正方案:")
    
    # 计算更合理的比例尺
    # 假设视频中的投影图像实际只覆盖了场地的一部分
    coverage_factors = [0.5, 0.6, 0.7, 0.8]
    
    print(f"   假设俯视图只覆盖场地的部分区域:")
    for factor in coverage_factors:
        adjusted_scale_x = (real_field_length * factor) / field_width
        adjusted_scale_y = (real_field_width * factor) / field_height
        
        # 测试10像素移动的速度
        test_pixels = 10
        test_speed = test_pixels * adjusted_scale_x * fps * 3.6
        
        print(f"   覆盖{factor*100:.0f}%场地: 比例尺={adjusted_scale_x:.4f}, 10像素/帧={test_speed:.1f}km/h")

def suggest_realistic_speeds():
    """建议合理的足球运动速度范围"""
    print(f"\n⚽ 足球运动速度参考:")
    print("=" * 30)
    
    speed_ranges = {
        "慢走": "1-3 km/h",
        "快走": "3-6 km/h", 
        "慢跑": "6-10 km/h",
        "中速跑": "10-15 km/h",
        "快跑": "15-20 km/h",
        "冲刺": "20-25 km/h",
        "全力冲刺": "25-30 km/h"
    }
    
    for motion, speed_range in speed_ranges.items():
        print(f"   {motion:<8}: {speed_range}")
    
    print(f"\n 根据您的观察:")
    print(f"   走路球员应该在: 1-6 km/h")
    print(f"   前锋冲刺应该在: 15-25 km/h")
    print(f"   如果两者差不多，说明比例尺有问题")

def calculate_corrected_scale():
    """计算修正的比例尺"""
    print(f"\n 比例尺修正建议:")
    print("=" * 30)
    
    # 假设走路球员实际应该是3km/h，但现在显示15km/h
    # 说明比例尺被高估了5倍
    current_scale_x = 105 / 528  # 0.199
    
    print(f"当前比例尺: {current_scale_x:.6f} m/pixel")
    
    # 不同的修正因子
    correction_factors = [0.2, 0.3, 0.4, 0.5]
    
    print(f"建议的修正比例尺:")
    for factor in correction_factors:
        corrected_scale = current_scale_x * factor
        print(f"   修正因子 {factor}: {corrected_scale:.6f} m/pixel")
        
        # 测试效果
        test_movement = 20  # 20像素移动
        fps = 25
        test_speed = test_movement * corrected_scale * fps * 3.6
        print(f"     20像素/帧移动 = {test_speed:.1f} km/h")

if __name__ == "__main__":
    analyze_current_scale()
    suggest_realistic_speeds()
    calculate_corrected_scale()
    
    print(f"\n 总结:")
    print("如果走路球员速度过高，需要:")
    print("1. 降低比例尺 (减少像素到米的转换)")
    print("2. 检查投影坐标系统是否正确")
    print("3. 验证俯视图是否与实际场地匹配")
    print("4. 考虑使用更精确的关键点检测")
