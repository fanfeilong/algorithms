import numpy as np
from scipy.signal import medfilt

def detect_anomalies(values, window_size=5, threshold=10):
    """
    检测异常速度数据点
    参数:
    - values: 速度数据列表
    - window_size: 滑动中位数窗口大小
    - threshold: 速度变化阈值 (单位: 速度)

    返回:
    - (True/False, 异常点索引列表, 异常点值列表)
    """
    if len(values) < window_size:
        return False, [], []  # 数据量不足，无法计算滑动中位数

    # 计算滑动中位数
    # 使用scipy.signal.medfilt计算滑动中位数
    # 使用中位数滤波而不是移动平均的原因:
    # 1. 中位数对异常值更不敏感,能更好地保持数据的突变特征
    # 2. 移动平均会被异常值显著影响,导致平滑后的基准线发生偏移
    # 3. 在速度数据中,中位数能更好地保持正常速度变化的特征
    smoothed_values = medfilt(values, kernel_size=window_size)

    # 计算差异
    diff = np.abs(np.array(values) - smoothed_values)

    # 找到异常点
    anomaly_indices = np.where(diff > threshold)[0]

    return bool(len(anomaly_indices)), anomaly_indices.tolist(), np.array(values)[anomaly_indices].tolist()

# 测试数据
test_values = [60, 61, 62, 61, 59, 100, 61, 60, 62, 63, 65, 120, 64, 63, 62]

# 测试入口
if __name__ == "__main__":
    has_anomalies, anomaly_indices, anomaly_values = detect_anomalies(test_values)
    print(f"异常数据检测结果: {has_anomalies}")
    print(f"异常数据索引: {anomaly_indices}")
    print(f"异常数据值: {anomaly_values}")
