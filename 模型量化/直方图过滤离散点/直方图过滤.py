import numpy as np

# 截断
def saturate(x, int_max, int_min):
    return np.round(np.clip(x, int_min, int_max)).astype(np.int8)

# 计算Scale
def scale_cal(x, int_max):
    max_val = np.max(np.abs(x))
    return max_val / int_max

# 量化 Q
def quant_float_data(x, scale, int_max, int_min):
    xq = np.round(x / scale)
    xq = saturate(xq, int_max, int_min) # 截断
    return xq

# 反量化
def dequant_data(xq, scale):
    x = (xq * scale).astype('float32')
    return x

def histgram_range(x, int_max):
    hist, range = np.histogram(x, 100) # 划分成100块
    total = len(x) # 数据量
    left = 0
    right = len(hist) - 1
    limit = 0.99 # 只保留99%的数据
    while True:
        cover_paecent = hist[left:right].sum() / total
        if cover_paecent <= limit:
            break
        # 双指针移动
        elif(hist[left] <= hist[right]):
            left += 1
        else:
            right -= 1
    
    left_val = range[left]
    right_val = range[right]
    dynamic_range = max(abs(left_val), abs(right_val))
    return  dynamic_range / int_max # cal scale

if __name__ == "__main__":
    data_float32 = np.random.randn(1000).astype('float32') # 测试数据
    int_max = 127 # 最大值
    int_min = -128 # 最小值
    
    # 这里的int_max实质是最大值和最小值中绝对值最大的那个，但由于实际工程一般不使用-128，因此这里使用了127
    scale1 =  scale_cal(data_float32, int_max)
    scale2 = histgram_range(data_float32, int_max) # 基于直方图求解scale
    print(scale1, scale2)
    # int8量化
    data_int8 = quant_float_data(data_float32, scale2, int_max, int_min)
    # 反量化
    data_dequant_float = dequant_data(data_int8, scale2)
    
    # 打印数据
    print("quant result: ", data_int8)
    print("dequant result: ", data_dequant_float)
    # 打印误差
    print("diff: ", data_dequant_float - data_float32)