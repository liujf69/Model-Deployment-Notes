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

if __name__ == "__main__":
    data_float32 = np.random.randn(3).astype('float32') # 测试数据
    int_max = 127 # 最大值
    int_min = -128 # 最小值
    
    # 这里的int_max实质是最大值和最小值中绝对值最大的那个，但由于实际工程一般不使用-128，因此这里使用了127
    scale =  scale_cal(data_float32, int_max)
    # int8量化
    data_int8 = quant_float_data(data_float32, scale, int_max, int_min)
    # 反量化
    data_dequant_float = dequant_data(data_int8, scale)
    
    # 打印数据
    print("quant result: ", data_int8)
    print("dequant result: ", data_dequant_float)
    # 打印误差
    print("diff: ", data_dequant_float - data_float32)