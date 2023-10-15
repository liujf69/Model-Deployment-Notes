# 非对称量化
# 引入偏移量Z解决截断的问题
# 量化数据是饱和的（量化后的最小值就是范围的最小值，例如int8的-128，量化后的最大值就是范围的最大值，例如int8中的127）

import numpy as np

# 截断
def saturate(x, int_max, int_min):
    return np.round(np.clip(x, int_min, int_max)).astype(np.int8)

# 计算Scale和偏移量Z, Scale = (R_max - R_min) / (Q_max - Q_min), Z = Q_max - Round(R_max / scale)
def scale_z_cal(x, int_max, int_min):
    scale = (x.max() - x.min()) / (int_max - int_min)
    z = int_max - (x.max() / scale)
    return scale, z

# 量化 Q = Round(R / Scale + Z)
def quant_float_data(x, scale, z):
    xq = np.round(x / scale) + z
    xq = saturate(xq, int_max, int_min) # 截断
    return xq

# 反量化
def dequant_data(xq, scale, z):
    x = ((xq - z) * scale).astype('float32')
    return x
    
if __name__ == "__main__":
    data_float32 = np.random.randn(3).astype('float32') # 测试数据
    data_float32[0] = -0.61
    data_float32[1] = -0.52
    data_float32[2] = 1.62
    int_max = 127 # 最大值
    int_min = -128 # 最小值
    
    # 计算scale和z
    scale, z = scale_z_cal(data_float32, int_max, int_min)
    # int8量化
    data_int8 = quant_float_data(data_float32, scale, z)
    # 反量化
    data_dequant_float = dequant_data(data_int8, scale, z)
    
    # 打印数据
    print("quant result: ", data_int8)
    print("dequant result: ", data_dequant_float)
    # 打印误差
    print("diff: ", data_dequant_float - data_float32)