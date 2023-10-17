# 直方图过滤离散点
当数据不存在离散点时，非对称量化得到的量化数据是饱和的。但是当数据存在离散点时，量化后的数据就会分布不合理。  
通过直方图可以有效过滤离散点，即在一定置信度范围内保留一定范围的数据，将范围外的数据当作离散点进行过滤。

# 代码
```python
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
```