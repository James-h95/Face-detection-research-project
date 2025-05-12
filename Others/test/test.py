import time
import numpy as np

# 创建两个大数组
arr1 = np.random.rand(10**6)
arr2 = np.random.rand(10**6)

# 1. 使用 for 循环
start = time.time()
result_for = [arr1[i] + arr2[i] for i in range(len(arr1))]
end = time.time()
print("For loop took:", end - start)

# 2. 使用 NumPy
start = time.time()
result_numpy = arr1 + arr2
end = time.time()
print("NumPy took:", end - start)
