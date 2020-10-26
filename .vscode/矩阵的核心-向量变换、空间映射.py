# 对角阵np.diag()
import numpy as np
A = np.diag([1, 2, 3])
# 单位矩阵np.eye()
I = np.eye(4)
# 矩阵乘以向量的本质,结果也是一个向量,原空间的向量坐标,被映射到了新的向量空间中得到的新坐标
# 矩阵乘以向量的本质也是该矩阵列向量进行线性组合的过程，组合系数是相乘的那个向量的坐标，其实是旧空间中的坐标不变在新空间中所表示的向量。向量变了。本质是一种向量映射

# 矮胖矩阵对空间的降维压缩
# 高瘦矩阵无法覆盖目标空间，向量的信息不增加。

# 求矩阵的秩np.linalg.matrix_rank()
import numpy as np
a_1 = np.array([[1, 1, 0],
                [1, 0, 1]])
print(np.linalg.matrix_rank(a_1))