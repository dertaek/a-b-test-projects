# numpy数组比python列表处理数据更高效

# 1.创建数组
import numpy as np
np.array([1, 4, 2, 5, 3])
np数组要求必须是同一类型的数据，如果不统一则会向上转换成统一
例如：np.array([3.14, 1, 2])将会被转换为[3.14, 1., 2.]

### 设置数组数据类型dtype()
例如：np.array([1, 2, 3, 4], dtype='float32')

### 嵌套列表构成的多维数组
np.array([range(i, i+3) for i in [2, 4, 6]])
结果：array=(\[[2, 3, 4],
              [4, 5, 6],
              [6, 7, 8]])

### 创建数值都是零的数组，长度为10
np.zero(10, dtype='int')

### 创建3*5的浮点型的数组，数值都是1
np.ones((3, 5), dtype=float)，3*5用小括号，float不加引号
结果：array(\[[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])

### 创建一个数组，值都用3.14填充
np.full((3, 5), 3.14)
array(\[[3.14, 3.14, 3.14, 3.14, 3.14],
       [3.14, 3.14, 3.14, 3.14, 3.14],
       [3.14, 3.14, 3.14, 3.14, 3.14]])

### 创建一个从0开始，20结束，步长为2的数组
np.arange(0, 20, 2)
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])

### 创建一个5个元素的数组，均匀分布在0-1上
np.linspace(0, 1, 5)
array([0.  , 0.25, 0.5 , 0.75, 1.  ])

### 生成3*3的在0-1上均匀分布的随机数数组
np.random.random((3, 3))
array(\[[0.16043423, 0.55573078, 0.1746708 ],
       [0.17790331, 0.19395533, 0.28162373],
       [0.55022424, 0.6184222 , 0.20972521]])
random.random()才能生成数组，random.rand不行

### 创建一个3*3的，在标准正态分布上取随机数的数组
np.random.normal(0, 1, (3, 3))
array(\[[ 0.15739944,  0.5215799 , -0.28232163],
       [-0.36109903, -0.31972245,  0.34080321],
       [-1.19140094,  0.85568363,  1.41904164]])

### 创建一个3*3的， [0, 10)区间的随机整数数组
np.random.randint(0, 10, (3, 3))
array(\[[3, 0, 9],
       [4, 2, 7],
       [8, 4, 1]])

### 创建一个3*3的单位矩阵
np.eye(3)
array(\[[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

### 创建一个由3个整数组成的未初始化的数组，数组的值时内存空间任意值
np.empty(3)
array([1., 1., 1.])

### 数组的属性
x1 = np.random.randint(10, size=6) 一维数组
x2 = np.random.randint(10, size=(3, 4)) 二维数组
x3 = np.random.randint(10, size(3, 4, 5)) 三维数组
print(x3.ndim)  3
print(x3.shape) (3, 4, 5)
print(x3.size)  60
print(x3.dtype) int32
print(x3.itemsize, "bytes") 数组元素字节大小
print(x3.nbytes, "bytes")   数组总字节数大小

# 数组索引
多维数组中用逗分隔的索引元组获取元素
x2[0, 0] 第一维的第一个数
也可以用这种索引方式修改数据
x2[0, 0] = 12
因为numpy数组是固定类型的所以对x2中加入小数会被短截成整数

# 数组切片跟range类似
x[start: end: step]
x[::-1] 逆序

# 多维切片
x2[:2, :3] 两行三列
x[::-1, ::-1]二维逆序

# 获取数据的行列
x2[:, 0] 第一列
x2[0, :]第一行可以省略为x[0]

### 修改列表切片不会修改列表本身，但是修改数组切片会修改数组本身，数组切片是视图不是副本，也就是说可以通过修改数组切片来修改数组
x = np.array([1, 2, 3, 4])
a = x[0:2]
a[0] = 9
x
out ：array([9, 2, 3, 4])

### 创建数组的副本 改变副本不改变原数据
x2_sub_copy = x2[:2, :2].copy() 创建副本
print(x2_sub_copy)
[[9 2]
 [5 4]]
x2_sub_copy[0, 1] = 88
x2_sub_copy
[[ 9 88]
 [ 5  4]]
 x2
array([[9, 2, 1, 3],
       [5, 4, 5, 8],
       [5, 7, 8, 9]])

### 数组的变形reshape
grid = np.array(1, 10).reshape((3, 3))
x = np.array([1, 2, 3])
x.reshape((1, 3))
array([[1, 2, 3]])

通过newaxis获得行向量
x[np.newaxis, :]
array([[1, 2, 3]])

x.reshape((3, 1))
array(\[[1],
       [2],
       [3]])

通过newaxis获得列向量
x[:, np.newaxis]
array([[1],
       [2],
       [3]])

### 数组的拼接和分裂
# 数组的拼接np.concatenate
一维数组
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])
array([1, 2, 3, 3, 2, 1])

二位数组
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
np.concatenate([grid, grid]) 按纵轴拼接
np.concatenate([grid, grid], axis=1) 横向拼接

# 垂直栈np.vstack纵向拼接
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
np.vstack([x, grid])
array([[1, 2, 3],
       [9, 8, 7],
       [6, 5, 4]])

# 水平栈数组np.hstack
y = np.array([[99],
              [99]])
np.hstack([grid, y])

# np.dstack从第三个维度拼接

# 数组的分裂split、hsplit、vsplit
x = [1, 2, 3, 99, 3, 2, 1]
x1, x2, x3=np.split(x, [3, 5]) 3和5是分割点
print(x1, x2, x3)
[1 2 3] [99  3] [2 1]
grid = np.arange(16).reshape((4, 4))
grid
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)
[[0 1 2 3]
 [4 5 6 7]]
[[ 8  9 10 11]
 [12 13 14 15]]
left, right = np.hsplit(grid, [2])
print(left)
print(right)
[[ 0  1]
 [ 4  5]
 [ 8  9]
 [12 13]]
[[ 2  3]
 [ 6  7]
 [10 11]
 [14 15]]

 ### 通用函数，使用向量的方式可以进行更快更灵活的操作，存在两种医院通用函数和二元通用函数。

### 广播适用于不同大小的数组进行通用函数运算的一种规则
# 对两个数组同时广播
a = np.array([0, 1, 2])
b = a[:, np.newaxis]
b
array(\[[0],
       [1],
       [2]])
a + b
array(\[[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])

### 广播的实际应用
# 归一化
x = np.random.random((10, 3))
array([[0.99300453, 0.15054333, 0.96942094],
       [0.62944889, 0.31208418, 0.33572911],
       [0.52333672, 0.81437792, 0.91412886],
       [0.73200702, 0.58095052, 0.18320312],
       [0.65090489, 0.81412108, 0.07863432],
       [0.94996172, 0.03540313, 0.94866817],
       [0.77249794, 0.14669197, 0.65202571],
       [0.92299792, 0.88620206, 0.6046923 ],
       [0.995873  , 0.47972669, 0.59428982],
       [0.97260109, 0.30311116, 0.99524181]]
xmean = x.mean(0)
array([0.81426337, 0.4523212 , 0.62760342])
x_centered = x - xmean
x_centered
array([[ 0.17874116, -0.30177787,  0.34181752],
       [-0.18481448, -0.14023702, -0.2918743 ],
       [-0.29092666,  0.36205672,  0.28652544],
       [-0.08225636,  0.12862931, -0.44440029],
       [-0.16335848,  0.36179988, -0.54896909],
       [ 0.13569835, -0.41691808,  0.32106475],
       [-0.04176543, -0.30562923,  0.02442229],
       [ 0.10873455,  0.43388085, -0.02291112],
       [ 0.18160963,  0.02740549, -0.0333136 ],
       [ 0.15833772, -0.14921005,  0.3676384 ]])
x_centered.mean(0) 
array([-1.33226763e-16,  6.66133815e-17,  7.77156117e-17])   其实就是0

# 画一个二维函数，基于二维函数显示图像
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** np.cos(10 + y * x) * np.cos(x)

### 比较、掩码和布尔逻辑
# 可以用来统计数组中有多少值大于某一给定值，或者删除这些门限异常点
# 和通用函数的比较操作
x = np.array([1, 2, 3, 4, 5])
x < 3
array([ True,  True, False, False, False])
(2 * x) == (x ** 2)
array([False,  True, False, False, False])

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4)) 要用rng打点调用randint才能调用随机数种子
array([[5, 0, 3, 3],
       [7, 9, 3, 5],
       [2, 4, 7, 6]])
x < 6 
array([[ True,  True,  True,  True],
       [False, False,  True,  True],
       [ True,  True, False, False]])

# 操作布尔数组
多少值小于6
np.count_nonzero(x < 6) 0是false 1是true
8
np.sum(x < 6) true是1 false是0
8

每行有多少值小于6
np.sum(x < 6, axis=1)
array([4, 2, 2])

有没有值大于8
np.any(x > 8)
True
有没有值小于零
np.any(x < 0)
False

是否所有值都小于10
np.all(x < 10)
True
是否所有值都等于6
np.all(x == 6)
False

是否行的所有值都小于8, 
np.all(x < 8, axis=1)
array([ True, False,  True]) 第一行是二行不是第三行是

np.sum((inches > 0.5) & (inches < 1))  &是且，降雨量在0.5-1之间的天数
29
np.sum(~((inches <= 0.5) | (inches >= 1))) ~代表not， |代表或
29

# 将bool数组作为掩码 (类似于用逻辑判断做特殊的索引)
x = np.array([[5, 0, 3, 3],
              [7, 9, 3, 5],
              [2, 4, 7, 6]])
x < 5 输出bool型
array([[False,  True,  True,  True],
       [False, False,  True, False],
       [ True,  True, False, False]])
x[x < 5] 将bool数组作为掩码
array([0, 3, 3, 3, 2, 4]) 

# 花哨索引，传递的是索引数组
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
[x[3], x[7], x[2]]或者
ind = [3, 7, 2]
x[ind]

花哨索引结果与索引数组形状一致，例如
ind = np.array([[3, 7],
                [4, 5]])
x[ind]
array([[71, 86],
       [60, 20]])

对多维数组也适用,跟索引数组形状一致
x = np.arange(12).reshape(3, 4)
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
x [row, col]
array([ 2,  5, 11])

x[row[:, np.newaxis], col] 索引广播了,花哨索引形状是索引数组广播后的形状
array([[ 2,  1,  3],
       [ 6,  5,  7],
       [10,  9, 11]])

# 组合索引
x[2, [2, 0, 1]]
x[1:, [2, 0, 1]]
mask = np.array([1, 0, 1, 0], dtype=bool)
x[row[:, np.newaxis], mask] 广播了

### 示例：随机点选择
mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
x = np.random.multivariate_normal(mean, cov, 100)
x.shape
利用花哨索引随机选取20个随机不重复的索引值，并利用索引值取到原数组对应的值
indices = np.random.choice(x.shape[0], 20, replace=False) x.shape[0]得到的行数，x.shape[1]的到的是列数，false表示不能取相同的数字。取得是整数。
selection = x[indices]
plt.scatter(x[:, 0], x[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1], facecolor='none', edgecolor='b', s=200) facecolor='none'就是圈，s是大小

# 用花哨索引修改值
x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)
[ 0 99 99  3 99  5  6  7 99  9]
x[i] -= 10
print(x)
[ 0 89 89  3 89  5  6  7 89  9]
x = np.zeros(10)
x[[0, 0]] = [4, 6]
print(x)
[6. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
i = [2, 3, 3, 4, 4, 4]
x[i] += 1
x
array([6., 0., 1., 1., 1., 0., 0., 0., 0., 0.]) 并没有累加
x = np.zeros(10)
np.add.at(x, i, 1)
print(x)
[0. 0. 1. 2. 3. 0. 0. 0. 0. 0.]