# 首先把我们用来采样的标准分布乘以c值与被采样的特殊分布画图作比较
import numpy as np
import matplotlib.pyplot as plt
# 此处使用正态分布作为标准分布
from scipy.stats import norm
import seaborn as sns
sns.set()

# 这是被采样的特殊分布密度函数
def p(x):
    return (0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)) / 1.2113

# 这是标准分布
norm_rv = norm(loc=1.4, scale=1.2)

# c值,要尽可能在标准分布整体大于被采样分布的基础上尽可能取的小，越小采样接受概率更高，采样速度更快。
c = 2.3

x = np.arange(-4., 6., 0.01)
plt.plot(x, p(x), color='r', lw=5, label='p(x)')
plt.plot(x, c*norm_rv.pdf(x), color='b', lw=5, label='2.3*norm(1.4, 1.2)')
plt.legend()
plt.show()