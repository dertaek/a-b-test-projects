# 此处需要巧用均匀分布来判别采样点是否拒绝。
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

def p(x):
    return (0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)) / 1.2113

# 先把均匀分布和标准分布准备好
uniform_rv = uniform(loc=0, scale=1)
norm_rv = norm(loc=1.4, scale=1.2)
c = 2.5

samples = []
# 采样500000次
for i in range(500000):
    y = norm_rv.rvs(1)[0]
    p1 = uniform_rv.rvs(1)[0]
# 此处必须用变形等式用乘法
    if p(y) >= p1 * c*norm_rv.pdf(y):
        samples.append(y)

x = np.arange(-3., 5., 0.01)
plt.gca().axes.set_xlim(-3, 5)
plt.plot(x, p(x), color='r', lw=5)
plt.hist(samples, color='g', bins=500, density=True, edgecolor='k', alpha=0.2)
plt.show()