# 导入库
mport matplotlib as mpl
import matplotlib.pyplot as p

# 择绘图样
plt.style.use('classic')

# 显示图像
plt.show() 注意每个session只能使用一次

# 在ipython shell中画图
%matplotlib 调用魔法命令
import matplotlib.pyplot as plt
此后任何plt命令都会打开窗口，增加新的命令图形就会更新，不会及时跟新的使用plt.draw()

# 在ipython notebook中交互式画图
%matplotlib notebook 启用交互式
%matplotlib inline 启用静态图形

# 将图形保存为文件
fig.savefig('my_figure.png')

# 用Ipython的image对象来显示图片内容
from ipython.display import image
image('my_figure.png')

# 查看可保存的格式 
plt.figure().canvas.get_supported_filetypes()

matplotlib的类matlab接口 (plt.plot()) plt.figure() 
创建图形容器并进行进一步设置，是一个类的实例。

创建两个子图中的第一个，设置坐标轴
plt.subplot(2, 1, 1) (行，列，第几个)
第二个
plt.subplot(2, 1, 2) 此时很难回到第一个子图，这也是这个接口的缺点

# 获取当前图型，查看信息
plt.gca()

面向对象接口，不再受当前图形的限制，变成了显式figure和axes方法
fig, ax = plt.subplot(2) fig用来创建图形网格，ax是一个包含俩个axes对象的数组
在每个对象上调用plot方法
ax[0].plot()
ax[1].plot()

# 调用seaborn风格
plt.style.use('seaborn-whitegrid')

# 简易线型图
fig, ax = plt.figure()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))
也可以使用类matlab接口
plt.plot(x, np.sin(x))
如果想在加一条线可以调用plt.plot(x, np.cos(x))

# 调整图型:线条颜色和风格
color='blue' 全称
     ='r' 标准颜色代码'rgbcmyk' c是青色，m是品红， k是黑色
     ='0.75' 范围在0-1之间的灰度值
     ='#FFDD44' 十六进制
     =(1.0, 0.2, 0.3) rgb元组范围在0-1
     ='chartreuse' HTML颜色名
linestyle='-' 实线
         ='--'虚线
         ='-.'点划线
         =':'实点线
还可以把颜色和线形组合起来设定例如：'-g'绿色实线

# 调整坐标轴上下限
plt.xlim(-1, 11)
如果需要逆序显示
plot.xlim(11, -1)
用一行代码搞定,xy坐标轴限设定
plot.axis([-1, 11, 2, 3]) 注意不要与axes混淆

# 收紧坐标不留空白
plot.axis('tight')
图形分辨率1：1
plot.axis('equal')
更多请参观axis程序文档

# 设置图形标签
plt.title('') 标题
plt.xlabel('x')
plt.ylabel('y')

# 当图形内有多条线时，创建图例
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':c', label="cos(x)")
plt.legend() 用于创建图例，会将每条线的颜色和风格与标签匹配。

ax.set()方法一次性设置所有属性
ax.set(xlim=(0, 10), ylim=(-2, 2), xlabel='x', ylabel='y', title='')

# 散点图
plt.plot(x, y, 'o', color='k')
散点被线连接 plt.plot(x, y, '-ok')

rng = np.random.RandomState(0) 伪随机数生成必须从rng调用
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', ">", 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label="maker='{0}'".format(marker))
    plt.legend() 生成图例
    plt.xlim(0, 1.8)

# 设置线条和散点数
plt.plot(x, y, '-p', color='gray', 
         markersize=15, linewidth=4,
         markerfacecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2)  p代表五边形，marker代表p

# 用plt.scatter来画散点图，具有更高的灵活性，可以单独控制每散点
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = np.random.rand(100) 设置颜色灰度sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis') cmap把灰色映射到紫绿色系
plt.colorbar() 生成颜色条

from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T
plt.scatter(features[0], features[1], alpha=0.2, s=100*features[3], c=iris.target, cmap='viridis') 
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

数据量较大时plt.plot效率高于scatter
基本误差线
x = np.linespace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k')   fmt控制误差线外观, yerr控制误差

# 直方图
plt.hist(data, bins=30, normed=True, alpha=0.5, histstype='stepfilled', color='steelblue', edgecolor='none')
对不同分布特征样本进行对比时将histtype='stepfilled', 与透明设置alpha搭配使用
若果只想计算频次分布直方不想画图，可以用print(np.histogram(data, bins=5))这是个列表