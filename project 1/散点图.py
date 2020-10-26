import xlwt
import xlrd
import scipy
# 聚类包
import scipy.cluster.hierarchy as sch
# 画图包
import matplotlib.pylab as plt
# 数组处理包
import numpy as np
# 数据导入包
import pandas as pd
# 时间处理包
import time 

# 读取模块
data_set = pd.read_excel(r'C:\Users\studio\Desktop\渠道信息.xlsx', header=0)

# 时间数据预处理模块
time_list = []
time_list2 = []
time_list3 = []
time_list4 = []
for k in range(len(data_set)):
    time_list.append(str(data_set['message_time'][k]))
for i in range(len(time_list)):
    time_list2.append(time.strptime(time_list[i], r'%Y-%m-%d %H:%M:%S'))
for i in range(len(time_list2)):
    time_list3.append(time.strftime('%H:%M:%S', time_list2[i]))
for j in range(len(time_list3)):
    time_list4.append((time_list3[j].split(':')))
b = [3600, 60, 1]
time_list5 = []
for i in range(len(time_list4)):
    f = []
    for j in range(3):
        f.append(int(time_list4[i][j]) * b[j])
    time_list5.append(f[0] + f[1] + f[2])


# 数据框修改模块
del data_set['message_time']
X = np.array(time_list5)
X_nor = (X-0) / 86400
data_set.insert(5, 'message_time', X_nor)

# 画图模块
X = data_set.iloc[:, [5, -1]].values.astype(np.float64)
a = X[:, 0]
plt.scatter(X[:, 0], X[:, 1], alpha=0.3,
            s=0.7*X[:, 1], cmap='viridis')
plt.xlabel('time')
plt.ylabel('reply numbers')
time_points = ['00:00', '04:00', '08:00', '12:00', '16:00',
                '20:00', '24:00']
plt.xticks([0, 0.16667, 0.33334, 0.5, 0.66667, 0.833333, 1], time_points)
plt.xlim(0, 1)
plt.show()