import xlwt
import xlrd
import scipy
# 聚类包
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
# 画图包
import matplotlib.pylab as plt
# 数组处理包
import numpy as np
# 数据导入包
import pandas as pd
# 时间处理包
import time
import seaborn as sns
sns.set()

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
one = [1 for i in range(len(time_list5))]
data_set.insert(7, 'one', one)
del data_set['message_time']
X = np.array(time_list5)
X_nor = (X- X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
data_set.insert(5, 'message_time', X_nor)

# 画图模块
X = data_set.iloc[:, [5, -1]].values.astype(np.float64)
n_cluster = 5
cls = KMeans(n_cluster).fit(X)
k_means_cluster_centers = np.sort(cls.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
colors = ['#4EACC5', '#FF9C43', '#4E9A06', '#8A2BE2', '#00FF7F']
for k, col in zip(range(n_cluster), colors):
    set_k = []
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
    for i in range(len(my_members)):
        if my_members[i] == True:
            set_k.append(i)
    df_i = pd.DataFrame(data=None, columns=['id', 'product_id', 'drug_user_id', 'wechat_number', 'wechat_message_type', 'message_time', 'len'])
    for i in range(len(set_k)):
        a = set_k[i]
        df_i.loc[a] = data_set.loc[a]
    df_i.to_csv(f'类{k}.csv', encoding='UTF-8')
plt.title('K-means')
time_points = ['00:00', '04:00', '08:00', '12:00', '16:00',
                '20:00', '24:00']
plt.xticks([0, 0.16667, 0.33334, 0.5, 0.66667, 0.833333, 1], time_points)
plt.xlabel('time')
plt.show()