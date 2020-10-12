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
one = [1 for i in range(len(time_list5))]
data_set.insert(7, 'one', one)
del data_set['message_time']
X = np.array(time_list5)
X_nor = (X- X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
data_set.insert(5, 'message_time', X_nor)

# 画图模块
X = data_set.iloc[:, [5, -1]].values.astype(np.float64)
d = sch.distance.pdist(X, 'euclidean').astype(np.float64)
Z = sch.linkage(d, method='average')
sch.dendrogram(Z)
plt.gca().set_ylim(0, 0.5)
plt.show()

# 选取阈值模块
cluster= sch.fcluster(Z, t=1.1547005, criterion='inconsistent')
cluster_set = list(zip(cluster, range(len(cluster))))
cluster_set.sort(key = lambda x : x[0])

# 输出类列表模块
for i in range(1, (cluster_set[-1][0] + 1)):
    class_i = []
    for j in range(len(cluster_set)):
        if cluster_set[j][0] == i:
            class_i.append(cluster_set[j][1])
    print(f'聚类{i}:\n', len(class_i))
    df_i = pd.DataFrame(data=None, columns=['id', 'product_id', 'drug_user_id', 'wechat_number', 'wechat_message_type', 'message_time', 'len'])
    for k in range(len(class_i)):
        a = class_i[k]
        df_i.loc[a] = data_set.loc[a]
    print(df_i)
    df_i.to_csv(f'{len(df_i)}.csv', encoding='UTF-8')