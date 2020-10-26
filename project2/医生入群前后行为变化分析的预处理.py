import pandas as pd
import time
data = pd.read_excel(r'C:\Users\studio\Desktop\KEPPRA_ORAL.xlsx', sheet_name='Sheet2', header=0)
filter_ = pd.read_excel(r'C:\Users\studio\Desktop\KEPPRA_ORAL.xlsx', sheet_name='Sheet1', header=0)
data_filtered = pd.DataFrame(columns=data.columns)
data_before = pd.DataFrame(columns=data.columns)
data_after = pd.DataFrame(columns=data.columns)
for i in range(len(filter_)):
    a = data[data['doctor_id'] == filter_['doctor_id'][i]]
    # 必须要data = data.apped(a)
    data_filtered = data_filtered.append(a)
# 必须建立索引
data_filtered = data_filtered.reset_index(drop=False)
time0 = []
time1 = []
for i in range(len(data_filtered['data_date'])):
    time0.append(str(data_filtered['data_date'][i]))
for i in range(len(time0)):
    time1.append(time.mktime(time.strptime(time0[i], r'%Y-%m-%d %H:%M:%S')))
data_filtered.insert(38, 'time', time1)
time2 = []
time3 = []
for i in range(len(filter_['入群时间'])):
    time2.append(str(filter_['入群时间'][i]))
for i in range(len(time2)):
    time3.append(time.mktime(time.strptime(time2[i], r'%Y-%m-%d %H:%M:%S')))
for i in range(len(data_filtered)):
    for j in range(len(time3)):
        if (data_filtered['doctor_id'][i] == filter_['doctor_id'][j]) and (data_filtered['time'][i] <= time3[j]):
            data_before = data_before.append(data_filtered.iloc[i])
            break
        elif (data_filtered['doctor_id'][i] == filter_['doctor_id'][j]) and (data_filtered['time'][i] > time3[j]):
            data_after = data_after.append(data_filtered.iloc[i])
            break
data_before.to_csv(r'C:\Users\studio\Desktop\入群之前.csv', encoding='gbk')
data_after.to_csv(r'C:\Users\studio\Desktop\入群之后.csv', encoding='gbk')