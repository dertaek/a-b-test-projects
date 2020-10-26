import pandas as pd
import  matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np
import random 
sns.set()
data_before = pd.read_csv(r"C:\Users\studio\Desktop\入群之前.csv", encoding='gbk')
data_after = pd.read_csv(r'C:\Users\studio\Desktop\入群之后.csv', encoding='gbk')
mean = []

# 入群前后的接通电话次数双样本T检验
call_answer_num_before = data_before['call_answer_num'].values
call_answer_num_after = data_after['call_answer_num'].values
q1 = []
h1 = []
for i in range(500):
    q = np.random.choice(call_answer_num_before, 50)
    h = np.random.choice(call_answer_num_after, 50)
    q1.append(np.mean(q))
    h1.append(np.mean(h))
print(stats.levene(q1,h1))
t_and_p = stats.stats.ttest_ind(q1, h1, equal_var=False)
print('入群前后的接通电话次数双样本T检验:\n', t_and_p)
plt.hist(q1, bins=50)
plt.show()
a = np.mean(call_answer_num_before)
b = np.mean(call_answer_num_after)
mean.append(a)
mean.append(b)


# 入群前后的覆盖电话次数双样本T检验
cover_call_num_before = data_before['cover_call_num'].values 
cover_call_num_after = data_after['cover_call_num'].values
t_and_p2 = stats.stats.ttest_ind(cover_call_num_before, cover_call_num_after)
print('入群前后的覆盖电话次数双样本T检验:\n', t_and_p2)


# 入群前后的有效电话次数双样本T检验
effective_call_num_before = data_before['effective_call_num'].values 
effective_call_num_after = data_after['effective_call_num'].values
t_and_p3 = stats.stats.ttest_ind(effective_call_num_before, effective_call_num_after)
print('入群前后的有效电话次数双样本T检验:\n', t_and_p3)
a = np.mean(effective_call_num_before)
b = np.mean(effective_call_num_after)
mean.append(a)
mean.append(b)


# 入群前后的有效覆盖电话双样本T检验
effective_cover_call_num_before = data_before['effective_cover_call_num'].values 
effective_cover_call_num_after = data_after['effective_cover_call_num'].values
t_and_p4 = stats.stats.ttest_ind(effective_cover_call_num_before, effective_cover_call_num_after)
print('入群前后的有效覆盖电话双样本T检验:\n', t_and_p4)

# 入群前后的有效覆盖电话时长双样本T检验
effective_cover_call_time_before = data_before['effective_cover_call_time'].values 
effective_cover_call_time_after = data_after['effective_cover_call_time'].values
t_and_p5 = stats.stats.ttest_ind(effective_cover_call_time_before, effective_cover_call_time_after)
print('入群前后的有效覆盖电话时长双样本T检验:\n', t_and_p5)
a = np.mean(effective_cover_call_time_before)
b = np.mean(effective_cover_call_time_after)
mean.append(a)
mean.append(b)

# 入群前后的微信回复次数双样本T检验
wechat_answer_num_before = data_before['wechat_answer_num'].values 
wechat_answer_num_after = data_after['wechat_answer_num'].values
t_and_p6 = stats.stats.ttest_ind(wechat_answer_num_before, wechat_answer_num_after)
print('入群前后的微信回复次数双样本T检验:\n', t_and_p6)

# 入群前后的微信回复字数双样本T检验
wechat_answer_words_before = data_before['wechat_answer_words'].values 
wechat_answer_words_after = data_after['wechat_answer_words'].values
t_and_p7 = stats.stats.ttest_ind(wechat_answer_words_before, wechat_answer_words_after)
print('入群前后的微信回复字数双样本T检验:\n', t_and_p7)
a = np.mean(wechat_answer_words_before)
b = np.mean(wechat_answer_words_after)
mean.append(a)
mean.append(b)

# 入群前后的微信回复天数双样本T检验
wechat_answer_days_before = data_before['wechat_answer_days'].values 
wechat_answer_days_after = data_after['wechat_answer_days'].values
t_and_p9 = stats.stats.ttest_ind(wechat_answer_days_before, wechat_answer_days_after)
print('入群前后的微信回复天数双样本T检验:\n', t_and_p9)
a = np.mean(wechat_answer_days_before)
b = np.mean(wechat_answer_days_after)
mean.append(a)
mean.append(b)

# 入群前后的阅读次数双样本T检验
read_num_before = data_before['read_num'].values 
read_num_after = data_after['read_num'].values
t_and_p10 = stats.stats.ttest_ind(read_num_before, read_num_after)
print('入群前后的阅读次数双样本T检验:\n', t_and_p10)

# 入群前后的有效文章阅读次数双样本T检验
effective_read_num_before = data_before['effective_read_num'].values 
effective_read_num_after = data_after['effective_read_num'].values
q1 = []
h1 = []
for i in range(500):
    q = np.random.choice(effective_read_num_before, 500)
    h = np.random.choice(effective_read_num_after, 100)
    q1.append(np.mean(q))
    h1.append(np.mean(h))
print(stats.levene(q1, h1))
t_and_p11 = stats.stats.ttest_ind(q1, h1, equal_var=False)
print('入群前后的有效文章阅读次数双样本T检验:\n', t_and_p11)
plt.hist(q1, bins=50)
plt.show()
plt.hist(h1, bins=50)
plt.show()
a = np.mean(effective_read_num_before)
b = np.mean(effective_read_num_after)
mean.append(a)
mean.append(b)


# 入群前后的有效文章阅读时长双样本T检验
effective_read_time_before = data_before['effective_read_time'].values 
effective_read_time_after = data_after['effective_read_time'].values
t_and_p12 = stats.stats.ttest_ind(effective_read_time_before, effective_read_time_after)
print('入群前后的有效文章阅读时长双样本T检验:\n', t_and_p12)
a = np.mean(effective_read_time_before)
b = np.mean(effective_read_time_after)
mean.append(a)
mean.append(b)

# 入群前后的有效问卷次数双样本T检验
effective_wenjuan_num_before = data_before['effective_wenjuan_num'].values 
effective_wenjuan_num_after = data_after['effective_wenjuan_num'].values
t_and_p13 = stats.stats.ttest_ind(effective_wenjuan_num_before, effective_wenjuan_num_after)
print('入群前后的有效问卷次数双样本T检验:\n', t_and_p13)
a = np.mean(effective_wenjuan_num_before)
b = np.mean(effective_wenjuan_num_after)
mean.append(a)
mean.append(b)

# 入群前后的有效直播覆盖次数双样本T检验
effective_live_num_before = data_before['effective_live_num'].values 
effective_live_num_after = data_after['effective_live_num'].values
t_and_p14 = stats.stats.ttest_ind(effective_live_num_before, effective_live_num_after)
print('入群前后的有效直播覆盖次数双样本T检验:\n', t_and_p14)

# 入群前后的有效直播覆盖时长双样本T检验
effective_live_time_before = data_before['effective_live_time'].values 
effective_live_time_after = data_after['effective_live_time'].values
t_and_p15 = stats.stats.ttest_ind(effective_live_time_before, effective_live_time_after)
print('入群前后的有效直播覆盖时长双样本T检验:\n', t_and_p15)

# 入群前后的诺信总时间双样本T检验
naxion_time_before = data_before['naxion_time'].values 
naxion_time_after = data_after['naxion_time'].values
t_and_p16 = stats.stats.ttest_ind(naxion_time_before, naxion_time_after)
print('入群前后的诺信总时间双样本T检验:\n', t_and_p16)
a = np.mean(naxion_time_before)
b = np.mean(naxion_time_after)
mean.append(a)
mean.append(b)

# 入群前后的患者量双样本T检验
hzl_before = data_before['hzl'].values 
hzl_after = data_after['hzl'].values
t_and_p17 = stats.stats.ttest_ind(hzl_before, hzl_after)
print('入群前后的患者量双样本T检验:\n', t_and_p17)
a = np.mean(hzl_before)
b = np.mean(hzl_after)
mean.append(a)
mean.append(b)

# 入群前后几个活跃渠道双样本T检验
effective_channel_num_before = data_before['effective_channel_num'].values 
effective_channel_num_after = data_after['effective_channel_num'].values
t_and_p18 = stats.stats.ttest_ind(effective_channel_num_before, effective_channel_num_after)
print('入群前后几个活跃渠道双样本T检验:\n', t_and_p18)
x = np.arange(11)*5
bar_width = 0.8
x_label = ['call_answer_num', 'effective_call_num', 'effective_cover_call_time', 'wechat_answer_words', 'wechat_answer_days', 'effective_read_num',
 'effective_read_time', 'effective_wenjuan_num', 'naxion_time', 'hzl', 'effective_channel_num']
a = np.mean(effective_channel_num_before)
b = np.mean(effective_channel_num_after)
mean.append(a)
mean.append(b)
mean_array = np.array(mean)
mean_array = mean_array.reshape(11, 2)
plt.bar(x, height=mean_array[:, 0], color='#4EACC5', label='before', width=0.8)
plt.bar(x+bar_width, height=mean_array[:, 1], color='#FF9C43', label='after', width=0.8)
plt.xticks(x+bar_width/2, x_label)
plt.legend()
plt.show()

effective_cover_rate_before = np.divide(np.mean(effective_cover_call_num_before), np.mean(cover_call_num_before))
effective_cover_rate_after = np.divide(np.mean(effective_cover_call_num_after), np.mean(cover_call_num_after))
stats.mannwhitneyu(effective_cover_rate_before,effective_cover_rate_after, alternative='two-sided')