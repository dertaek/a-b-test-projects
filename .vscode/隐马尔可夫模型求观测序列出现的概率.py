# HMM 模型。可以应用于语音识别、词性自动标注等自然语言处理。
# 观测随机序列的明线，和隐藏着状态序列的暗线。我们只能通过随机序列去推断状态序列。
# 实例：盒子摸球实验
# 实例：婴儿的日常
# 核心三要素：状态转移矩阵、观测矩阵、初始隐含状态概率向量。并且状态转移矩阵和初始状态向量就已经决定了HMM
# t时刻隐状态至于前一时刻隐状态有关，t时刻观测只于该时刻隐状态有关
# 研究内容：1.观测序列概率估计——给定模型的情况下求观测序列出现的概率 2.隐含状态序列解码——根据观测序列解码出状态序列。

import numpy as np
from hmmlearn import hmm

# 隐状态集合
states = ['box1', 'box2', 'box3']
# 观测集合
observasations  = ['black', 'white', 'yellow']
# 初始概率向量
start_probability = np.array([0.3, 0.5, 0.2])
# 状态转移矩阵
transition_probability = np.array([[0.4, 0.4, 0.2],
                                 [0.3, 0.2, 0.5],
                                 [0.2, 0.6, 0.2]])
# 观测矩阵
emission_probability  = np.array([[0.2, 0.7, 0.1],
                                [0.5, 0.1, 0.4],
                                [0.4, 0.1, 0.5]])
# 选用multinomialHMM 对离散观测状态建模
model = hmm.MultinomialHMM(n_components=len(states))
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# 观测序列本质上是自变量
observation_list = np.array([2, 1, 0, 2, 1]) 
# a = obervation_list[:, np.newaxis] 与observation[:, neewaxis]等价，变形成列向量。
# 计算观测序列的概率,model.score不是概率而是概率值的自然对数
print(np.exp(model.score(observation_list.reshape(-1, 1))))