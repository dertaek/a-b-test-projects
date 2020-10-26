# 状态解码。一直观测序列，求最有可能的状态序列。
# 最大路径概率
import numpy as np
from hmmlearn import hmm

status = ['box1', 'box2', 'box3']
observations = ['black', 'white', 'black']
start_probability = np.array([0.3, 0.5, 0.2])
transition_probability = np.array([[0.4, 0.4, 0.2],
                                  [0.3, 0.2, 0.5],
                                  [0.2, 0.6, 0.2]])
emission_probability = np.array([[0.2, 0.8],
                                [0.6, 0.4],
                                [0.4, 0.6]])
model = hmm.MultinomialHMM(n_components=len(status))
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability
observation_list = np.array([0, 1, 0])
# 对隐含状态进行解码。logprob是在观测序列出现概率最大的状态序列下出现该观测序列的概率的自然对数。最大概率是np.exp(logprob).
logprob, box_list = model.decode(observation_list.reshape(-1, 1), algorithm='viterbi') 
print(np.exp(logprob))
print(box_list)
for i in range(len(observation_list)):
    print(status[box_list[i]])