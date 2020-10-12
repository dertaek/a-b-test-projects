import matplotlib.pylab as plt
import pandas as pd
data_set2 = pd.read_csv(r'd:\py\114.csv', header=0)
x = data_set2.iloc[:, [2]].values
b = []
for i in range(len(x)):
    b.append(int(x[i]))
nu = len(set(b))
plt.hist(x, bins=nu, color='b', normed=True, alpha=0.5)
plt.show()