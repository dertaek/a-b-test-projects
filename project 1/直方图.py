import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
sns.set()

data_set1 = pd.read_csv(r'd:\py\.vscode\类0.csv', header=0)
data_set2 = pd.read_csv(r'd:\py\.vscode\类1.csv', header=0)
data_set3 = pd.read_csv(r'd:\py\.vscode\类2.csv', header=0)
data_set4 = pd.read_csv(r'd:\py\.vscode\类3.csv', header=0)
data_set5 = pd.read_csv(r'd:\py\.vscode\类4.csv', header=0)
x = data_set1.iloc[:, [7]].values
x2 = data_set2.iloc[:, [7]].values
x3 = data_set3.iloc[:, [7]].values
x4 = data_set4.iloc[:, [7]].values
x5 = data_set5.iloc[:, [7]].values
plt.hist(x, bins=30, normed=True, alpha=0.4, histtype='stepfilled', color='#4EACC5',  label='6:00-10:00')
plt.hist(x2, bins=30, normed=True, alpha=0.4, histtype='stepfilled', color='#FF9C43', label='10:00-13:30')
plt.hist(x3, bins=30, normed=True, alpha=0.4, histtype='stepfilled', color='#4E9A06', label='13:30-17:00')
plt.hist(x4, bins=30, normed=True, alpha=0.4, histtype='stepfilled', color='#8A2BE2', label='17:00-19:30')
plt.hist(x5, bins=30, normed=True, alpha=0.4, histtype='stepfilled', color='#00FF7F', label='19:30-')
plt.legend()
plt.gca().set_xlim(0,800)
plt.show()