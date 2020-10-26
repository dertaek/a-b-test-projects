import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

x = [1, 2, 3, 4, 5]
height = [0.633, 2.329, 2.41, 4.327, 0.774]
colors = ['#4EACC5', '#FF9C43', '#4E9A06', '#8A2BE2', '#00FF7F']
labels = ['6:00-10:00' ,'10:00-13:30' , '13:30-17:00', '17:00-19:30', '19:30-']
plt.bar(x, height=height, color=colors, label=labels)
plt.legend()
plt.show()