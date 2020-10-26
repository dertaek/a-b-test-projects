import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = [152, 489, 506, 649, 209]
colors = ['#4EACC5', '#FF9C43', '#4E9A06', '#8A2BE2', '#00FF7F']
labels = ['6:00-10:00' ,'10:00-13:30' , '13:30-17:00', '17:00-19:30', '19:30-']
plt.pie(data, colors=colors, labels=labels, labeldistance = 1.2, autopct = '%3.2f%%', pctdistance = 0.6)
plt.legend('perfect')
plt.axis('equal')
plt.show()