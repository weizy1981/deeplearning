from pandas import read_csv
from matplotlib import pyplot as plt


filename = 'international-airline-passengers.csv'
footer = 3

# 导入数据
data = read_csv(filename, usecols=[1], engine='python', skipfooter=footer)
#图表展示
plt.plot(data)
plt.show()

# 查看最初的5条记录
print(data.head(5))
