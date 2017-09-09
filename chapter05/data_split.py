from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

seed = 7
# 设定随机数种子
np.random.seed(seed)

# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# 分割输入x和输出Y
x = dataset[:, 0 : 8]
Y = dataset[:, 8]

# 分割数据集
x_train, x_validation, Y_train, Y_validation = train_test_split(x, Y, test_size=0.2, random_state=seed)

# 构建模型
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(x_train, Y_train, validation_data=(x_validation, Y_validation), epochs=150, batch_size=10)