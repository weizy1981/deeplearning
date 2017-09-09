from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 7
# 设定随机数种子
np.random.seed(seed)

# 导入数据
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# 分割输入x和输出Y
x = dataset[:, 0 : 8]
Y = dataset[:, 8]

kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
cvscores = []
for train, validation in kfold.split(x, Y):
    # 创建模型
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(x[train], Y[train], epochs=150, batch_size=10, verbose=0)

    # 评估模型
    scores = model.evaluate(x[validation], Y[validation], verbose=0)

    # 输出评估结果
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

# 输出均值和标准差
print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))
