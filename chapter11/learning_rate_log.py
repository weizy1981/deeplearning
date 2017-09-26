from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from math import pow, floor


# 导入数据
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# 设定随机种子
seed = 7
np.random.seed(seed)

# 计算学习率
def step_decay(epoch):
    init_lrate = 0.1
    drop = 0.5
    epochs_drop = 10
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    return lrate

# 构建模型函数
def create_model(init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    #模型优化
    learningRate = 0.1
    momentum = 0.9
    decay_rate = 0.0
    sgd = SGD(lr=learningRate, momentum=momentum, decay=decay_rate, nesterov=False)

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


lrate = LearningRateScheduler(step_decay)
epochs = 200
model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=5, verbose=1, callbacks=[lrate])
model.fit(x, Y)