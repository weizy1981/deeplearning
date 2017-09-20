from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
# 导入数据
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# Convert labels to categorical one-hot encoding
Y_labels = keras.utils.to_categorical(Y, num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)
# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# 构建模型
model = create_model()
model.fit(x, Y_labels, epochs=200, batch_size=5, verbose=0)

scores = model.evaluate(x, Y_labels, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))