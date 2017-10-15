from keras.datasets import imdb
import numpy as np
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Flatten
from keras.models import Sequential

seed = 7
top_words = 5000
max_words = 500
out_dimension = 32
batch_size = 128
epochs = 2

def create_model():
    model = Sequential()
    # 构建嵌入层
    model.add(Embedding(top_words, out_dimension, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    np.random.seed(seed=seed)
    # 导入数据
    (x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=top_words)
    # 限定数据集的长度
    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_validation = sequence.pad_sequences(x_validation, maxlen=max_words)

    # 生成模型
    model = create_model()
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
              batch_size=batch_size, epochs=epochs, verbose=2)