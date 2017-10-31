from nltk import word_tokenize
from gensim import corpora
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils

filename = 'Alice.txt'
document_split = ['.', ',', '?', '!', ';']
batch_size = 128
epochs = 20
model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'
dict_file = 'dict_file.txt'

def load_dataset():
    # 读入文件
    with open(file=filename, mode='r') as file:
        document = []
        documents = []
        lines = file.readlines()
        for line in lines:
            # 删除非内容字符
            value = clear_data(line)
            if value != '':
                # 对一行文本进行分词
                for str in word_tokenize(value):
                    # 跳过章节标题
                    if str == 'CHAPTER':
                        # documents.append(document)
                        # document = []
                        break
                    # 按照特定的标点符号分割成不同的句子
                    elif str in document_split:
                        document.append(str)
                        documents.append(document)
                        document = []
                    elif str != '':
                        document.append(str)

        return documents

def clear_data(str):
    # 删除字符串中的特殊字符或换行符
    value = str.replace('\ufeff', '').replace('\n', '')
    return value

def word_to_integer(documents):
    # 生产字典
    dic = corpora.Dictionary(documents)
    # 保存字典到文本文件
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    #print(dic)
    # 将单词转换为整数
    values = []
    max_len = 0
    for document in documents:
        value = []
        for word in document:
            # 查找每个单词在字典中的编码
            value.append(dic_set[word])
        values.append(value)
        if max_len < len(value):
            max_len = len(value)
    return values, max_len, len(dic_set)

def build_model(dict_len, words_len, class_num):
    model = Sequential()
    model.add(Embedding(input_dim=dict_len, output_dim=32, input_length=words_len))
    model.add(LSTM(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=class_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def make_y(values):
    y = []
    for value in values:
        y.append(value[0])
    # 将次行的第一个单词作为当行的输出
    y = np.array(y).reshape(len(y), 1)
    y = y[1:, :]
    return y

def make_x(values, max_len):
    # 补齐句子的长度
    dataset = sequence.pad_sequences(values, maxlen=max_len, value=-1)
    dataset = np.array(dataset)
    x = dataset[0: (dataset.shape[0] - 1), :]
    return x

if __name__ == '__main__':
    documents = load_dataset()
    values, max_len, dict_len = word_to_integer(documents)
    x_train = make_x(values, max_len)
    # 将数字调整到0-1之间
    x_train = x_train / float(dict_len)
    y_train = make_y(values)
    # one-hot编码
    y_train = np_utils.to_categorical(y_train)

    class_num = y_train.shape[1]

    model = build_model(dict_len=dict_len, words_len=max_len, class_num=class_num)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
    # 存储模型到Json文件
    model_json = model.to_json()
    with open(model_json_file, 'w') as file:
        file.write(model_json)
    # 保存权重数值到文件
    model.save_weights(model_hd5_file)