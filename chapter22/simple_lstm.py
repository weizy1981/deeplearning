from nltk import word_tokenize
from gensim import corpora
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np
from keras.utils import np_utils
from pyecharts import WordCloud

filename = 'Alice.txt'
document_split = ['.', ',', '?', '!', ';']
batch_size = 128
epochs = 200
model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'
dict_file = 'dict_file.txt'
dict_len = 2789
max_len = 20
document_max_len = 33200

def load_dataset():
    # 读入文件
    with open(file=filename, mode='r') as file:
        document = []
        lines = file.readlines()
        for line in lines:
            # 删除非内容字符
            value = clear_data(line)
            if value != '':
                # 对一行文本进行分词
                for str in word_tokenize(value):
                    # 跳过章节标题
                    if str == 'CHAPTER':
                        break
                    else:
                        document.append(str.lower())

        return document

def clear_data(str):
    # 删除字符串中的特殊字符或换行符
    value = str.replace('\ufeff', '').replace('\n', '')
    return value

def word_to_integer(document):
    # 生成字典
    dic = corpora.Dictionary([document])
    # 保存字典到文本文件
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    #print(dic)
    # 将单词转换为整数
    values = []
    for word in document:
        # 查找每个单词在字典中的编码
        values.append(dic_set[word])
    return values

def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=dict_len, output_dim=32, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=256))
    model.add(Dropout(0.2))
    model.add(Dense(units=dict_len, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def make_y(document):
    dataset = make_dataset(document)
    y = dataset[1:dataset.shape[0], 0]
    return y

def make_x(document):
    dataset = make_dataset(document)
    x = dataset[0: dataset.shape[0] - 1, :]
    return x

def make_dataset(document):
    dataset = np.array(document[0:document_max_len])
    dataset = dataset.reshape(int(document_max_len / max_len), max_len)
    return dataset

# 生成词云
def show_word_cloud(document):
    # 需要清楚的标点符号
    left_words = ['.', ',', '?', '!', ';', ':', '\'', '(', ')']
    # 生成字典
    dic = corpora.Dictionary([document])
    # 计算得到每个单词的使用频率
    words_set = dic.doc2bow(document)

    # 生成单词列表和使用频率列表
    words, frequences = [], []
    for item in words_set:
        key = item[0]
        frequence = item[1]
        word = dic.get(key=key)
        if word not in left_words:
            words.append(word)
            frequences.append(frequence)
    # 使用pyecharts生成词云
    word_cloud = WordCloud(width=1000, height=620)
    word_cloud.add(name='Alice\'s word cloud', attr=words, value=frequences, shape='circle', word_size_range=[20, 100])
    word_cloud.render()


if __name__ == '__main__':
    document = load_dataset()
    show_word_cloud(document)

    # 将单词转换为整数
    values = word_to_integer(document)
    x_train = make_x(values)
    y_train = make_y(values)
    # one-hot编码
    y_train = np_utils.to_categorical(y_train, dict_len)

    model = build_model()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
    # 存储模型到Json文件
    model_json = model.to_json()
    with open(model_json_file, 'w') as file:
        file.write(model_json)
    # 保存权重数值到文件
    model.save_weights(model_hd5_file)


