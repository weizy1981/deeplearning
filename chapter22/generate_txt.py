from nltk import word_tokenize
from gensim import corpora
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import sequence

filename = 'Alice.txt'
document_split = ['.', ',', '?', '!', ';']
batch_size = 128
epochs = 20
model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'
dict_file = 'dict_file.txt'
dict_len = 3123
max_len = 20
document_max_len = 33200

def load_dict():
    dic = corpora.Dictionary.load_from_text(dict_file)
    return dic

def load_model():
    # 从Json加载模型
    with open(model_json_file, 'r') as file:
        model_json = file.read()

    # 加载模型
    model = model_from_json(model_json)
    model.load_weights(model_hd5_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def word_to_integer(document):
    # 生成字典
    dic = load_dict()
    dic_set = dic.token2id
    # 将单词转换为整数
    values = []
    for word in document:
        # 查找每个单词在字典中的编码
        values.append(dic_set[word])
    return values

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

if __name__ == '__main__':
    model = load_model()
    document = 'Alice is a little girl'
    values = word_to_integer(document)
    print(values)

