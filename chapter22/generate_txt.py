from nltk import word_tokenize
from gensim import corpora
from keras.models import model_from_json
import numpy as np

model_json_file = 'simple_model.json'
model_hd5_file = 'simple_model.hd5'
dict_file = 'dict_file.txt'
words = 200
max_len = 20
myfile = 'myfile.txt'

def load_dict():
    # 从文本导入字典
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
    # model.summary()
    return model

def word_to_integer(document):
    # 导入字典
    dic = load_dict()
    dic_set = dic.token2id
    # 将单词转换为整数
    values = []
    for word in document:
        # 查找每个单词在字典中的编码
        values.append(dic_set[word])
    return values

def make_dataset(document):
    dataset = np.array(document)
    dataset = dataset.reshape(1, max_len)
    return dataset

def reverse_document(values):
    # 导入字典
    dic = load_dict()
    dic_set = dic.token2id
    # 将编码转换为单词
    document = ''
    for value in values:
        word = dic.get(value)
        document = document + word + ' '

    return document
if __name__ == '__main__':
    model = load_model()
    start_doc = 'Alice is a little girl, who has a dream to go to visit the land in the time.'
    document = word_tokenize(start_doc.lower())
    new_document = []
    values = word_to_integer(document)
    new_document = [] + values


    for i in range(words):
        x = make_dataset(values)
        # prediction = model.predict_classes(x, verbose=0)[0]
        prediction = model.predict(x, verbose=0)
        prediction = np.argmax(prediction)
        values.append(prediction)
        new_document.append(prediction)
        values = values[1: ]

    new_document = reverse_document(new_document)
    with open(myfile, 'w') as file:
        file.write(new_document)

