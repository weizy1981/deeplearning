from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=5000)

x_train = sequence.pad_sequences(x_train, maxlen=500)
x_validation = sequence.pad_sequences(x_validation, maxlen=500)

Embedding(5000, 32, input_length=500)
