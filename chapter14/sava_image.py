from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras import backend
import os

backend.set_image_data_format('channels_first')

# 导入数据
(X_train, y_train), (X_validation, y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')

# ZCA白化
imgGen = ImageDataGenerator(zca_whitening=True)
imgGen.fit(X_train)

# 创建目录，并保存图像
try:
    os.mkdir('image')
except:
    print('The fold is exist!')
for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9, save_to_dir='image', save_prefix='oct',
                                    save_format='png'):
    for i in range(0, 9):
        plt.subplot(331 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break
