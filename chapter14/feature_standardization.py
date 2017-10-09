from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras import backend

backend.set_image_data_format('channels_first')

# 导入数据
(X_train, y_train), (X_validation, y_validation) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_validation = X_validation.reshape(X_validation.shape[0], 1, 28, 28).astype('float32')

# 图像特征化
imgGen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
imgGen.fit(X_train)

for X_batch, y_batch in imgGen.flow(X_train, y_train, batch_size=9):
    for i in range(0, 9):
        plt.subplot(331 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()
    break