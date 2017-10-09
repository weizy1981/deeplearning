from keras.datasets import mnist
from matplotlib import pyplot as plt

# 从Keras导入Mnist数据集
(X_train, y_train), (X_validation, y_validation) = mnist.load_data()

# 显示9张手写数字的图片
for i in range(0, 9):
    plt.subplot(331 + i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

plt.show()