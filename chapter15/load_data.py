from keras.datasets import cifar10
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np

# 导入数据
(X_train, y_train), (X_validation, y_validation) = cifar10.load_data()

for i in range(0, 9):
    plt.subplot(331 + i)
    plt.imshow(toimage(X_train[i]))

# 显示图片
plt.show()

# 设定随机种子
seed = 7
np.random.seed(seed)