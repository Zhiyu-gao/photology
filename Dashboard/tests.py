from django.test import TestCase

# Create your tests here.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 定义模型
model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

# 获取权重和偏置
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=10)  # 你需要训练模型才能得到权重和偏置

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

# 获取权重和偏置
w, b = model.get_weights()

# 绘图
plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()
