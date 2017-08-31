import tensorflow as tf

# 声明两个占位符
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 定义表达式
add = tf.add(a, b)

# 执行运算
session = tf.Session()
binding = {a : 1.5, b : 2.5}
c = session.run(add, feed_dict=binding)
print(c)