#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
# @Time    : 19-12-18 下午4:37

# @Author  : zhu fa

# @Software: PyCharm
"""
"""
tensorflow version must below 1.15
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 为什么shape参数也可以写成[,]的形式，原因可参考placeholder的定义
# x = tf.placeholder("float", [None, 784])
x = tf.placeholder("float", (None, 784))

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型

# 计算交叉熵 参考为啥交叉熵可以作为损失函数：https://blog.csdn.net/tsyccnh/article/details/79163834#t7
# 训练过程中使用了占位符，当feed_dict一次传进来100张时，损失函数是100张总的交叉商的和
y_ = tf.placeholder("float", shape=(None, 10))
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.initialize_all_variables()
# 启动我们的模型，并且初始化变量
sess = tf.Session()
sess.run(init)

# 让模型循环训练1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    '''
    f = open('2.txt','a')
    j=1
    for i in batch_ys[1]:
      f.write(str(i) + ' ')

    f.close()
    '''
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 验证模型准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print 'accuracy: ' + str(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 测试验证训练好的模型
X = np.loadtxt("1.txt")
WW = sess.run(W)
dot = np.dot(X, WW)
result = dot + sess.run(b)
Y = tf.nn.softmax(result)
print sess.run(Y)
# print sess.run(tf.nn.softmax(b)), sess.run(b), sess.run(b + W)
