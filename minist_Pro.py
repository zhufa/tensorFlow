#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
# @Time    : 20-6-9 上午11:47

# @Author  : zhufa

# @Software: PyCharm
"""
"""
tensorflow version must below 1.15
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 返回带初始值的权重变量
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


# 返回带初始值的偏置变量
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积:x为输入图像，W为卷积核，padding补0，步长为1，图像尺寸不变
def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')


# 池化:
def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')


# 第一层:卷积+池化
W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")  # 5×5的卷积核，出入为1通道的图，输出为32，即32个卷积核
b_conv1 = bias_variable([32], "b_conv1")

x = tf.placeholder("float", (None, 784), name='input_x')
# 将x转化为28×28的图像矩阵/向量/张量，-1表示视原x的情况计算而出，最后的1表示通道，若为彩色图像则为3
reshaped_x = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(reshaped_x, W_conv1) + b_conv1)  # 卷积后的图像尺寸不变
h_pool1 = max_pool_2x2(h_conv1)  # 池化后的图像尺寸为14×14

# 第二层卷积+池化
W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 卷积后的图像尺寸不变
h_pool2 = max_pool_2x2(h_conv2)  # 池化后的图像尺寸为7×7

# 密集连接层：1024个神经元（全连接）
W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")

reshaped_h_pool2 = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(reshaped_h_pool2, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float", name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10], "W_fc2")
b_fc2 = bias_variable([10], "b_fc2")
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数及训练模型
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 准备训练，初始化所有变量
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# 测试模型准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

'''设置模型保存器'''
m_saver = tf.train.Saver()

# 训练20000次，每次随机抓取100对测试样本，每100次输出当前的准确率
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        step_accuracy = accuracy.eval(session=sess,
                                      feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        print "step %d test accuracy: %g" % (i, step_accuracy)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

# 保存模型参数，如何加载模型并使用参见 mnist_test.py
# m_saver.save(sess, "model/mnist-model", global_step=10000)

print "test accuracy: %g" % accuracy.eval(session=sess,
                                          feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
