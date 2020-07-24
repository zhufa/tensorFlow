#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

# 模型加载与预测 参考自https://www.cnblogs.com/USTC-ZCC/p/11249625.html和https://blog.csdn.net/roger_royer/article/details/86520235

"""
# @Time    : 20-7-9 下午2:54

# @Author  : zhufa

# @Software: PyCharm
"""
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    # 加载元图和权重
    saver = tf.train.import_meta_graph('model/mnist-model-10000.meta')
    saver.restore(sess, tf.train.latest_checkpoint("model/"))
    graph = tf.get_default_graph()  # 获取当前默认计算图

    '''
    # 获取权重
    W_conv1 = graph.get_tensor_by_name("W_conv1:0")  # get_tensor_by_name后面传入的参数，如果没有重复，需要在后面加上“:0”
    b_conv1 = graph.get_tensor_by_name("b_conv1:0")  # 不知道name，或者name报错的，去原训练代码文件里调试或者输出你要的参数，看看他的name属性

    print ("------------------------------------------------------")
    print ('fc2_w:',sess.run(W_conv1))  # 可以打印查看，这里因为数据太多了，显示太占地方了，就不打印了
    print ("#######################################")
    print ('fc2_b:',sess.run(b_conv1))
    print ("------------------------------------------------------")
    '''

    b = np.loadtxt("test/txt/66.txt")
    x = b.reshape([1, 784])

    feed_dict = {"input_x:0": x, "keep_prob:0": 1.0}

    y_ = graph.get_tensor_by_name("Softmax:0")

    yy = sess.run(y_, feed_dict)  # 将Y转为one-hot类型
    print ('yy:', yy)
    print ("the answer is: ", sess.run(tf.argmax(yy, 1)[0]))
    print ("------------------------------------------------------")
