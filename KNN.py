# -*- coding:UTF-8 -*-
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning

# 导入MNIST数据集
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# 对MNIST数据集做一个数量限制
Xtrain, Ytrain = mnist.train.next_batch(5000)  # 5000用于训练（nn candidates)
Xtest, Ytest = mnist.test.next_batch(200)  # 200用于测试

print "Xtrain.shape:", Xtrain.shape, "Xtest.shape:", Xtest.shape
print "Ytrain.shape", Ytrain.shape, "Ytest.shape", Ytest.shape

# 计算图输入占位符
xtrain = tf.placeholder("float", [None, 784])
xtest = tf.placeholder("float", [784])

# 使用L1距离进行最近邻计算
distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), axis=1)

# 预测：获得最小距离的索引（根据最近邻的类标签进行判断）
pred = tf.arg_min(distance, 0)

# 初始化节点
init = tf.global_variables_initializer()
accuracy = 0.0

# 启动会话：
with tf.Session() as sess:
    sess.run(init)
    Ntest = len(Xtest)  # 获取测试样本数量
    for i in range(Ntest):
        # 获取当前测试样本的最近邻
        nn_index = sess.run(
            pred, feed_dict={xtrain: Xtrain, xtest: Xtest[i, :]})

        # 获得最近邻预测标签，并于真实标签进行比较
        pred_class_label = np.argmax(Ytrain[nn_index])
        true_class_label = np.argmax(Ytest[i])
        print "Test", i, "Predicted Class Label:", pred_class_label,
        print "True CLass Label", true_class_label

        # 计算准确率
        if pred_class_label == true_class_label:
            accuracy += 1
    print "Done!"
    accuracy /= Ntest
    print "Accuracy:", accuracy
