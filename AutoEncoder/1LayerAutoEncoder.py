# -*- coding:UTF-8 -*-
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning

# 参数
learning_rate = 0.01
training_epoch = 20
batch_size = 256
display_step = 1
example_show = 10

# 网络模型参数
n_hidden_units = 256  # 隐藏层输入神经元数量（编码器和解码器都有相同规模的数量）
n_input_units = 784  # 输入层神经元数量（ MNIST: 28x28）
n_output_units = n_input_units  # 解码器输出层必须等于输入数据的units数量


def WeightsVariable(n_in, n_out, name_str):
    return tf.Variable(
        tf.random_normal([n_in, n_out]), dtype=tf.float32, name=name_str)


def BiasesVariable(n_out, name_str):
    return tf.Variable(
        tf.random_normal([n_out]), dtype=tf.float32, name=name_str)


# 构建编码器
def Encoder(x_origin, activate_func=tf.nn.sigmoid):
    # 编码器hidden1
    with tf.name_scope('Layer'):
        weight = WeightsVariable(n_input_units, n_hidden_units, 'Weight')
        bias = BiasesVariable(n_hidden_units, 'Bias')
        x_code = activate_func(tf.add(tf.matmul(x_origin, weight), bias))
    return x_code


def Decoder(x_code, activate_func=tf.nn.sigmoid):
    # 解码器hidden1
    with tf.name_scope('Layer'):
        weight = WeightsVariable(n_hidden_units, n_output_units, 'Weight')
        bias = BiasesVariable(n_output_units, 'Bias')
        x_decode = activate_func(tf.add(tf.matmul(x_code, weight), bias))
    return x_decode


# 构建计算图
with tf.Graph().as_default():
    # 计算图输入
    with tf.name_scope('X_Origin'):
        X_Origin = tf.placeholder(tf.float32, [None, n_input_units])
    # 构建编码器模型
    with tf.name_scope('Encoder'):
        X_code = Encoder(X_Origin)
    # 构建解码器模型
    with tf.name_scope('Decoder'):
        X_decode = Decoder(X_code)
    # 因为我们希望重构的数据(X_decode)和原数据是一样的，所以我们用误差平方和损失
    with tf.name_scope('Loss'):
        Loss = tf.reduce_mean(tf.pow(X_Origin - X_decode, 2))
    # 训练节点
    with tf.name_scope('Train'):
        Optimizer = tf.train.RMSPropOptimizer(learning_rate)
        Train = Optimizer.minimize(Loss)

    # 初始化所有变量节点
    init = tf.global_variables_initializer()

    print('Logging...')
    summary_writer = tf.summary.FileWriter(
        logdir='logs', graph=tf.get_default_graph())
    summary_writer.flush()

    # 导入数据
    mnist = input_data.read_data_sets('../mnist_data/', one_hot=True)

    # 产生会话Session，启动计算图
    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(mnist.train.num_examples / batch_size)
        # 训练指定轮数，每一轮包含若干批次(total_batch)
        for epoch in range(training_epoch):
            # 每一轮（epoch）都要把所有的batch跑一遍
            for i in range(total_batch):
                # batch_ys 不用
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # 运行优化节点和Loss节点
                _, loss = sess.run([Train, Loss], feed_dict={
                                   X_Origin: batch_xs})
            if epoch % display_step == 0:
                print('epoch  {:d}  loss={:.9f}' .format(epoch, loss))

        # 关闭summary_writer
        summary_writer.close()
        print('Done.')

        # 把训练好的AutoEncoder模型用在测试集上，输出重建后的样本数据
        reconstructions = sess.run(
            X_decode, feed_dict={X_Origin: mnist.test.images[:example_show]})

        # 比较原始图像与重建后的图像
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(example_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(reconstructions[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()
