# -*- coding:UTF-8 -*-
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning


# Xavier Initialization 均匀初始化. fan_in是输入节点的数量，fan_out是输出节点的数量
def xavier_init(fan_in, fan_out, constant=1):
    # 均匀分布的方差计算：D(x) = (max - min) ** 2 / 12 = 2 / (fan_in + fan_out)
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high, dtype=tf.float32)


# 定义一个降噪自编码器的类 -- 加性高斯噪声的自动编码器
class AdditiveGaussianNoiseAutoencoder(object):
    # 在 __init__ 中构造计算图
    # n_input: 输入变量数，n_hidden: 隐藏层节点数，transfer_function: 隐藏层激活函数
    # optimizer: 优化器，scale: 高斯噪声系数

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.training_scale = scale

        # 建立一个字典类型的权重集：
        # self.weights['W1'], self.weights['b1']
        # self.weights['W2'], self.weights['b2']
        self.weights = dict()

        with tf.name_scope('RawInput'):
            self.x = tf.placeholder(tf.float32, [None, n_input])

        # 先对输入x添加噪声，即self.x + scale * tf.random_normal((n_input,))
        with tf.name_scope('NoiseAdder'):
            self.scale = tf.placeholder(tf.float32)
            self.noise_x = self.x + self.scale * tf.random_normal((n_input,))

        # 再用tf.matmul()让被噪声污染的信号与隐藏层的权重相乘，用tf.add()添加偏置
        # 最后用transfer()对加权汇总结果进行激活函数处理
        with tf.name_scope('Encoder'):
            # 权重W1
            self.weights['W1'] = tf.Variable(
                xavier_init(self.n_input, self.n_hidden), name='weight1')
            # 偏置b1
            self.weights['b1'] = tf.Variable(
                tf.zeros([self.n_hidden]), dtype=tf.float32, name='bias1')
            # 加噪声
            self.hidden = self.transfer(
                tf.add(tf.matmul(self.noise_x,
                                 self.weights['W1']), self.weights['b1']))

        # reconstruction 重构节点：经过隐藏层后，在输出层进行数据复原和重建操作，不需要激活函数
        with tf.name_scope('Reconstruction'):
            # 权重W2
            self.weights['W2'] = tf.Variable(
                tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32),
                name='weight2')
            # 偏置b2
            self.weights['b2'] = tf.Variable(
                tf.zeros([self.n_input], dtype=tf.float32), name='bias2')
            # 重构reconstruction
            self.reconstruction = tf.add(
                tf.matmul(self.hidden, self.weights['W2']), self.weights['b2'])

        # 自编码器的损失函数：平方误差损失（重建信号和原始信号的误差平方和）
        with tf.name_scope('Loss'):
            self.cost = 0.5 * tf.reduce_sum(
                tf.pow(tf.subtract(self.reconstruction, self.x), 2))

        with tf.name_scope('Training'):
            self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('run...')

    # 在一个批次上训练模型
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={
                                  self.x: X, self.scale: self.training_scale})
        return cost

    # 在给定样本集合上计算损失(用于测试阶段)
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={
                             self.x: X, self.scale: self.training_scale})

    # 返回自编码器隐藏层的输出结果，获得抽象后的高阶特征表示
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={
                             self.x: X, self.scale: self.training_scale})

    # 将隐藏层的高阶特征作为输入，将其重建为原始输入数据
    def generate(self, hidden=None):
        if hidden == None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={
                             self.hidden: hidden})

    # 整体运行一遍复原过程，包括提取高阶特征以及重建原始数据，输入原始数据，输出复原后的数据
    def reconstructioin(self, X):
        return self.sess.run(self.reconstructioin, feed_dict={
            self.x: X, self.scale: self.training_scale})

    # 获取隐藏层的权重
    def getWeights(self):
        return self.sess.run(self.weights['W1'])

    # 获取隐藏层的偏置
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


AGN_AC = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200,
                                          transfer_function=tf.nn.softplus,
                                          optimizer=tf.train.AdamOptimizer(
                                              learning_rate=0.01),
                                          scale=0.01)
print('Logging...')
writer = tf.summary.FileWriter(logdir='DA_logs', graph=AGN_AC.sess.graph)
writer.close()

# 读取数据集
mnist = input_data.read_data_sets('mnist_data', one_hot=True)


# 使用sklearn.preprocess的数据标准化操作（0均值，标准差为1的高斯分布）预处理数据
# 首先在训练集上估计均值与方差，然后将其作用到训练集和测试集
def standard_scale(X_train, X_test):
    '''
    sklearn.preprocessing 具体解释：
    http://scikit-learn.org/stable/modules/preprocessing.html
    http://blog.csdn.net/sinat_33761963/article/details/53433799
    '''
    perprocessor = prep.StandardScaler().fit(X_train)
    X_train = perprocessor.transform(X_train)
    X_test = perprocessor.transform(X_test)
    return X_train, X_test


# 获取随机block数据的函数: 取一个从0到len(data) - batch_size的随机整数
# 以这个随机整数为起始索引，抽出一个batch_size的批次样本
def get_random_block_from_data(data, batch_size):
    # 从下面两行代码可以看出这是有放回抽样
    # len(data) - batch_size 是为了防止索引下标益处，也就是star_index + batch_size > len(data)
    star_index = np.random.randint(0, len(data) - batch_size)
    return data[star_index: (star_index + batch_size)]


# 使用标准化操作变换数据集
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 定义训练参数
n_samples = int(mnist.train.num_examples)  # 训练样本的总数
training_epochs = 20  # 最大训练轮数，每n_samples / batch_size个批次为1轮
batch_size = 128  # 每个批次的样本数量
display_step = 1  # 输出训练结果的间隔

# 开始训练。每一轮epoch训练开始时，将平均损失avg_cost重置为0
# 计算总共需要的batch数量(样本总数／batch_size). 这里使用有放回抽样，所以不能保证每一个样本都被抽到
# 并参与训练
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    # 在每个batch的循环中，随机抽取一个batch的数据，使成员函数
    # partial_fit训练这个数据，计算cost，累积到当前回合的平均cost中
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = AGN_AC.partial_fit(batch_xs)
        avg_cost += cost / batch_size
    avg_cost /= total_batch

    if epoch % display_step == 0:
        print('epoch:%04d, cost = %.9f' % (epoch + 1, avg_cost))

# 计算测试集上的cost
print('Total cost:', str(AGN_AC.calc_total_cost(X_test)))
