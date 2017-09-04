# -*- coding:UTF-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning


with tf.Graph().as_default():
    # Input:样本特征向量及其真实标签集合的占位符，Y_: One-Hot存储，所以是10
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
        Y_ = tf.placeholder(tf.float32, shape=[None, 10], name='Y')

    # Inference：前向预测
    with tf.name_scope('Inference'):
        # 权重和偏置
        W = tf.Variable(tf.zeros([784, 10]), name='Weight')
        b = tf.Variable(tf.zeros([10]), name='Bias')
        Ypred = tf.add(tf.matmul(X, W), b)

    # SoftMax：把Ypred变成概率分布，就一行
    with tf.name_scope('SoftMax'):
        Prob_pred = tf.nn.softmax(logits=Ypred)

    # 计算交叉熵损失
    with tf.name_scope('EntropyLoss'):
        xentropy_loss = -tf.reduce_sum(Y_ * tf.log(Prob_pred), axis=1)
        Loss = tf.reduce_mean(xentropy_loss)

    # 使用梯度下降， 学习率：0.5
    with tf.name_scope('GradientDescent'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(Loss)

    # 定义评估节点
    with tf.name_scope('Evaluate'):
        correct_prediction = tf.equal(tf.argmax(Ypred, 1), tf.argmax(Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化变量
    InitOp = tf.global_variables_initializer()

    # 保存计算图，目录、文件夹名称不要有空格
    wirter = tf.summary.FileWriter('SoftMaxLog', graph=tf.get_default_graph())
    wirter.close()

    # 加载数据集
    mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

    # 开启会话(Session)
    sess = tf.InteractiveSession()

    # 初始化所有变量
    sess.run(InitOp)

    # 训练模型，训练1000次，每次抓取100个样本
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, train_loss = sess.run(
            [train_step, Loss], feed_dict={X: batch_xs, Y_: batch_ys})
        print 'Train step:', step + 1, 'Trian loss:', train_loss

    # 计算精确度
    accuracy_score = sess.run(
        accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
    print 'Accuracy:', accuracy_score
