# -*- coding:UTF-8 -*- 
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning

# 产生训练数据集
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182,
                     7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366,
                     2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_train_samples = train_X.shape[0]
print '训练样本数量: ', n_train_samples

# 产生测试样本
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
n_test_samples = test_X.shape[0]
print '测试样本数量: ', n_test_samples

with tf.Graph().as_default():
    # 设计一个名称域‘Input’
    with tf.name_scope('Input'):
        # 设置输入占位符并起个名字‘Input/X’ ‘Input/Y’
        X = tf.placeholder(tf.float32, name='X')
        Y = tf.placeholder(tf.float32, name='Y')

    with tf.name_scope('Inference'):
        # 设置模型参数变量并起个名字
        W = tf.Variable(tf.zeros([1]), name='Weight')
        b = tf.Variable(tf.zeros([1]), name='Bias')

        # inference: y = wx + b
        Ypred = tf.add(tf.multiply(X, W), b)

    with tf.name_scope('Loss'):
        # 计算损失
        Loss = tf.reduce_mean(tf.pow((Y - Ypred), 2)) / 2

    with tf.name_scope('Train'):
        # 创建梯度下降优化器
        Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

        # Train：定义训练节点将梯度下降法应用于Loss
        # 在优化器内部对权重W和b进行更新，在TensorBoard中可以看到
        # 路径：GradientDescent图中update_Variable里面的ApplyGradientDescent
        TrainOp = Optimizer.minimize(Loss)

    with tf.name_scope('Evaluate'):
        # 添加评估节点
        evaloss = tf.reduce_mean(tf.pow((Y - Ypred), 2)) / 2

    # 初始化Variable类型变量节点
    init = tf.global_variables_initializer()

    # 保存计算图，目录、文件夹名称不要有空格
    writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
    writer.flush()

    sess = tf.Session()
    sess.run(init)  # 运行初始化节点，完成初始化

    for step in range(1000):
        # TrainOp没有返回值，所以用'_'
        _, train_loss, train_w, train_b = sess.run(
            [TrainOp, Loss, W, b], feed_dict={X: train_X, Y: train_Y})

        # 每训练5次后输出当前模型的损失
        if (step + 1) % 5 == 0:
            print "Step:", step + 1, "Train loss=",
            train_loss, "W=", train_w, "b=", train_b

        # 每训练5次后对当前模型进行测试
        if (step + 1) % 5 == 0:
            test_loss, test_w, test_b = sess.run(
                [evaloss, W, b], feed_dict={X: test_X, Y: test_Y})
            print "Step:", step + 1, "Test loss=",
            test_loss, "W=", test_w, "b=", test_b

    print "训练完毕"

    W, b = sess.run([W, b])

    training_loss = sess.run(Loss, feed_dict={X: train_X, Y: train_Y})
    print "训练集上的损失：", training_loss

    test_loss = sess.run(evaloss, feed_dict={X: test_X, Y: test_Y})
    print "测试集上的损失：", test_loss
