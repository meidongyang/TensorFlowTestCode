# -*- coding:UTF-8 -*- 
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning

with tf.Graph().as_default():
    # 输入占位符
    X = tf.placeholder(tf.float32)
    Y_ = tf.placeholder(tf.float32)

    # 模型参数变量
    W = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))

    # inference: y = wx + b
    Ypred = tf.add(tf.multiply(X, W), b)

    # 计算损失
    Loss = tf.reduce_mean(tf.pow((Y_ - Ypred), 2)) / 2

    # 创建梯度下降优化器
    Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # Train：定义训练节点将梯度下降法应用于Loss
    # 在优化器内部对权重W和b进行更新，在TensorBoard中可以看到
    # 路径：GradientDescent图中update_Variable里面的ApplyGradientDescent
    TrainOp = Optimizer.minimize(Loss)

    # 添加评估节点
    evaloss = tf.reduce_mean(tf.pow((Y_ - Ypred), 2)) / 2

    # 保存计算图，目录、文件夹名称不要有空格
    writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
    writer.flush()
