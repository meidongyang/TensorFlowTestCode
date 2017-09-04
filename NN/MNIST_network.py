# -*- coding:UTF-8 -*-

'''结构：两个隐藏层的神经网络
构建模型的步骤：
1. inference() - 向前预测过程（从输入到预测输出的计算图路径)
2. loss() - 为inference()中需要计算损失的layer添加损失计算节点
3. training() - 为loss()添加需要计算梯度和应用梯度的节点
4. evaluation() - 评估
'''
import os
import os.path
import sys
import argparse
import math
import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning

NUM_CLASSES = 10  # MNIST有10个类：0、1、2、3、4、5、6、7、8、9.

IMAGE_SIZE = 28  # MNIST的图像都是28x28像素
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE  # 展开成784维的特征向量

# 全局变量，用来存放基本的模型(超)参数.
FLAGS = None


# 1. inference() - 向前预测过程（从输入到预测输出的计算图路径)
def inference(images, hidden1_units, hidden2_units):
    ''' tf.truncated_normal - 按照正态分布随机初始化weights
    stddev（标准差）：1.0/sqrt(weights的行数).
    以weights矩阵的行数规范化标准差，就是要让weights矩阵中的每一列都服从0均值的正态分
    布，这样不会给输入信号添加人为的偏置
    '''
    # hidden1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # hidden2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear Regression
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


# 2. loss() - 为inference()中需要计算损失的layer添加损失计算节点
def loss(logits, labels):
    labels = tf.to_int64(labels)
    # 计算交叉熵损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


# 3. training() - 为loss()添加需要计算梯度和应用梯度的节点
def training(loss, learning_rate):
    '''设置training()函数：

    创建一个汇总节点（summarizer）跟踪记录损失值随着时间的变化，显示在TensorBoard上.

    创建一个优化节点（optimizer）并对所有的可训练变量（trainable variables）应用梯度节点，也就是更新参数.

    该函数的返回节点（Op）必须要被传入到‘sess.run()’中来启动模型训练过程.

    Args（参数）:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns（返回值）:
      train_op: The Op for training.
    '''

    # 为保存loss值添加一个标量汇总(scalar summary).
    tf.summary.scalar('loss', loss)
    # 根据给定的学习率来创建梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # 创建一个变量用来跟踪global step
    global_step = tf.Variable(0, name='global_step', trainable=True)
    # 在训练节点，使用optimizer将梯度下降法应用到可训练参数上以此来减小损失函数的损失值
    # 同时不断增加global step计数器
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# 4. evaluation() - 评估
def evaluation(logits, labels):
    '''对于分类器模型，我们可以使用 in_top_k Op.
    返回一个shape为[batch_size]的bool tensor
    只要在所有的预测输出logits里面的前k个预测结果中有一个是正确的，则
    对应的样本就算正确，对应的位置就是True.
    在这里k=1.
    '''
    correct = tf.nn.in_top_k(logits, labels, k=1)
    # 返回当前批次样本中预测正确的样本数量
    return tf.reduce_mean(tf.cast(correct, tf.int32))


# 创建 placeholder variables 来表示输入张量
def placeholder_inputs(batch_size):
    # 注意：placeholders的shape要与数据集的shape相互匹配
    # 除了第一个维度的数量变成batch size而不是整个数据集的全部个数
    images_placeholder = tf.placeholder(
        tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


# 为参数指定的placeholder创建feed_dict，用下一批次的‘batch_size’个样本填充
# batch_size大小对学习效果的影响参考之一：https://www.zhihu.com/question/32673260
def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                   FLAGS.fake_data)
    feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
    return feed_dict


# 在给定的数据集上执行一次评估操作
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    # 运行一个回合(one epoch)的评估过程.
    true_count = 0  # 对正确预测的样本进行计数
    # 每个回合的执行步数
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    # 样本数量
    num_examples = steps_per_epoch * FLAGS.batch_size

    # 累加每个batch中预测正确的样本数量
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    # 所有批次上的样本的精确度 -- 总精确度
    precision = float(true_count) / num_examples
    print 'Num examples:', num_examples, ' Num correct:', true_count
    print ' Precision:', precision


# 对MNIST网络训练指定的次数（一次训练称为一个training step）
def run_training():
    # 获取用于训练、验证和测试的数据集及类别标签集
    data_sets = input_data.read_data_sets(
        FLAGS.input_data_dir, FLAGS.fake_data)

    # 告诉TensorFlow这个模型将会被构建在默认的Graph上.
    with tf.Graph().as_default():
        # 为图像特征向量数据和类别标签数据创建输入placeholder
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)
        # 从向前推断模型中构建用于预测的计算图
        logits = inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # 为计算图添加计算损失的节点.
        Loss = loss(logits, labels_placeholder)
        # 为计算图添加计算和应用梯度的训练节点
        train_op = training(Loss, FLAGS.learning_rate)
        # 添加评估节点，用于在评估过程中比较logits和ground truth（真实标签）.
        eval_correct = evaluation(logits, labels_placeholder)

        # 基于 TF collection of Summaries构建汇总张量
        summary = tf.summary.merge_all()
        # 添加变量初始化节点（Variable initializer Op）
        init = tf.global_variables_initializer()
        # 创建一个saver 用于写入训练过程中的模型检查点文件(checkpoints).
        saver = tf.train.Saver()

        # 创建会话（Session）用来运行计算图中的节点
        sess = tf.Session()
        # 实例化一个 SummaryWriter 输出 summaries 和 graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # 运行初始化节点 -- 初始化所有变量（Variables）
        sess.run(init)

        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # 使用真实的图像和类标签填充feed dict
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            # 在当前batch上把模型运行一步（run one step）
            # 返回值是从'train_op'（忽略不要）和'loss'节点拿到的activations
            _, loss_value = sess.run([train_op, Loss], feed_dict=feed_dict)

            # 计算训练当前batch花费的时间
            duration = time.time() - start_time

            # 每隔100步写入一次summaries并输出overview
            if step % 100 == 0:
                print 'Step', step, 'Loss=', loss_value, duration, 'sec'
                # 更新事件
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # 周期性保存一个checkpoint并对当前模型进行评估，文件后缀ckpt 意思是 checkpoint
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # 在训练集上评估模型
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # 在验证集上评估模型
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # 在测试集上评估模型
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)


# 创建日志文件夹
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    # 开始训练，调用run_training()函数
    run_training()


# 用ArgumentParser类把模型的(超)参数全部解析到全局变量FLAGS里
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Initial learning rate.')
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.')
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.')
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='mnist_data',
        help='Directory to put the input data.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='3layerNN',
        help='Directory to put the log data.')
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, use fake data for unit testing.',
        action='store_true')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
