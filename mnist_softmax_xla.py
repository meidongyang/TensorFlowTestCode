# -*- coding:UTF-8 -*-

'''TensorFlow线性代数编译框架XLA(Accelerated Linear Algebra/加速线性代数)
它是一个用于 TensorFlow 的编译器。XLA 使用 JIT 编译技术来分析用户在运行时（runtime)
创建的 TensorFlow 图，专门用于实际运行时的维度和类型，它将多个 op 融合在一起并为它们形
成高效的本地机器代码——能用于 CPU、GPU 和自定义加速器（例如谷歌的 TPU）。

'''
import argparse
import sys
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 防止运行时出现莫名其妙的Warning

FLAGS = None


def main(_):
    # 导入数据集
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # 创建一个模型
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b

    # 创建损失函数和优化器
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    config = tf.ConfigProto()
    jit_level = 0
    if FLAGS.xla:
        # 开启XLA JIT编译
        jit_level = tf.OptimizerOptions.ON_1

    config.graph_options.optimizer_options.global_jit_level = jit_level
    run_metadata = tf.RunMetadata()
    sess = tf.Session(config=config)
    tf.global_variables_initializer().run(session=sess)
    # Train
    train_loops = 1000
    for i in range(train_loops):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        # Create a timeline for the last loop and export to json to view
        # 用chrome浏览器，在网址栏中输入 chrome://tracing/
        # 然后导入运行出来的json文件即可
        # 可以忽略不看
        if i == train_loops - 1:
            sess.run(train_step,
                     feed_dict={x: batch_xs, y_: batch_ys},
                     options=tf.RunOptions(
                         trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata)
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.ctf.json', 'w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format())
        else:
            # Train
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 测试
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels})
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='mnist_data',
                        help='Directory for storing input data.')
    parser.add_argument('--xla', type=bool, default=True,
                        help='Trun xla via JIT on')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
