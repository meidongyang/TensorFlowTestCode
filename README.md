# TensorFlowTestCode
一些常见的API和Tips
-------

##### 忽略掉一些TensorFlow的Warning

```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```
上面这段代码可以在运行时忽略掉一些TensorFlow的Warning（大多数是提示一些包没有被编译（compiled）之类的，如SSE4.2、AVX、AVX2、FMA等）.
最好还是解决一下这些Warning，要是懒得弄就添加上面的代码.

-------
##### tf.reduce_sum

```
tf.reduce_sum(input_tensor,axis=None,keep_dims=False,
              name=None, reduction_indices=None)
```
沿着张量的某个维度求和

例：
   
``` 
X: [[1, 1, 1],
    [1, 1, 1]]
```
    
    tf.reduce_sum(X) ==> 6
    tf.reduce_sum(X, axis=0) ==> [2, 2, 2]
    tf.reduce_sum(X, axis=1) ==> [3, 3]
    tf.reduce_sum(X, axis=1, keep_dims=True) ==> [[3],
                                                  [3]]
    tf.reduce_sum(X, [0, 1]) ==> 6
    
-------

##### tf.arg_min
    tf.arg_min(input, dimension, name=None)
返回’input‘张量在某个维度上的最小值的索引。
对应的还有   tf.arg_max 等

-------
##### Summary类对不同类型的数据进行汇总

* ###### 对标量数据汇总和记录：

        tf.summary.scalar(tags, values, collections=None, name=None)
tags - 标签
values - 值

* ###### 对图像数据汇总和记录：

    
        tf.summay.image(tag, tensor, max_outputs=3, collections=None, name=None)

其中tensor的shape必须是4维：[batch_size, height, width, channels].
TensorBoard中显示的image summary总是最后一次迭代中那一组batch中的数据.

* ###### 对音频数据汇总和记录：


        tf.summay.audio(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None)
                               
sample_rate 是对音频的采样率


* ###### 记录变量的直方图，输出带直方图的汇总protobuf：

        tf.summary.histogram(tag, values, collections=None, name=None)

-------

##### tf.train.GradientDescentOptimizer
    
    Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    trainOp = Optimizer.minimize(Loss)
用梯度下降法更新参数，使Loss降低.

-------
##### Variable( )、placeholder( )、constant( )的区别

* 在 TensorFlow 中 Variable() 是一个类，而 placeholder() 和 constant()是函数.

* 初始化阶段和初始化的方法不同.
    Variable() 必须在 run() 整个图之前就初始化好，用 assign() 或 initializer(). 在运行过程中 TensorFlow 的自动求导及反向传播方法可以不断的调整它的值（比如参数 W／b ）；
    placeholder() 在开始运行／运行过程中不断的重新初始化，用 feed_dict={} 方法对其赋初值（比如数据集 X 和标签 Y）；
    constant() 是个常量，它可以被用在 assign()／initializer() 中为 Variable() 赋值，也可以被用在 feed_dict={} 中为 placeholder() 赋值.
    
* Variable 和 placeholder 都用于为计算图规定或声明数据，比如数据的大小、类型等，以完成图的构建.
    但为什么又要分为 Variable 和 placeholder ？
    因为在整个图所描述的那些张量中，输入张量的形态是最自由的，所以没法用一个统一的类来描述它，也就是说可以给整个图喂任何数据：语音、文本、图像、视频···
    但是图内中间各层的变量 Variable ，必须要遵循一致的约定和规范，以此实现前向和反向传播中的各种自动化运算（比如求导、求梯度）.
    一致的约定和规范就是类，这就是为什么 TensorFlow 中的 Variable() 和 placeholder() 看起来功能相似，但一个是 Class，而另一个是 Function.
    
* constant() 非常底层，只是用来为变量 Variable 或占位符 placeholder() 生成数据的.
    变量 Variable 和占位符 placeholder() 只是对图中所需的数据的规定和声明，在初始化之前，它们是空的，没有任何数据.

填充数据可以用 constant／random／ones 等这些更底层的组件.
    


    




