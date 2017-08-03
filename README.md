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
-------
##### tf.truncated_normal  截断正态分布

    tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None, name=None)
返回一个从截断正态分布（truncated normal distribution）抽取的随机数.
产生的随机数服从给定的均值 mean 和标准差 stddev 的正态分布，那些到均值的距离超过2倍标准差的随机数将被丢弃，然后重新抽取，直到得到足够数量的随机数为止，x 的范围：[mean - 2 * stddev, mean + 2 * stddev].
参数 Args：

* shape: A 1-D integer Tensor or Python array. 指定输出张量的shape.
* mean: A 0-D Tensor or Python value of type 'dtype'. 截断正态分布的均值.
* stddev: A 0-D Tensor or Python value of type 'dtype'. 截断正态分布的标准差.
* dtype: 输出随机数的数据类型.
* seed: 随机数种子.
* name: 为这个Op起个名字（可选）.

> 返回值 Returns：
> 填充着从指定的截断正态分布中抽取的随机数的张量. A Tensor of the specified shape filled with random truncated normal values.

> 截断正态分布是截断分布（Truncated Distribution）的一种，是指限制变量 x 取值范围（scope）的一种分布.
> 假设 x 原来服从正态分布，那么限制 x 的取值在(a, b)范围内之后，x 的概率密度函数为：
\\[f(x; \mu, \sigma, a, b)=\frac{\frac{1}{\sigma }\phi(\frac{x-\mu}{\sigma})}{\phi(\frac{b-\mu}{\sigma})-\phi(\frac{a-\mu}{\sigma})}\\]


-------
##### tf.identity

    tf.identity(input, name=None)
    
Return a Tensor with the same shape and contents as the input Tensor or value.
返回一个与 input 的 shape 和 value 一模一样的 Tensor.

-------
### 卷积API

``` 
tf.nn.conv1d(value, filters, stride, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
```   
```
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
```   
``` 
tf.nn.conv3d(input,filter, strides, padding, name=None)
```   
``` 
tf.nn.convolution(input, filter, padding, strides=None, dilation_rate=None, name=None, data_format=None)
```    
```     
tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)
``` 
``` 
tf.nn.separable_conv2d(input, depthwise_filter, pointerwise_filter, strides, padding, name=None)
``` 
``` 
tf.nn.atrous_conv2d(value, filters, rate, padding, name=None)
```  
``` 
tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', data_format=NHWC, name=None)
```   

* input: 指需要做卷积的输入张量（Tensor），shape=[batch, in_height, in_width, in_channels], 即[训练时数据集中的一个 batch 的图片数量，图片高度，图片宽度，颜色通道数]，是一个4维 Tensor ，要求数据类型为 float32 或 float64.
* filter: Tensor, shape=[filter_height, filter_width, in_channels, out_channels], 即[滤波器（卷积核）的高度，滤波器的宽度，颜色通道数，滤波器的个数]. 要求数据类型与 input 相同. 其中，第三维 in_channels 就是 input 的第四维.
* strides: 步长，一维向量，长度4. 卷积 2D 图像时 strides[0] = strides[3] = 1, 需要调整的是 strides[1] 和 strides[2].
* padding: string，只能是' SAME '或' VALID '其中之一，这个值决定了不同的卷积方式.
* use_cudnn_on_gpu: bool, 是否使用cudnn加速，默认为 True.
 
返回值是一个Tensor，就是我们常说的 feature map（特征图）. feature map 也可以作为 conv2d 的 input.

> 输入张量的 shape，卷积滤波器核的 shape，输出张量的 shape 该如何计算？

> 输入数据体的尺寸 W1 x H1 x D1

> 4个超参数：

> | 名称 | 标记 |
> | --- | --- | --- |
> | 滤波器的数量 | K |
> | 滤波器的空间尺寸 | F |
> | 步长 | S |
> | 零填充数量 | P |

> 输出数据体的尺寸为 W2 x H2 x D2, 其中：
> \\[W2=\frac{W1 - F + 2P}{S} + 1\\]
> \\[H2=\frac{H1 - F + 2P}{S} + 1\\]
> \\[D2=K （有多少个滤波器就有多少个特征图）\\]
> W2 和 H2 的计算方法相同.


上述公式中的超参数 K 和 F 确定了权重 weight（也就是 filter）的 shape=[K, K, D, F].
超参数 K 确定了偏置 biases 的 shape=[K].

如果想让输入与输出张量的空间尺寸保持不变，那么 padding="SAME", strides=[1, 1, 1, 1]. 此时公式的零填充数量 P 是 TensorFLow自动计算的.
P 的计算公式为：
    \\[P=\frac{(W2 - 1)*S - W1 + F}{2}\\]

当 padding="VALID" 时， P=0， 也就是不填充. 卷积后的输出张量的空间尺寸会减小 F - 1 圈.

当输入空间尺寸为偶数，滤波器核的数量为奇数时，不要把 S 设为 2， 因为不能整除，所以 S = 1.
如果想大幅缩减尺寸，后接 pool 层即可.

-------
### Pooling API

    
```
tf.nn.avg_pool(value, ksize, strides, padding, data_format=NHWC, name=None)
```
```
tf.nn.max_pool(value, ksize, strides, padding, data_format=NHWC, name=None)
```
```
tf.nn.max_pool_with_argmax(value, ksize, strides, padding, Targmax=None, name=None)
```
```
tf.nn.max_pool3d(value, ksize, strides, padding, name=None)
```
```
tf.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random=None, overlapping=None, deterministic=None, seed=None, seed2=None, name=None)
```
```
tf.nn.fractional_max_pool(value, pooling_ratio, pseudo_random=None, overlapping=None, deterministic=None, seed=None, seed2=None, name=None)
```
```
tf.nn.pool(input, window_shape, pooling_type, padding, dilation_rate=None, strides=None, name=None, data_format=None)
```
* value: 需要池化操作的输入，一般池化层接在卷积层后面，所以输入通常是feature map.
* ksie: 池化窗口大小，4维向量，一般是[1, height, width, 1].
* strides: 和卷积的 strides 类似，一般是[1, stride, stride, 1].
* padding: ' VALID '或' SAME ', 大多数时候为' VALID '.
* data_format: 张量数据格式，可以是' NHWC '或' NCHW ', 一般用默认值. 必须和 conv2d 的格式一致.

> Pooling layer 公式：
> 
> 输入数据体的尺寸 W1 x H1 x D1

> 2个超参数：

> | 名称 | 标记 |
> | --- | --- | --- |
> | 空间尺寸 | F |
> | 步长 | S |

> 输出数据体的尺寸为 W2 x H2 x D2, 其中：
> \\[W2=\frac{W1 - F}{S} + 1\\]
> \\[H2=\frac{H1 - F}{S} + 1\\]
> \\[D2=D1\\]
> 很少使用零填充，F太大对网络有破坏性.
