#!/usr/bin/python
# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# 加载数据
mnist = input_data.read_data_sets('/mnt/hgfs/share/learn/ML/mnist_data', one_hot=True)


# 权重初始化;用一较小正数初始化偏置项以避免神经元节点输出恒为0
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 卷积步长为1边界用0填充;保证输出和输入是同一个大小
# 输入图像shape=[batch,in_height,in_width,in_channels]float32/float64
# 卷积核shape=[filter_height, filter_width, in_channels, out_channels]
def conv2d(x, W):
    # strides 在图像每一维的步长;返回一个Tensor即feature map
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 池化用2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 784为图片的维度数;None对应shape的第一个维度,代表了这批输入图像的数量
x = tf.placeholder("float", [None, 784])
# 一个10维向量代表图片所述的类别
y_ = tf.placeholder("float", shape=[None, 10])
# 将x变为一个4维向量　[一个batch的图片数量,宽,高,通道数]
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积
# 卷积的权重张量形状是[5, 5, 1, 32],前两个维度是patch的大小,接着是输入的通道数目,最后是输出的通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1) , b_conv1))
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
# 每个5x5的patch会得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2) , b_conv2))
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# 加入一个有1024个神经元的全连接层将图片尺寸减小到7x7
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout;用占位符来代表一个神经元的输出在dropout中保持不变的概率
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层; softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
# 交叉熵损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 学习速率设为1e-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# tf.argmax()从一个tensor中寻找最大值的序号
# tf.equal()判断数字类别是否是正确的类别
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 统计全部样本预测的accuracy：用tf.cast()将bool值转换为float32然后求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# 运行交互计算图
sess = tf.InteractiveSession()
# 全局参数初始化
sess.run(tf.initialize_all_variables())

for i in range(2000):
    batch = mnist.train.next_batch(50)
    # 每100次迭代输出一次日志
    if i % 100 == 0:
        # 训练过程中启用dropout;在feed_dict中加入额外的参数keep_prob来控制dropout比例
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 测试过程中关闭dropout
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print ("test accuracy %g" % acc)

sess.close()
