#实现LetNet5模型
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

#定义前向传播网络
def lenet5(inputs):
    #输入转化为4维数组,batch 图片大小
    inputs = tf.reshape(inputs,[-1,28,28,1])
    #定义第一层卷积层,深度32 过滤器大小[5，5],使用全0填充
    net = slim.conv2d(inputs,32, [5,5],padding='SAME',scope='layer1-conv')
    #定义最大池化层,过滤器2x2 步长2
    net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
    #定义第三次卷积层 64层深度， 5x5过滤器, 全0填充
    net = slim.conv2d(net,64 , [5,5],padding='SAME',scope='layer3-conv')
    #第四层最大池化层 步长2 过滤器 2X2
    net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')
    #将4维矩阵转化维2维，方便全连接计算
    net = slim.flatten(net,scope='flatten')
    #两次全连接,500和10个隐藏节点
    net = slim.fully_connected(net,500,scope='layer5')
    net = slim.fully_connected(net,10,scope='output')
    return net

def train(mnis):
    x = tf.placeholder(tf.float32,[None,784], name='x-input')
    y_ = tf.placeholder(tf.float32,[None,10],name= 'y-input')

    y = lenet5(x)

    cross_entry = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    loss = tf.reduce_mean(cross_entry)
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(10000):
            xs,ys = mnis.train.next_batch(100)
            _,loss_value = sess.run([train_op,loss],feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                x_valid,y_valid = mnis.validation.images,mnis.validation.labels
                corr = sess.run(evaluation_step,feed_dict={x: x_valid, y_: y_valid})
                print('经过 %d 轮训练,验证数据正确率为 %.2f %%' % (i,corr * 100))
                print('经过 %d 轮训练,损失为 %s' % (i, loss_value))

        x_test, y_test = mnis.test.images, mnis.test.labels
        test_corr = sess.run(evaluation_step, feed_dict={x: x_test, y_: y_test})
        print('最终测试数据正确率为 %.2f %%' % (test_corr * 100))


def main():
    mnis = input_data.read_data_sets("/root/ML/data/ini_data/mnis_data",one_hot=True)
    train(mnis)

if __name__ == "__main__":
    main()