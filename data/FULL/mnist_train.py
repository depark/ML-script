#定义神经网络训练过程
import sys
sys.path.append('/root/ML')

import tensorflow as tf
from data.FULL import mnist_interface
# from data.CNN import interface
import os
from tensorflow.examples.tutorials.mnist import input_data


#神经网络训练参数
#每伦训练数据
BATCH_SIZE = 100
#正则化在损失函数中参数
REGULARIZATION_RATE = 0.0001
#初始学习率
LEARNING_RATE_BASE = 0.8
#学习衰减率
LEARNING_RATE_DECARY = 0.99

MOVING_AVEAGE_DECAY = 0.99      #滑动平均衰减率

#训练次数
TRAING_STEPS = 30000
#模型保存路径
MODEL_SAVE_PATH = "/root/ML/MODEL/full"
#模型保存名字
MODEL_NAME = "model.ckpt"

mnist = input_data.read_data_sets('/root/ML/data/ini_data/mnis_data', one_hot=True)


def train():

    # 定义输入输出
    x = tf.placeholder(tf.float32,shape=[None,28*28],name="x-input")
    # x = tf.placeholder(tf.float32, [
    #     BATCH_SIZE,  # 第一维为一个batch样例中个数
    #     mnist_interface.IMAGE_SIZE,  # 第二维和第三维表示图片尺寸
    #     mnist_interface.IMAGE_SIZE,
    #     mnist_interface.NUM_CHANNELS  # 第四维表示图片深度,黑白为1  RGB格式为3
    # ])
    # x_image = tf.reshape(x,[BATCH_SIZE,28,28,1])

    y_ = tf.placeholder(tf.float32,[None,mnist_interface.OUTPUT_NODE],name="y-input")


    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #直接全连接进行前向传播
    y = mnist_interface.interface(x,regularizer)
    # y_test = mnist_interface.interface(x,None,True)

    # 卷积网络训练
    # y = interface.interface(x_image,True, regularizer)
    # 卷积网络测试
    # y_test = interface.interface(x_image,False, None)

    # # #定义验证正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
    # #
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #定义迭代伦数
    global_step = tf.Variable(0,trainable=False)

    #定义滑动平均模型
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVEAGE_DECAY,global_step
    )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #定义损失函数，交叉熵函数,y为不经过softmax转换的前向传播输出,labels参数为正确答案，由于正确答案只有一个
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    # 计算所有样例交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #总损失 交叉熵 和 正则损失
    losses = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    #定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECARY
    )

    #反向传播算法,自动更新global_step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses,global_step=global_step)
    #多个计算合在一起
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    #模型持久化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # test_feed = {x: mnist.test.images,y_: mnist.test.labels}
        #训练
        for i in range(TRAING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss, step = sess.run([train_op, losses, global_step],feed_dict={x: xs, y_: ys})

            #每1000次训练保存模型
            if i % 1000 == 0:
                test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y_: mnist.test.labels})
                print("测试正确率为 %g." % test_acc)
                print("经过 %d 次训练,在测试集上 损失度是 %g." % (step,loss))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        # test_acc = sess.run(accuracy,feed_dict=test_feed)

        # print("最终测试正确率为 %g." % test_acc)


if __name__ == "__main__":
    train()