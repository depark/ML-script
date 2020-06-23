#训练mnist图形识别
#

import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data   #导入测试数据


#数据集参数

INPUT_NODE = 784     #28x28 一条数据生成一维向量

OUTPUT_NODE = 10    # 分类标准，区分0-9 10个数字


#神经网络结构

LAYER1_NODE = 500               # 第一隐藏层节点，500个节点 使用一层隐藏

BATCH_SIZE = 100                # batch训练个数

LEARNING_RATE_BASE = 0.8        #基础学习率

LEARNING_RATE_DECARY = 0.99     #学习衰减率

REGULARIZATION_RATE = 0.0001    #正则化在损失函数中参数  λ

TRAINING_STEPS = 50000          #训练轮数

MOVING_AVEAGE_DECAY = 0.99      #滑动平均衰减率


#定义前向传播算法

def inferance(input_tensor, avg_class, weights1,biases1, weights2, biases2):
    '''
    :param input_tensor: 输入变量,(训练集，测试集，验证集)
    :param avg_class:    计算滑动变量函数,获得衰减后的参数
    :param weights1:     第一层到隐藏层参数集
    :param biases1:      第一层到隐藏层线性偏移量
    :param weights2:     隐藏层到输出层 权重参数
    :param biases2:      隐藏层到输出层 偏移量
    :return:          返回前向传播计算输出
    '''

    if avg_class == None:
        #没有滑动衰减函数
        # 计算隐藏层的前向传播,使用relu激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
        # 隐藏层到输出层计算,没加入激活函数, 计算损失函数加入softmax函数
        output = tf.matmul(layer1,weights2) + biases2

    else:
        # 权重变量使用滑动算法计算得出
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))+ avg_class.average(biases1))

        output = tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

    return output


# 训练过程

def train():
    # 导入数据
    mnist = input_data.read_data_sets('D:\learn\ML\data',one_hot=True)

    #定义模型的输入和输出
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name='x-input')

    #定义正确标签
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    #生成隐藏层参数和偏移量
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))  #第一层隐藏节点数

    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))  # 输出层的节点

    #不用滑动平均衰减计算正常传播算法得出的输出
    y = inferance(x, None, weight1, biases1, weight2, biases2)

    #定义存储训练伦数的变量, 初始值0
    global_step = tf.Variable(0, trainable=False)

    #定义滑动平均函数
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVEAGE_DECAY, global_step)

    #定义滑动平均操作，更新衰减率,更新列表中参数--> 更新标记不训练的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用平滑平均后的正向传播结果
    average_y = inferance(x, variable_averages, weight1, biases1, weight2, biases2)

    #损失函数 bratch
    #交叉熵计算损失函数,训练集中每条记录的交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))

    #计算训练集所有交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #计算正则化总损失,一般只计算权重正则化，不用偏置项
    regularizer_total = regularizer(weight1) + regularizer(weight2)

    #总损失等于 交叉熵加上正则化总损失
    loss = cross_entropy_mean + regularizer_total

    #学习率衰减
    learninging_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECARY
    )

    #优化损失函数，反向传播算法
    train_step = tf.train.GradientDescentOptimizer(learninging_rate).minimize(loss,global_step=global_step)

    #同时进行反向传播算法更新参数，同时更新滑动平均值
    train_op = tf.group(train_step, variable_averages_op)

    #检验滑动
    # 平均模型的正向传播结果和正确结果的、正确率
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

    #结果从布尔转换为float，再计算平均值,这个平均值就是模型在这组数据上的正确率
    accurarcy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #开始训练
    with tf.Session() as sess:
        #初始化参数
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #验证数据,通过验证数据来大致判断停止的条件和评判训练的效果

        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        #测试数据
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        #开始迭代训练
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                #计算滑动平均模型在验证数据上的结果
                validate_acc = sess.run(accurarcy, feed_dict = validate_feed)
                print('经过 %d 次训练, 使用滑动平均模型测试结果为 %s' % (i,validate_acc))

            # 进行训练
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict= {x: xs, y_: ys})

        #训练结束，在测试数据上检测最终正确率
        test_acc = sess.run(accurarcy, feed_dict=test_feed)
        print('最终测试结果为 %g' % test_acc)


def main():
    train()


if __name__ == '__main__':
    main()





