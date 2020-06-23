# -*- coding: utf-8 -*-
#预测sin正弦取值,使用深层循环神经网络

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

#LSTM隐藏节点个数
HIDDEN_DIZE = 30
#LSTM层数
NUM_LAYERS = 2

#LSTM训练序列长度
TIMESTAMPS = 10
#训练轮数
TRAINING_STEPS = 10000
#batch大小
BATCH_SIZE = 32

#数据准备
#训练数据个数
TRAINING_EXAMPLES = 10000
#测试数据个数
TESTING_EXAMPLES = 10000
#采样间隔
SAMPLE_GAP = 0.01

#数据分类
def generate_data(seq):
    #seq: 所有训练数据
    #生成训练数据 每TIMESTAMPS个数字 预测第TIMESTAMPS个值
    X = []
    Y = []
    #使用sin前TIMESTAMPS个数值预测第TIMESTAMPS+1个数值
    for i in range(len(seq) - TIMESTAMPS):
        X.append([seq[i:i+TIMESTAMPS]])
        Y.append([seq[i+TIMESTAMPS]])
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float32)

#使用多层LSTM结构
def lastm_model(X,y,is_training):
    #建立LSTM模型
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_DIZE) for _ in range(NUM_LAYERS)
    ]
    )

    #将多层LSTM连接为RNN结构网络并计算前向传播过程
    outputs,_ = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

    #outputs是顶层LSTM每一步输出结果维度是[batch_size,time,HIDDEN_DIZE],只关注最后一刻输出结果
    output = outputs[:, -1, :]

    #对于LSTM加上全连接层并计算损失,平方差损失函数

    #进行一次全连接后返回结果
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn= None)
    #测试时直接返回预测结果
    if not  is_training:
        return predictions,None,None

    #计算损失函数，平方差损失函数

    loss = tf.losses.mean_squared_error(labels=y,predictions=predictions)
    #创建优化器
    train_ops = tf.contrib.layers.optimize_loss(loss,tf.train.get_global_step(),optimizer='Adagrad',learning_rate=0.1)

    return predictions,loss,train_ops

def train(sess,train_x,train_y):
    #将数据随机打乱分类
    ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X,y = ds.make_one_shot_iterator().get_next()

    #调用模型训练
    with tf.variable_scope('model'):
        predictions, loss, train_ops = lastm_model(X,y,True)

    #进行训练
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _,l = sess.run([train_ops, loss])
        if i % 100 == 0:
            print("经过 %s 轮训练,损失为 %s" % (str(i),str(l)))

#验证测试数据
def run_test(sess,test_x,test_y):
    #测试数据
    ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
    ds = ds.batch(1)
    X,y = ds.make_one_shot_iterator().get_next()

    # 调用模型获得预测结果,不用传入真实值
    with tf.variable_scope('model',reuse=True):
        prediction, _, _ = lastm_model(X, [0.0], False)

    #将预测结果存入数组
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p,l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    #计算rmse作为指标,平方差作为损失度
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt((predictions - labels) ** 2).mean(axis=0)
    print('测试损失度为 ' + str(rmse))

    #对预测数进行绘图
    plt.figure()
    plt.plot(predictions,label='predictions')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(labels,label='real')
    plt.legend()
    plt.show()

def main():
    # 使用正弦函数生成训练和测试数据
    test_start = (TRAINING_EXAMPLES + TIMESTAMPS) * SAMPLE_GAP
    test_end = test_start + (TESTING_EXAMPLES + TIMESTAMPS) * SAMPLE_GAP
    train_x,train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES + TIMESTAMPS),dtype=np.float32))
    test_x,test_y = generate_data(np.sin(np.linspace(test_start,test_end,TESTING_EXAMPLES + TIMESTAMPS),dtype=np.float32))

    with tf.Session() as sess:
        #训练
        train(sess,train_x,train_y)
        #预测
        run_test(sess,test_x,test_y)

if __name__ == "__main__":
    main()