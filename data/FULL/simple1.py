#训练神经网络问题完整过程,线性激活函数运算

'''

'''


import tensorflow as tf
from numpy.random import RandomState
import matplotlib.pyplot as plt

plt.plot()


#定义神经网络的参数,两个矩阵参数

w1 = tf.Variable(tf.random.normal([2,3],stddev=1,seed=1,name='w1'))    # 2x3 矩阵
w2 = tf.Variable(tf.random.normal([3,1],stddev=1,seed=1,name='w2'))    # 3x1 矩阵


#定义训练集,使用None可以测试不同的batch大小

x = tf.compat.v1.placeholder(tf.float32,shape=(None,2),name='x-input')   # 定义2个特征向量数据集

y_= tf.compat.v1.placeholder(tf.float32,shape=(None,1),name='y-input')   # 定义测试分类集 正确答案


#定义前向传播过程,输入与参数的矩阵相乘

a=tf.matmul(x,w1)
y = tf.matmul(a,w2)    #模型计算答案


#定义损失函数和反向传播算法,将输出转化为概率

# y = tf.sigmoid(y)

#计算交叉熵
# cross_entropy = -tf.reduce_mean(
#     y_* tf.math.log(tf.clip_by_value(y,1e-10,1.0)) +
#     (1-y_) * tf.math.log(tf.clip_by_value(1-y,1e-10,1.0))
# )

#单个输出交叉熵计算
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ , logits=y)


#定义反向传播算法，用来优化参数
train_step = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cross_entropy)


#模拟随机生成数据

rdm = RandomState(1)
data_size = 1280 *2
X= rdm.rand(data_size,2)

#x1+x2<1认为是正,其他认为是负    0表示负样本  1表示正样本
Y = [[int(x1+x2 <1)] for (x1,x2) in X]

#标准梯度下降
tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

tf.train.AdadeltaOptimizer


#进行程序

with tf.compat.v1.Session() as sess:

    # 初始化变量
    init_ops = tf.global_variables_initializer()
    sess.run(init_ops)

    #训练之前参数
    print('初始参数')
    print(sess.run(w1))
    print(sess.run(w2))


    #模拟进行5000次训练
    for i in range(5000):

        start = (i * 80) % data_size
        end = min(start+80,data_size)

        sess.run(train_step, feed_dict={x: X[start:end],y_: Y[start:end]})

        if i % 1000 ==0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X,y_:Y})

            print('经过 %s 次训练，交叉熵为%s' % (i,total_cross_entropy))

    W1 = sess.run(w1)
    W2 = sess.run(w2)
    print("最终参数")
    print(W1)
    print(W2)














