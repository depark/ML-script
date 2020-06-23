#神经网络,加入正则化，计算损失函数， 调整参数
# 5层神经网络 L2正则化的损失函数


import tensorflow as tf

#计算当前层的L2正则化参数，保存到loss，模型复杂度
def get_weight(shape,lambd):
    #shape: 参数的维度   上层节点数X下层节点数
    # lambd: 模型复杂度损失在总损失中的比例 λ
    # 参数权重
    var = tf.Variable(tf.random.normal(shape), dtype=tf.float32)
    tf.compat.v1.add_to_collection(
        'loss', tf.contrib.layers.l2_regularizer(lambd)(var)
    )
    return var

#输入的特征张量训练集,两个特征值
x = tf.compat.v1.placeholder(tf.float32,shape=(None,2))

# 正确答案集合
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

#定义每次训练的块大小
batch_size = 8

#神经网络每层的节点
layer_dimension = [ 2, 10 , 10, 10, 1 ]

#神经网络层数
n_layers = len(layer_dimension)

#循环参数
#第一层输入
cur_layer = x
#第一层网络
in_dimension = layer_dimension[0]

#五层生成神经网络结构,前向传播，最终得出多模型经过函数的结果
for i in range(1, n_layers):
    #下一层网络
    out_dimension = layer_dimension[i]
    #生成当前层的变量，并计算L2正则化损失函数加入计算集合
    # 变量维度为上一层节点数*下一层节点数
    weight = get_weight([in_dimension, out_dimension], 0.001)
    #活动函数偏移量, 生成0.1 的偏置项
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用ReLU 激活函数,输出下一层
    cur_layer= tf.nn.relu(tf.matmul(cur_layer,weight) + bias)
    #输出到下一层
    in_dimension = layer_dimension[i]

#定义前向传播,正则化接入图集
#损失函数,均方差损失函数
mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

#均方差损失加入图集
tf.add_to_collection('loss',mes_loss)

#所有损失函数加起来
loss = tf.add_n(tf.get_collection('loss'))

