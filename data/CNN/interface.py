#LeNet5卷积神经网络训练 mnist数据，前向传播过程

import tensorflow as tf

BATCH_SIZE = 100

#神经网络参数
INPUT_NODE = 28*28
OUTPUT_NODE = 10

#图片参数
IMAGE_SIZE = 28    #图片尺寸  28*28
NUM_CHANNLES = 1   # 图片深度
NUM_LAYERS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5  #过滤矩阵 5*5

#第二卷积层尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层节点数
FC_SIZE = 512

#定义输入矩阵,四维矩阵
# x = tf.placeholder(tf.float32,[
#     BATCH_SIZE,          # 第一维为一个batch样例中个数
#     IMAGE_SIZE,          # 第二维和第三维表示图片尺寸
#     IMAGE_SIZE,
#     NUM_CHANNLES         # 第四维表示图片深度,黑白为1  RGB格式为3
# ])

def interface(input_tensor,train,regularizer):
    #定义第一层卷积层变量,输入28x28 全0填充,步长为1,深度32   输出为 28x28x32
    with tf.variable_scope("layer_cover1"):
        conv1_weights = tf.get_variable("conv1_weights",[CONV1_SIZE, CONV1_SIZE, NUM_CHANNLES, CONV1_DEEP],      # 第一层卷积变量 1-2过滤矩阵,3 输入矩阵深度,4 一层卷积深度
                                        initializer=tf.truncated_normal_initializer(stddev=0.1)
                                        )
        conv1_biases = tf.get_variable("conv1_biases",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        #使用边长5 深度32过滤器 步长1 全0填充,卷积层计算
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1,1,1,1], padding="SAME"
        )
        #激活函数和偏置项
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #实现第二层池化层,使用最大池化层,过滤器 2X2 步长2，全0填充， 输入为28x28x32 输出为 14x14x32
    with tf.variable_scope("layer2_pool1"):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")    #ksize 过滤器大小 2x2  strides 步长 2 2，使用全0填充

    #第三层卷积层 输入为 14x14x32 输出为 14x14x64
    with tf.variable_scope("layer3_cover2"):
        conv2_weights = tf.get_variable("conv2_weights",[CONV2_SIZE,CONV2_SIZE, CONV1_DEEP,CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("conv2_biases",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))

        #前向传播过程，卷积层计算,使用边长5 深度64，步长1 全0填充
        conv2 = tf.nn.conv2d(
            pool1,conv2_weights,strides=[1,1,1,1],padding="SAME"
        )

        #激活函数和偏移项
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #第四次池化层 输入 14x14x64 输出 7x7x64
    with tf.variable_scope("layer4_pool2"):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #转化为向量,使得能够进行全连接,输出为 [1,7,7,64] 矩阵的四维,第一个为batch的数据个数
    pool_shape = pool2.get_shape().as_list()
    print(pool_shape)
    #计算一个batch数据个数  7*7*64
    nodes= pool_shape[1] * pool_shape[2] * pool_shape[3]

    #转换成一个batch向量，转换成Batch_size个向量
    reshape = tf.reshape(pool2,[-1,nodes])
    print(reshape.shape)


    #向量长度为7x7x64 输出为长度为512的向量，dropout 在训练时会随机将部分节点输出改为0,避免过拟合，dropout一般只在全连接层使用
    with tf.variable_scope("layer5_fc1"):
        fc1_weights = tf.get_variable("fc1_weights",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层需要正则化
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        #偏移项
        fc1_biases = tf.get_variable("fc1_biases",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        #全连接前向传播过程,矩阵乘法和激活函数
        fc1 = tf.nn.relu(tf.matmul(reshape,fc1_weights) + fc1_biases)
        #训练时候加入dropout
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    #声明第六层全连接层前向传播过程,输入为长度512向量,输出为10向量，通过softmax之后获得最后分类结果
    with tf.variable_scope("layer6_fc2"):
        fc2_weights = tf.get_variable("fc2_weights",[FC_SIZE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))

        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable("fc2_biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1,fc2_weights) + fc2_biases

    return logit






