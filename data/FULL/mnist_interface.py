#  定义前向传播过程和参数

#mnist数据模型

import tensorflow as tf


#初始化参数
#输入一维矩阵
INPUT_NODE = 28*28
#图片像素大小
IMAGE_SIZE = 28
#图层数
NUM_CHANNELS = 1

#输出矩阵列
OUTPUT_NODE = 10
#第一个隐藏层节点数
LAYER1_NODE = 500

def get_weight_variables(shape, regularizer):
    '''
    定义初始化变量和正则化变量
    :param shape: 变量维度
    :param regularizer:  正则化函数
    :return:  变量矩阵
    '''
    weights = tf.get_variable("weights",shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection("losses",regularizer(weights))
    return weights

#定义前向传播过程
def interface(input_tensor,regularizer,reuse=None):
    #定义第一层,变量和偏置项
    with tf.variable_scope("layer1",reuse=reuse):
        #定义第一层变量
        weights = get_weight_variables([INPUT_NODE,LAYER1_NODE],regularizer)
        #定义第一层偏置项
        biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
    #定义第二层
    with tf.variable_scope("layer2",reuse):
        weights = get_weight_variables([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights) + biases

    return layer2

