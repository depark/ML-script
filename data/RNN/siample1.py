#简单循环神经网络前向传播

import numpy as np





def Rnn_interface(x,state):
    # 定义状态权重和输入权重
    # 状态权重
    w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    # 输入权重
    w_cell_input = np.asarray([0.5, 0.6])
    # 偏移量
    b_cell = np.asarray([0.1, -0.1])

    # 定义输出的全连接层参数
    w_output = np.asarray([[1.0], [2.0]])
    b_output = 0.1

    before_active = np.dot(state,w_cell_state) + x * w_cell_input + b_cell

    state = np.tanh(before_active)

    output = np.dot(state,w_output) + b_output

    print('全连接神经网络 %s' % before_active)
    print('状态值 %s' % state)
    print('输出值 %s' % output)
    return state,output

#输入向量
X = [1,2]
#初始状态
state = [0.0,0.0]
outputs = []

for i,x in enumerate(X):
    state,output = Rnn_interface(x,state)
    outputs.append(output)

