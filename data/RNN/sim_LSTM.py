#定义一个LSTM网络前向传播

import tensorflow as tf

#定义隐藏层的长度
lstm_hidden_size = 10
#每次训练数
batch_size = 100
#训练数据长度
num_step = 100

#定义一个基本的lstm网络
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

#定义一个深层循环网络,num_layers表示有多少个lstm层
stack_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [ lstm for _ in range(num_layers)]
)

#定义初始状态为全0数组
state = lstm.zero_state(batch_size, tf.float32)

#深度循环网络初始状态全0
state = stack_lstm.zero_state(batch_size, tf.float32)


#损失
loss = 0.0

for i in range(num_step):
    #第一次定义变量,之后复用之前变量
    if i >0: tf.get_variable_scope().reuse_variables()
    #定义lstm网络前向 current_input为输入 state为前一个时刻状态(h,c), 返回输出(h) 和更新后的状态(h,c)
    lstm_output,state = lstm(current_input, state)

    #深度网络前向传播
    stacked_lstm_output,state = stack_lstm(current_input, state)


    #当前输出传入全连接层得到最后输出
    final_output = full_connect(lstm_output)

    #定义损失函数
    loss += calc_loss(final_output,expect_output)
