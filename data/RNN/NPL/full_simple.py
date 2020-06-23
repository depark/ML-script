#完整的LSTM神经网络模型,预测自然语音
import numpy as np
import tensorflow as tf
import os
import sys,logging
sys.path.append('/root/ML')
from data.RNN.NPL.PTB_pre import *


FILE_PATH = '/root/ML/data/ini_data/PTB/data/'
os.chdir(FILE_PATH)

#各种数据
TRAIN_DATA = "ptb.train"
EVAL_DATA = "ptb.valid"
TEST_DATA = "ptb.test"
HIDDEN_SIZE = 800                       #隐藏层节点数
NUM_LAYERS = 4                          #LSTM网络结构层数
VOCAB_SIZE = 10000                      #词典规模
TRAIN_BATCH_SIZE = 20                   #训练数据batch大小
TRAIN_NUM_STEP = 35                     #训练数据截断长度,每次输入的单词长度

EVAL_BATCH_SIZE = 1                     #测试数据batch大小
EVAL_NUM_STEP = 1                       #测试数据截断长度
NUM_EPOCH = 20                           #使用训练数据轮数
LSTM_KEEP_PROB = 0.9                    #LSTM节点dropout,保留概率
EMBEDDING_KEEP_PROB = 0.9               #词向量保留概率
MAX_GRAD_NORM = 5                       #控制梯度膨胀的梯度大小上线
SHARE_EMB_AND_SOFTMAX = True            #在softmax和词向量层共享参数

LOG_FILE = '/opt/NPL.log'


logging.basicConfig(filename=LOG_FILE,level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class PTBModel():
    #定义前向传播和反向传播过程
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        #定义LSTM输入和输出
        self.input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])

        #定义LSTM结构,使用dropout的深层神经循环网络
        dropout_keep_prod = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cell = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prod)
            for _ in range(NUM_LAYERS)
        ]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
        #初始化最初状态,全0向量
        self.initial_status = cell.zero_state(batch_size,tf.float32)

        #定义单词词向量矩阵
        embedding = tf.get_variable("embedding",[VOCAB_SIZE,HIDDEN_SIZE])

        #输入单词转化为词向量
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)

        #只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs,EMBEDDING_KEEP_PROB)
            #定义输出列表，将不同的lstm结构的输出收集起来再一起传给softmax层
        outputs = []
        state= self.initial_status
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step >0:tf.get_variable_scope().reuse_variables()
                cell_out,state = cell(inputs[:, time_step, :],state)
                outputs.append(cell_out)
        #输出转换为[HIDDEN_SIZE]
        output = tf.reshape(tf.concat(outputs,1),[-1,HIDDEN_SIZE])

        #softmax层, 将RNN在每个位置上的输出转换为各个单词的logits
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight",[HIDDEN_SIZE,VOCAB_SIZE])
        bias = tf.get_variable("bias",[VOCAB_SIZE])
        logits = tf.matmul(output,weight) + bias

        #定义交叉熵损失函数和平均损失函数
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets,[-1]),
            logits=logits
        )
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        #只在训练时定义反向传播操作
        if not is_training:return None

        trainable_variables = tf.trainable_variables()

        #控制梯度大小,定义优化方法和训练步骤
        grads,_ = tf.clip_by_global_norm(
            tf.gradients(self.cost,trainable_variables),MAX_GRAD_NORM
        )
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(grads,trainable_variables))

    # @property
    # def train_op(self):
    #     return self.train_op

#前向传播计算过程
def run_epoch(sess,model,batches,train_op,output_log,step):
    #计算平均perplexity辅助变量
    total_costs = 0
    iters = 0
    state = sess.run(model.initial_status)
    #训练一个epoch
    for x,y in batches:
        #在当前batch上允许train_op并计算损失值,交叉熵损失函数计算的就是下一个单词为给定单词的概率
        cost,state,_ = sess.run(
            [model.cost,model.final_state,train_op],
            {model.input_data: x, model.targets: y,model.initial_status: state}
        )
        total_costs += cost
        iters += model.num_steps

        #只有在训练时输出日志
        if output_log and step % 100 == 0:
            logger.info("经过 %d 伦训练, perplexity 是 %.3f" % (step,np.exp(total_costs / iters)))
        step += 1
    return step, np.exp(total_costs / iters)

#将输入数据文件转化为数值文件
# def read_data(file_path):
#     #词汇列表
#     out_words = []
#     word_to_id = change_fileto_num(file_path)
#     fin_obj = codecs.open(file_path,'r')
#     for line in fin_obj:
#         words = line.strip().split() + ["<eos>"]
#         out_words.extend([int(get_id(word,word_to_id)) for word in words])
#     fin_obj.close()
#     return out_words

def make_batches(id_list,batch_size, num_step):
    # 总batch数量
    num_batch = (len(id_list) - 1) // (batch_size * num_step)

    data = np.array(id_list[: num_batch * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batch * num_step])
    # 分成num_batch 个batch,存入数组,每个单独的batch为 [batch_size,num_step]
    data_batchs = np.split(data, num_batch, axis=1)

    # 生成label,在当前数值往后移一位
    label_data = np.array(id_list[1: num_batch * batch_size * num_step + 1])
    label_data = np.reshape(label_data, [batch_size, num_batch * num_step])
    # 分成num_batch 个batch,存入数组
    label_data_batchs = np.split(label_data, num_batch, axis=1)

    return list(zip(data_batchs, label_data_batchs))

def main():

    #定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05,0.05)

    #定义训练用的循环神经网卡模型
    with tf.variable_scope("lanaguage_model",reuse=None, initializer=initializer):
        train_model = PTBModel(True,TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)

    #定义测试用循环神经网络
    with tf.variable_scope("lanaguage_model",reuse=True, initializer=initializer):
        eval_model = PTBModel(False,EVAL_BATCH_SIZE,EVAL_NUM_STEP)

    config = tf.ConfigProto(device_count={"CPU": 30},
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0
                            )
    config.gpu_options.allow_growth = True

    #训练模型
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        train_batches = make_batches(read_data(TRAIN_DATA),TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)
        eval_batches = make_batches(read_data(EVAL_DATA),EVAL_BATCH_SIZE,EVAL_NUM_STEP)
        test_batches = make_batches(read_data(TEST_DATA),EVAL_BATCH_SIZE,EVAL_NUM_STEP)
        step = 0
        logger.info('''
            训练参数:  
                节点数: %d
                网络层数: %d
                迭代次数: %d        
        ''' % (HIDDEN_SIZE, NUM_LAYERS, NUM_EPOCH))
        for i in range(NUM_EPOCH):
            logger.info("第 %d 伦迭代" % (i + 1))
            step,train_pplx = run_epoch(sess,train_model,train_batches,train_model.train_op,True,step)
            logger.info("第 %d 伦 训练perplexity: %.3f" % (i+1,train_pplx))
            _,eval_pplx = run_epoch(sess,eval_model,eval_batches,tf.no_op(),False,0)
            logger.info("第 %d 伦 验证perplexity: %.3f" % (i + 1, eval_pplx))
        _, test_pplx = run_epoch(sess, eval_model, test_batches, tf.no_op(), False, 0)
        logger.info("测试perplexity: %.3f" % test_pplx)

if __name__ == "__main__":
    main()