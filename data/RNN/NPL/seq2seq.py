#Seq2seq模型训练和测试
import tensorflow as tf
import codecs

#转为单词编号的文件,英文翻译为中文
SRC_TRAIN_DATA = '/root/ML/data/ini_data/WMT/en-zh/data/train.en'
TRG_TRAIN_DATA = '/root/ML/data/ini_data/WMT/en-zh/data/train.zh'

VOCAB_EN = '/root/ML/data/ini_data/WMT/en-zh/data/train.en.vocab'
VOCAB_ZH = '/root/ML/data/ini_data/WMT/en-zh/data/train.zh.vocab'


CHECKPOINT_PATH = '/root/ML/MODEL/SEQ/seq2seq_ckpt' #模型保存路径
HIDDEN_SIZE = 1024                 # 隐藏层节点数
NUM_LAYERS = 2                     # 深层神经网络lstm层数
SRC_VOCAB_SIZE = 10000             # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000              # 目标语音词汇表大小
BATCH_SIZE = 100                   # BATCH大小
NUM_EPOCH = 5                      # 训练轮数
KEEP_PROB = 0.8                    # 节点不被drop的概率
MAX_GRAD_NORM = 5                  # 控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True       # softmax和词向量之间恭喜参数

MAX_LEN = 80                        # 每个句子最大单词数
SOS_ID = 1                          # sos 在词汇库中编号
EOS_ID = 2                          # eos词汇中编号

#定义NMTModel模型
class MNTModel(object):
    def __init__(self):
        #定义编码和解码使用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)
        ])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)
        ])
        #源语言和目标语音定义词向量
        self.src_embedding = tf.get_variable('src_emb',[SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb',[TRG_VOCAB_SIZE, HIDDEN_SIZE])

        #定义softmax层变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
                "softmax_weight",[HIDDEN_SIZE,TRG_VOCAB_SIZE]
            )
        self.softmax_bias = tf.get_variable("softmax_biax",[TRG_VOCAB_SIZE])

    #定义前向计算图
    def forward(self,src_input,src_size,trg_input,trg_label,trg_size):
        #输入句子长度
        batch_size = tf.shape(src_input)[0]

        #将输入和输出转换为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding,src_input)
        trg_emb = tf.nn.embedding_lookup(self.src_embedding, trg_input)

        #在词向量进行dropout
        src_emb = tf.nn.dropout(src_emb,KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        #使用dynamic_rnn构造编码器
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32
            )

        #构造解码器
        with tf.variable_scope("decoder"):
            dec_outputs, dec_state = tf.nn.dynamic_rnn(
                self.dec_cell, trg_emb, trg_size, initial_state = enc_state
            )

        #计算每一步的log perplexity
        #获得最顶层的神经网络输出
        output = tf.reshape(dec_outputs, [-1,HIDDEN_SIZE])
        #全连接层
        logits = tf.matmul(output,self.softmax_weight) + self.softmax_bias
        #计算交叉熵损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,labels=tf.reshape(trg_label,[-1])
        )

        #计算平均损失,
        #将填充位的权重设置为0
        label_weights = tf.sequence_mask(
            trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32
        )
        label_weights = tf.reshape(label_weights,[-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        #定义反向传播操作
        trainable_variables = tf.trainable_variables()

        #控制梯度大小 定义优化算法和训练步骤
        grads = tf.gradients(cost / tf.to_float(batch_size),trainable_variables)
        grads,_ = tf.clip_by_global_norm(grads,MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(
            zip(grads,trainable_variables)
        )
        return cost_per_token, train_op

    #训练一个epoch，返回全局步数
    def run_epoch(self, sess, cost_op, train_op, saver, step):
        #循环获取dataset数据
        while True:
            try:
                cost,_ = sess.run([cost_op,train_op])
                if step % 10 == 0:
                    print('经过 %d 轮训练, per token cost is  %.3f' % (step, cost))
                #每200步保存一个checkpoint
                if step % 10 ==0:
                    saver.save(sess,CHECKPOINT_PATH,global_step = step)
                step += 1
            except tf.errors.OutOfRangeError:
                break
        return step

    #定义解码过程 翻译过程,输入为一个英文句子,输出为一个中文句子
    def interface(self, src_input):
        src_size = tf.convert_to_tensor([len(src_input)], dtype = tf.int32)
        src_input = tf.convert_to_tensor([src_input],dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding,src_input)

        #构造编码器
        with tf.variable_scope("encoder"):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(
                self.enc_cell, src_emb, src_size, dtype=tf.float32
            )

        #设置最大步数,防止出现无限循环
        MAX_DEC_LEN = 100
        with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
            # 使用一个边长的tensorarry存储生成的句子
            init_arry = tf.TensorArray(dtype=tf.int32, size=0,dynamic_size=True, clear_after_read=False)
            #填入第一个单词<sos>作为解码器的输入
            init_arry = init_arry.write(0, SOS_ID)
            #构建初始的循环状态,包含循环神经网络隐藏状态,保存生成句子的TensorArry，以及记录解码步数的一个整数step
            init_loop_var = (enc_state, init_arry, 0 )
            #循环条件: 循环直到解码器输出<eos> 或者达到最大步数
            def continue_loop_condition(state,trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step,MAX_DEC_LEN-1)
                ))

            def loop_body(state, trg_ids, step):
                #读取最后一步输出的单词,读取骑词向量
                trg_input = [ trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
                dec_outputs,next_state = self.dec_cell.call(
                    state = state, inputs=trg_emb
                )
                #计算每个可能的输出单词对应的logits,选取logit值最大的单词作为输出
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output,self.softmax_weight) + self.softmax_bias)
                print('======================')
                print(logits)
                next_id = tf.argmax(logits,axis=1 , output_type=tf.int32)
                #将这一步输出写入trg_ids
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1
            #执行tf.while_loop 返回最终状态
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var
            )
            return trg_ids.stack()



#数据处理
class HangdlerDataset(object):
    def __init__(self):
        pass

    #对已经转化为单词编号的文件进行调整,输出为整数格式的 句子和句子长度
    def MakeDataset(self,file_path):
        dataset = tf.data.TextLineDataset(file_path)
        #根据空格将单词编号且分开放入一个一维向量
        dataset = dataset.map(lambda string: tf.string_split([string]).values)
        #字符串形式的编号转化为整数
        dataset = dataset.map(lambda string: tf.string_to_number(string,tf.int32))
        #统计每个句子的单词数量, 与句子内容一同保存到dataset
        dataset = dataset.map(lambda x: (x, tf.size(x)))
        return dataset

    #分别对源语言和目标语言文件进行处理,进行填充和batching
    def MakeSrcTrgDataset(self,src_path,trg_path,batch_size):
        #读取源文件和目标文件,并进行处理为句子和句子长度
        src_data = self.MakeDataset(src_path)
        trg_data = self.MakeDataset(trg_path)
        #通过zip操作将两个数据和并为一个,每项数据为 [源句子,源句子长度,目标句子,目标句子长度]
        dataset = tf.data.Dataset.zip((src_data,trg_data))
        #删除内容为空和长度过长的句子
        def FilterLength(src_tunlp,trg_tunlp):
            ((src_input,src_len),(trg_label,trg_len)) = (src_tunlp,trg_tunlp)
            #过滤出句子长度大于1 小于50
            src_len_ok = tf.logical_and(
                tf.greater(src_len,1),tf.less_equal(src_len,MAX_LEN)
            )
            trg_len_ok = tf.logical_and(
                tf.greater(trg_len,1),tf.less_equal(trg_len,MAX_LEN)
            )
            return tf.logical_and(src_len_ok,trg_len_ok)
        dataset = dataset.filter(FilterLength)

        #解码器目标输出 1. <sos> X Y Z 2. X Y Z <eos>
        def MakeTrgInput(src_tuple, trg_tuple):
            ((src_input,src_len),(trg_label, trg_len)) = (src_tuple,trg_tuple)
            #去除句子最后一个<eos> 最前面加上<sos>
            trg_input = tf.concat([[SOS_ID],trg_label[:-1]],axis=0)
            return ((src_input,src_len),(trg_input,trg_label, trg_len))
        dataset = dataset.map(MakeTrgInput)

        dataset = dataset.shuffle(10000)
        #定义输出维度,维度((src_input,src_len),(trg_input,trg_label, trg_len))
        padded_shape = (
            (tf.TensorShape([None]),    # 源句子长度未知向量
            tf.TensorShape([])),        # 源句子长度为单个数字
            (tf.TensorShape([None]),    # 解码器输入/输出 长度未知变量
            tf.TensorShape([None]),
            tf.TensorShape([])),        # 目标句子长度一个数字
        )
        batched_dataset = dataset.padded_batch(batch_size,padded_shape)
        return batched_dataset

#训练
def train_main():
    #定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05,0.05)

    #定义循环用的神经网络
    with tf.variable_scope('nmt_model',reuse=None,initializer=initializer):
            train_model = MNTModel()

    #定义数据
    data = HangdlerDataset().MakeSrcTrgDataset(SRC_TRAIN_DATA,TRG_TRAIN_DATA,BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src,src_size),(trg_input,trg_label,trg_size) = iterator.get_next()

    #定义前向计算图,将数据传给前向传播
    cose_op, train_op = train_model.forward(src,src_size,trg_input,trg_label,trg_size)

    #训练模型
    saver = tf.train.Saver()
    step = 0

    config = tf.ConfigProto(device_count={"CPU": 30},
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0
                            )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print('进行第 %d 轮迭代' % (i+1))
            sess.run(iterator.initializer)
            step = train_model.run_epoch(sess,cose_op,train_op,saver,step)

#转化句子为数字
def emb_word(sentence,lanag):
    #英语,将英语单词转化为数字
    if lanag == 'en':
        with codecs.open(VOCAB_EN, 'r', 'utf-8') as vocab_obj:
            vocab_word = [w.strip() for w in vocab_obj.readlines()]
        word_dic_en = {k:v for (k,v) in zip(vocab_word,range(len(vocab_word)))}
        chg_sentence = [ int(word_dic_en[w.strip()]) for w in sentence.split()] + [EOS_ID]
    else:
        vocabfile = VOCAB_ZH
        with codecs.open(vocabfile,'r','utf-8') as vocab_obj:
            vocab_word = [ w.strip() for w in vocab_obj.readlines()]
        word_dic_zh = {v:k for (k, v) in zip(vocab_word, range(len(vocab_word)))}
        chg_sentence = ''.join([word_dic_zh[int(w)] for w in sentence])
    return chg_sentence

#翻译过程
def decode_main(sentence):
    # 定义训练用的神经网络模型
    with tf.variable_scope("nmt_model", reuse=None):
        model = MNTModel()

    ckpt = tf.train.get_checkpoint_state('/root/ML/MODEL/SEQ/')
    #定义一个测试数据
    test_sentence = emb_word(sentence,'en')
    #建立解码所需计算图
    output_op = model.interface(test_sentence)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,'/root/ML/MODEL/SEQ/seq2seq_ckpt-100')

    #读取翻译结果
    output = sess.run(output_op)
    print('使用模型 %s' % ckpt.model_checkpoint_path)
    print(output)
    print(emb_word(output,'zh'))
    sess.close()

if __name__ == "__main__":
    sen = 'This turns out'
    decode_main(sen)
    # train_main()