#PTB数据集预处理,原始数据转化为batch

import numpy as np
import tensorflow as tf
import codecs
import collections
from operator import itemgetter

RAW_DATA = "/root/ML/data/ini_data/PTB/data/ptb.train.txt"
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35


#将元素数据单词转化成数值,词汇字典,写入词汇表文件
def change_fileto_num(raw_file,vocab_out):
    counts = collections.Counter()
    with codecs.open(raw_file,'r','utf8') as f:
        for line in f:
            for word in line.strip().split():
                counts[word] +=1
    #对单词频率进行排序
    sorted_words_to_cnt = sorted(counts.items(),key=itemgetter(1),reverse=True)
    #采集单词
    sorted_words = [x[0] for x in sorted_words_to_cnt]
    #文本换行出加入句子结束符 <eos>
    # sorted_words = ["<eos>"] + sorted_words
    #机器翻译转化加入<unk>和句子起始符加入词汇表<sos>，并删除低频词汇
    sorted_words = ["<unk>","<sos>","<eos>"] + sorted_words
    if len(sorted_words) > 10000:
        sorted_words = sorted_words[:10000]
    #写入词汇表文件
    with codecs.open(vocab_out,'w','utf-8') as file_out:
        for word in sorted_words:
            file_out.write(word+"\n")

    #建立词汇到单词的编号映射
    word_to_id = { k: v for (k,v) in zip(sorted_words,range(len(sorted_words)))}
    return word_to_id


#获取单词编号,如果获取的是低频词,替换为"<unk>"
def get_id(word,word_to_id):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

#将原始数据文件转换为词汇数字内容文件
def raw_to_num(fin,fout,vocab_out):
    word_to_id =change_fileto_num(fin,vocab_out)
    fin_obj = codecs.open(fin,'r','utf-8')
    fout_obj = codecs.open(fout,'w','utf-8')
    for line in fin_obj:
        words = line.strip().split() + ["<eos>"]
        out_line = ' '.join([str(get_id(w,word_to_id)) for w in words]) + '\n'
        fout_obj.write(out_line)
    fin_obj.close()
    fout_obj.close()

#将处理好的词汇编码转换成list
def read_data(file_path):
    with open(file_path,'r') as fin:
        #将所有行连接成一个句子
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list= [int(w) for w in id_string.split()]
    return id_list

#生成batch数据分片
def make_batch(id_list, batch_size, num_step):
    #总batch数量
    num_batch = (len(id_list) - 1) // (batch_size * num_step)

    data = np.array(id[: num_batch * batch_size * num_step])
    data = np.reshape(data,[batch_size, num_batch * num_step])
    #分成num_batch 个batch,存入数组
    data_batchs = np.split(data,num_batch,axis=1)

    #生成label,在当前数值往后移一位
    label_data = np.array(id[1: num_batch * batch_size * num_step + 1])
    label_data = np.reshape(label_data, [batch_size, num_batch * num_step])
    # 分成num_batch 个batch,存入数组
    label_data_batchs = np.split(label_data, num_batch, axis=1)

    return list(zip(data_batchs,label_data_batchs))