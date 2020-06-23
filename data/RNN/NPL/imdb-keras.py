#keras 对情感分析模型
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM
from keras import regularizers


#最多使用的单词书
max_features = 20000
#循环神经网络截断长度
maxlen = 80
batch_size = 32

#加载数据,并将但粗转化为统一的ID
(trainX,trainY), (testX,testY) = imdb.load_data(num_words=max_features)

#统一语句长度,长度不够的使用默认值0填充,超过的长度忽略超过的部分
trainX = sequence.pad_sequences(trainX,maxlen=maxlen)
testX = sequence.pad_sequences(testX,maxlen=maxlen)


#进行建模
model = Sequential()

#构建embedding层, 128代表向量维度 对应词向量为[ 20000,128],128为隐藏层节点
model.add(Embedding(max_features,128))

#构建LSTM层,输出只会得到最后一个节点输出,如需要输出每个时间点的结果 return_sequences设置true
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2,activity_regularizer=regularizers.l2(0.01),kernel_regularizer=regularizers.l1_l2(0.01,0.1)))
#构建全连接层,输出只有一个节点，激活函数为sigmoid
model.add(Dense(1,activation='sigmoid'))

#定义损失函数 优化函数和评测指标
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#训练,指定训练数据
model.fit(
    trainX,trainY,batch_size=batch_size,
    epochs=20,
    validation_data=(testX,testY),
    use_multiprocessing=True,
    workers=25)

#在测试数据上评测模型
score = model.evaluate(testX,testY,batch_size=batch_size)
print('测试数据损失: %s' % score[0])
print('测试数据正确率: %s' % score[1])