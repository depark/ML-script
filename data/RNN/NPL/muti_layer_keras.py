#更丰富的网络结构定义

import keras
from keras.datasets import mnist
from keras.layers import Input,Dense
from keras.models import Model
from keras import backend as K

img_row,img_cols = 28,28
num_class = 10

#加载mnis数据,train为60000x28x28 trainY 为每张图片对应数字
(trainX,trainY),(testX,testY) = mnist.load_data()


#图像转化为0-1之间的实数
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0
trainX = trainX.reshape(len(trainX),28*28)
testX = testX.reshape(len(testX),28*28)

#将标准答案转化为需要的格式
trainY = keras.utils.to_categorical(trainY, num_class)
testY = keras.utils.to_categorical(testY, num_class)

#定义输入,不考虑batch-size
inputs = Input(shape=(784,))
#定义全连接层 500节点,relu激活函数
x = Dense(500,activation='relu')(inputs)

#定义输出层，加上softmax进行输出
predictions = Dense(10,activation='softmax')(x)


#通过model类创建模型,model类在初始化时候需要指定模型输入输出
model = Model(inputs=inputs,outputs=predictions)

#定义损失优化评测
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='SGD',
              metrics=['accuracy'])


#训练
model.fit(trainX,trainY,batch_size=128,epochs=20,validation_data=(testX,testY))