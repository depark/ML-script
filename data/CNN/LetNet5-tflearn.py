import tflearn
from tflearn.layers.core import input_data,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnis



#读取mnist数据
trainX,trainY, testX, testY = mnis.load_data('/root/ML/data/ini_data/mnis_data',one_hot=True)

#转化成卷积网络输入格式
trainX = trainX.reshape([-1,28,28,1])
testX = testX.reshape([-1,28,28,1])

#构建神经网络,定义一个输入数据格式
net = input_data(shape=[None,28,28,1],name='input')

#封装一个深度为32 过滤器5X5 激活函数为Relu的卷积层
net = conv_2d(net,32,5,activation='relu')
#定义过滤器为2X2的最大池化层
net = max_pool_2d(net,2)
#定义其他层卷积层,池化层，全连接层
net = conv_2d(net,64,5,activation='relu')
net = max_pool_2d(net,2)
net = fully_connected(net,500,activation='relu')
net = fully_connected(net,10,activation='softmax')

#使用tflearn封装好的函数定义学习任务,指定优化器sgd,学习率0.01，损失函数为交叉熵
net = regression(net,optimizer='sgd', learning_rate=0.01,loss='softmax_categorical_crossentropy')

#定义的网络结果训练模型，并在指定的而验证数据上验证模型的效果,tflearn将模型的训练过程封装到类中
model = tflearn.DNN(net,tensorboard_verbose=0)

#对指定数据进行训练
model.fit(trainX,trainY,n_epoch=20,validation_set=([testX,testY]),show_metric=True)