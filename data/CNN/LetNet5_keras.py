import keras
from keras.datasets import mnist
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D
import tensorflow as tf
import numpy as np

def LetNet():
    img_row,img_cols = 28,28
    num_class = 10

    #加载mnis数据,train为60000x28x28 trainY 为每张图片对应数字
    (trainX,trainY),(testX,testY) = mnist.load_data()

    #增加维度 黑白图片为1 彩色3  根据底层支持的模型
    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0],1,img_row,img_cols)
        testX = testX.reshape(testX.shape[0],1,img_row,img_cols)
        input_shape = (1,img_row,img_cols)
    else:
        trainX = trainX.reshape(trainX.shape[0],img_row,img_cols,1)
        testX = testX.reshape(testX.shape[0],img_row,img_cols,1)
        input_shape = (img_row,img_cols,1)

    #图像转化为0-1之间的实数
    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255.0
    testX /= 255.0

    #将标准答案转化为需要的格式
    trainY = keras.utils.to_categorical(trainY, num_class)
    testY = keras.utils.to_categorical(testY, num_class)

    #时殷弘keras定义模型

    model = Sequential()
    #第一层深度32 过滤器5x5 卷积层
    model.add(Conv2D(6,kernel_size=(5,5),activation='relu',input_shape=input_shape))
    #过滤器大小2x2最大池化层
    model.add(MaxPool2D(pool_size=(2,2)))
    #深度64 过滤器5x5 卷积层
    model.add(Conv2D(16,kernel_size=(5,5),activation='relu'))
    #过滤层2x2最大池化层
    model.add(MaxPool2D(pool_size=(2,2)))
    #卷积层输出拉直作为全连接层的输入
    model.add(Flatten())

    #全连接层 500个节点
    # model.add(Dense(500,activation='relu'))

    #120节点
    model.add(Dense(120,activation='relu'))

    #84节点
    model.add(Dense(84,activation='relu'))

    #全连接层,最后输出
    model.add(Dense(num_class,activation='softmax'))

    # keras.utils.multi_gpu_model(model,gpus=2)

    #定义损失函数 优化函数 评测方法
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])

    #进行训练，输入训练数据 验证数据 batch带线啊哦
    model.fit(trainX,trainY,batch_size=128,epochs=200,validation_data=(testX,testY))

    #在测试数据上计算准确率
    score = model.evaluate(testX,testY)

    # model.predict()

    #model.save('/root/ML/MODEL/LetNet5/LetNet5-keras.h5')
    print('测试损失为: ', score[0])
    print('测试正确率: ', score[1])


def image(image_file):
    sess = tf.Session()
    image_raw = tf.gfile.FastGFile(image_file, 'rb').read()
    # 编码成为三维矩阵
    image_data_pr = tf.image.decode_png(image_raw)
    image_data_pr = tf.image.rgb_to_grayscale(image_data_pr)
    # 数据类型转换为实数
    image_data = tf.image.convert_image_dtype(image_data_pr, dtype=tf.float32)
    return  sess.run(image_data)



def LetNet5_predict(image_file):
    image_data = image(image_file)

    #加载模型
    from keras.models import load_model
    model = load_model("D:\share\learn\ML\MODEL\letnet5\LetNet5-keras.h5")

    y_predict = model.predict(tf.expand_dims(image_data,0))

    print('预测数字为 %s' % np.argmax(y_predict,axis=1))