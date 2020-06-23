#定义神经网络模型测试过程
import sys
sys.path.append('/root/ML')
import tensorflow as tf
from data.FULL import mnist_interface,mnist_train
from data.CNN import interface
from time import sleep

import numpy as np
from PIL import Image


def ImageToMatrix(image):

    im = Image.open(image)

    im = im.convert("L")
    data = im.getdata()

    data = np.matrix(data,dtype=np.float32)/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(1,28*28))
    return new_data


def evaluate():
    #定义输入输出
    x = tf.placeholder(tf.float32,[None,mnist_interface.INPUT_NODE],name="x-input")

    y_ = tf.placeholder(tf.float32,[None,mnist_interface.OUTPUT_NODE], name="y-output")

    validate_feed = {
        x: mnist_train.mnist.validation.images,
        y_: mnist_train.mnist.validation.labels
    }
    #正向传播过程
    #输入转换
    x_image = tf.reshape(x,[-1,28,28,1])

    y = interface.interface(x_image,None, None)

    #正向传播结果正确率
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #加载模型,和滑动平均变量
    variable_average = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVEAGE_DECAY)

    variables_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    while True:
        with tf.Session() as sess:
            #自动找到最新模型文件名
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)

            # for model in  ckpt.all_model_checkpoint_paths:
            if ckpt and ckpt.model_checkpoint_path:
                print('Model '+ ckpt.model_checkpoint_path)
                saver.restore(sess,ckpt.model_checkpoint_path)
                #获取迭代伦数
                global_step = ckpt.model_checkpoint_path.split("-")[-1]

                accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                # all_result = []
                # for png in os.listdir("D:\learn\ML\\test-photo"):
                #     im = ImageToMatrix("D:\learn\ML\\test-photo\\" + png)
                #     y = mnist_interface.interface(im, None)
                #     result = tf.argmax(y, 1)
                #     all_result.append(sess.run(result))
                print("第 %s 伦训练模型在测试集上表现 %g." % (global_step,accuracy_score))
            else:
                continue
        sleep(10)

if __name__ == "__main__":
    evaluate()