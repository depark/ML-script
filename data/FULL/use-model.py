#使用模型预测

import tensorflow as tf
from PIL import Image
import numpy as np
import os
from data.FULL import  mnist_interface,mnist_train

def ImageToMatrix(image):

    im = Image.open(image)

    im = im.convert("L")
    data = im.getdata()

    data = np.matrix(data,dtype=np.float32)/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(1,28*28))
    return new_data




if __name__ == "__main__":
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("D:\learn\ML\MODEL")
        saver.restore(sess, ckpt.model_checkpoint_path)
        for png in os.listdir("D:\learn\ML\\test-photo"):
            im = ImageToMatrix("D:\learn\ML\\test-photo\\"+png)
            y = mnist_interface.interface(im,None)
            result = tf.argmax(y,1)
            print("%s 结果为 %s" % (png,sess.run(result)))