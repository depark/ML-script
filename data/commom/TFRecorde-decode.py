#TFrecord 图像数据处理,生成TDRecord数据文件

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


mnis_path = '/root/ML/data/ini_data/mnis_data'



#生成整数型
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#生成字符串属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnis = input_data.read_data_sets(mnis_path,dtype=tf.uint8,one_hot=True)

images = mnis.train.images
labels = mnis.train.labels

#图像分辨率
pixels = images.shape[1]
num_examples = mnis.train.num_examples

#输出TFrecord文件
TFilenema = '/root/ML/data/ini_data/tfrecords/mnist.tfrecords'

writer = tf.python_io.TFRecordWriter(TFilenema)

for index in range(num_examples):
    #图像矩阵转化成字符串
    image_raw = images[index].tostring()
    #将一个样例转化为Example Protocol Buffer 讲所有数据写入这个结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()
