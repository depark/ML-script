#TFRecode 解析文件

import tensorflow as tf


TFilenema = '/root/ML/data/ini_data/tfrecords/mnist.tfrecords'

reader = tf.TFRecordReader()

filename_queue = tf.train.string_input_producer([TFilenema])

#从文件中读取一个样例,read_up_to读取多个样例
_,serialized_example = reader.read(filename_queue)

#解析读入的一个样例,解析多个可以用parse_example
features = tf.parse_single_example(
    serialized_example,
    features={
        #提供两种不同的解析方法,tf.FixedLenFeature 解析结果为一个tensor,tf.VarLenFeature解析结果为 SparseTensor，用于处理稀疏数据
        'image_raw': tf.FixedLenFeature([],tf.string),
        'pixels': tf.FixedLenFeature([],tf.int64),
        'label': tf.FixedLenFeature([],tf.int64)
    }
)

#解析成图像对应像素组
image = tf.decode_raw(features['image_raw'],tf.uint8)
label = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)

sess = tf.Session()
#启用多线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#每次获取TFRecord文件中一个样例,获取完重投获取

for i in range(10):
    print(sess.run([image,label,pixels]))