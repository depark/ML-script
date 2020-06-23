#预处理mnis数据,将mnis训练数据分成10个文件,每个文件写5500条数据,生成数据文件

import tensorflow as tf
import os,glob
import numpy as np
import threading,queue


from tensorflow.examples.tutorials.mnist import input_data


mnis_path = '/root/ML/data/ini_data/mnis_data'

#生成整数型
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#生成字符串属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def mnis_decode():
    #生成mnis的文件
    #写入文件个数
    num_shares = 10

    #每个文件写入数据数量
    instance_per_shard = 5500

    #写入文件路径
    file_path = '/root/ML/data/ini_data/tfrecords/mnist'

    mnis = input_data.read_data_sets(mnis_path,dtype=tf.float32,one_hot=True)
    images = mnis.train.images
    labels = mnis.train.labels

    pixels = images.shape[1]

    for i in range(num_shares):
        filename = os.path.join(file_path,'mnist.tfrecord-%.5d-of-%.5d' % (i,num_shares))
        writer = tf.python_io.TFRecordWriter(filename)
        #数据封装为example接口写入文件
        batched = mnis.train.next_batch(instance_per_shard)
        for j in range(instance_per_shard):
            image_raw = batched[0][j].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'pixels': _int64_feature(pixels),
                'label': _int64_feature(np.argmax(batched[1][j])),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        writer.close()

def _handler_image(file_list,current_label):
    sess = tf.Session()
    while not file_list.empty():
        t= threading.currentThread()
        image = file_list.get()
        print('%s: %d --> 还剩 %d 张图片' % (t.getName(),current_label,file_list.qsize()))
        image_raw = tf.gfile.FastGFile(image, 'rb').read()
        image_data = sess.run(tf.image.decode_image(image_raw))
        height, width, channel = image_data.shape

        if np.random.randint(100) < 70:
            image_raws.append(image_raw)
            labels.append(current_label)
            heights.append(height)
            widths.append(width)

def flower_image():
    global image_raws, labels, heights, widths, label_dict
    #原始数据转换像素矩阵
    INPUT_DATA = "/root/ML/data/ini_data/flower_photos"
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    with tf.Session() as sess:
        current_label = 0
        label_dict = {}
        image_raws = []
        labels = []
        heights = []
        widths = []
        for sub_dir in sub_dirs[1:]:
            file_list = queue.Queue()
            print('====================')
            label_dict[current_label] = sub_dir.split('/')[-1]
            #收集对应目录下图片
            glob_file = os.path.join(sub_dir,'*.jpg')
            for file in glob.glob(glob_file):
                file_list.put(file)
            num = file_list.qsize()
            print('总共 %d 图片' % num)
            threads = [
                threading.Thread(target=_handler_image,args=(file_list,current_label)) for i in range(20)
            ]
            for t in threads:
                t.start()
            t.join()
            current_label += 1

    return   image_raws,labels,heights,widths,label_dict

def flower_tfrecord_decode():
    channels = 3
    file_path = '/root/ML/data/ini_data/tfrecords/flower'
    image_raws, labels, heights, widths, label_dict = flower_image()
    print('======================')
    print(label_dict)
    for i in range(10):
        #存10个文件
        filename = os.path.join(file_path, 'flower.tfrecord-%.5d-of-%.5d' % (i, 10))
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(len(labels)):
            #每个文件存500个样本
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raws': _bytes_feature(image_raws[j]),
                'labels': _int64_feature(labels[j]),
                'heights': _int64_feature(heights[j]),
                'widths': _int64_feature(widths[j]),
                'channels': _int64_feature(channels)
            }))
            writer.write(example.SerializeToString())
        writer.close()

flower_tfrecord_decode()