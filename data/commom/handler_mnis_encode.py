#解析生成的十个数据文件
import tensorflow as tf

sess1 = tf.Session()
#获取文件列表
files = tf.train.match_filenames_once('/root/ML/data/ini_data/tfrecords/mnist/mnist.tfrecord-*')

#创建输入队列,队列列表为获取的文件列表,使用随机读取文件
filename_queue = tf.train.string_input_producer(files,shuffle=True,num_epochs=1)

#解析一个样本
reader = tf.TFRecordReader()

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


#生成batch训练数据

image = tf.decode_raw(features["image_raw"],tf.float32)
label = features["label"]

#batch数量
batch_size=100

#队列容量
capacity = 1000 + 3*1000


#组合样例
example_batch, label_batch = tf.train.batch(
    [image,label],batch_size=batch_size,capacity=capacity
)


with tf.Session() as sess:
    #输出batch
    tf.initialize_all_variables().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('准备开始')
    #获取组合数据,作为神经网络输入
    for i in range(10):
        cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
        print(len(cur_example_batch))
        print(len(label_batch))
    coord.request_stop()
    coord.join(threads)


# with tf.Session() as sess:
#     #生成数据
#     #初始化变量  match_filenames_once
#     tf.local_variables_initializer().run()
#
#     print('获取的文件列表')
#     print(sess.run(files))
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     for i in range(6):
#         image = sess.run(tf.decode_raw(features['image_raw'],tf.float32))
#         print(image.shape)
#         print(sess.run(features["label"]))
#     coord.request_stop()
#     coord.join(threads)