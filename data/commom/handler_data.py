#预处理数据过程
import tensorflow as tf
from data.commom.image_handler import preprocess_for_train

#导入数据文件
files = tf.train.match_filenames_once('/root/ML/data/ini_data/tfrecords/flower/flower.tfrecord-*')

filename_queue = tf.train.string_input_producer(files,shuffle=False)

reader = tf.TFRecordReader()
#读取文件流，定义存储样式
_,serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([],tf.string),
        'label': tf.FixedLenFeature([],tf.int64),
        'height': tf.FixedLenFeature([],tf.int64),
        'width': tf.FixedLenFeature([],tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    }
)

image,label = features['image'],features['label']
height,width = features['height'],features['width']
channels = features['channels']

#原始数据解析像素矩阵 根据尺寸还原图像
decode_image = tf.decode_raw(image, tf.uint8)
decode_image.set_shape([height,width,channels])

#定义神经网络输入层图片大小
image_size = 299

#图像预处理
distored_image = preprocess_for_train(
    decode_image,image_size,image_size,None
)

#整合成神经网络的batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

image_batch,label_batch = tf.train.shuffle_batch(
    [distored_image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue
)

#定义神经网络结构,imagebatch作为输入   label_batch作为输出
learning_rate = 0.01
