#使用高级框架Dataset解析文本数据
import tensorflow as tf
import sys
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

sys.path.append('/root/ML')
from data.commom.image_handler import preprocess_for_train


files = tf.train.match_filenames_once('/root/ML/data/ini_data/tfrecords/flower/flower.tfrecord-*')

def parses(record):
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([],tf.string),
            'label': tf.FixedLenFeature([],tf.int64),
            'height': tf.FixedLenFeature([],tf.int64),
            'width': tf.FixedLenFeature([],tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
        }
    )
    #图片还原
    decode_image = tf.decode_raw(features['image'],tf.uint8)
    decode_image.set_shape([features['height'],features['width'],features['channels']])
    label = features['label']
    return decode_image,label

image_size = 299  #定义神经网络输入图片尺寸
batch_size = 100  # 训练的batch
shuffle_buffer = 10000   #随机打乱数据buffer

#读取数据
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(parses)    #数据遍历处理parses


#数据预处理
dataset = dataset.map(
    lambda image, label: preprocess_for_train(image,image_size,image,None)
)
#数据组合
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

#重复数据集
NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)


#定义数据迭代器
iterator = dataset.make_initializable_iterator()
image_batch,label_batch = iterator.get_next()

#定义神经网络
learning_rate = 0.01
with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits, _ = inception_v3.inception_v3(image_batch, num_classes=5)

tf.losses.softmax_cross_entropy(tf.one_hot(label_batch,5), logits, weights=1.0)
loss = tf.losses.get_total_loss()
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    sess.run(iterator.initializer)

    while True:
        try:
            _,loss = sess.run(train_step,loss)
            print('损失度为 %2.f' % loss)
        except tf.errors.OutOfRangeError:
            break

