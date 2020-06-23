#inception-v3 一个inception过程,预处理图片数据过程
#原始图片转化为299X299 尺寸, 数据分类为训练数据,测试数据,验证数据
import glob
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.platform import gfile

#原始图片路径
INPUT_DATA = "/root/ML/data/ini_data/flower_photos"

#输出目录
OUTPUT_FILE = "/root/ML/data/change_data/flower/flower_process_data.npy"

#测试数据和验证数据比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAG = 10

'''
'''

#处理原始数据过程,图片转化为299X299尺寸  分出测试数据和验证数据
def create_image_lists(sess,testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    #初始化个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0    #标签类

    #循环5个数据集,转换图片,第一个为根目录
    for sub_dir in sub_dirs[1:]:
        print('开始转换' + sub_dir)
        #使用glob获取一个种类目录所有图片
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        leng = len(file_list)
        #处理图片,转化为299x299
        for image_name in file_list:
            #读取原始数据
            image_raw_data = gfile.FastGFile(image_name,'rb').read()
            #jpeg 转化为矩阵
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image,dtype=tf.float32)
            image = tf.image.resize_images(image,[299,299])
            image_value = sess.run(image)

            #随机数据分类
            change = np.random.randint(100)
            if change < validation_percentage:
                validation_images.append(image_value)            # 划入图片矩阵
                validation_labels.append(current_label)          # 对应的种类标签
            elif change < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    #打乱训练数据
    stated = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(stated)
    np.random.shuffle(training_labels)

    return np.asanyarray([training_images,training_labels,
                          validation_images,validation_labels,
                          testing_images,testing_labels
                          ])

#主函数
def main():
    config = tf.ConfigProto(device_count={"CPU": 30},
                            inter_op_parallelism_threads=4,
                            intra_op_parallelism_threads = 8,
                            log_device_placement=False
                            )
    with tf.Session(config=config) as sess:
        process_data = create_image_lists(sess,TEST_PERCENTAG,VALIDATION_PERCENTAGE)
        #通过numpy保存数据
        np.save(OUTPUT_FILE,process_data)

if __name__ == '__main__':
    main()