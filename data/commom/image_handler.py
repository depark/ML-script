#图像预处理
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# img_file = 'D:\share\learn\ML\\test-photo\\meinv.jpg'

class Image_Resize():
    #图像预处理: 图像大小统一
    def __init__(self,image_file):
        #预处理图片,解码图片
        self.sess = tf.Session()
        #读取图片
        image_raw = tf.gfile.FastGFile(image_file,'rb').read()
        #编码成为三维矩阵
        image_data_pr = tf.image.decode_jpeg(image_raw)
        #转化为一维
        image_data_pr = tf.image.rgb_to_grayscale(image_data_pr)
        #数据类型转换为实数
        image_data = tf.image.convert_image_dtype(image_data_pr,dtype=tf.float32)
        self.image_data = self.sess.run(image_data)

    def resize_image(self,method):
        with tf.Session() as sess:
        #调整图片矩阵大小,第一个为原始图像矩阵,第二为调整大小,第三为调整算法
            resized_image = sess.run(tf.image.resize_images(self.image_data,[300,300],method=method))
            plt.imshow(resized_image)
            plt.show()
            #转化为整数类型保存
            img_byte = tf.image.convert_image_dtype(resized_image, dtype=tf.uint8)
            image_raw = sess.run(tf.image.encode_jpeg(img_byte))
            with tf.gfile.GFile('D:\share\learn\ML\\test-photo\\%s.jpg' % method,'wb') as f:
                f.write(image_raw)
            return resized_image

#图像预处理
def distort_color(image,color_ording=0):
    #随机调整图像 对比度，亮度，饱和度，色相
    if color_ording == 0:
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)

    elif color_ording == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)

    elif color_ording == 2:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
    #图像预处理,输入原始图像,输出神经网络输入,  随机获取标注框的部分图片信息,输出神经网络张量矩阵
    if bbox is None:
        #提供标注框 如果没有整个图片就是标注框
        # bbox = tf.constant([0.05, 0.18, 0.9 , 0.58],dtype=tf.float32,shape=(1,1,4))
        bbox = tf.constant([0,0,1,1],dtype=tf.float32,shape=(1,1,4))

    #转换图像张量类型为float32
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    #随机截取图像,避免需要关注物体大小影响图像识别
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox,min_object_covered=0.7)
    distored_image = tf.slice(image,bbox_begin,bbox_size)

    #将随机截取的图像调整为神经网络输入层的大小,调整算法随机
    distored_image = tf.image.resize_images(
        distored_image, [height,width],method=np.random.randint(4)
    )

    #随机左右翻转图像
    distored_image = tf.image.random_flip_left_right(distored_image)

    distored_image = tf.image.random_flip_up_down(distored_image)

    #随机调整色彩
    distored_image = distort_color(distored_image,color_ording=np.random.randint(3))

    return distored_image



