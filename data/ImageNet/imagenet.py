from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import DenseNet121,DenseNet169,DenseNet201
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import time
import argparse
import cv2

#针对参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,help="path to the input image")
# ap.add_argument("-model", "--model", type=str, default="vgg16",help="name of pre-trained network to use")
# args = vars(ap.parse_args())

#模型定义
MODELS = {
"vgg16": VGG16,
"vgg19": VGG19,
"inception": InceptionV3,
"xception": Xception, # TensorFlow ONLY
"resnet": ResNet50,
"DenseNet121": DenseNet121,
"DenseNet169": DenseNet169,
"DenseNet201": DenseNet201
}

def network(model_name,imagefile):
    start = time.time()
    if model_name not in MODELS.keys():
        raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")
    
    #不同模型输入图形尺寸定义
    inputShape=(224,224)
    preprocess = imagenet_utils.preprocess_input
    
    if model_name in ("inception", "xception"):
        inputShape = (299,299)
        preprocess = preprocess_input
    
    #加载网络模型,使用imagenet权重,加载权重
    print("[INFO] loading {}...".format(model_name))
    Network = MODELS[model_name]
    model = Network(weights="imagenet")
    
    #输入图像,并对图像预处理
    print("[INFO] loading and pre-processing image...")
    image = load_img(imagefile, target_size=inputShape)
    image = img_to_array(image)
    
    #增加一个维度
    image = np.expand_dims(image, axis=0)
    #预处理函数
    image = preprocess(image)
    
    #输入图像给网络,并输出分类结果
    print("[INFO] classifying image with '{}'...".format(model_name))
    #根据网络返回预测值
    preds = model.predict(image)
    #解析为键值对和概率,返回前五个预测值
    P = imagenet_utils.decode_predictions(preds)
    f=open('/root/ML/data/ImageNet/result%s' % imagefile.split('/')[-1].split('.')[0]+'.txt','a',encoding='utf-8')
    f.write(model_name+'\n')
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        result = "{}. {}: {:.2f}%".format(i + 1, label, prob * 100)
        print(result)
        f.write(result+'\n')
    end = time.time()
    cost_time = '预测使用时间 %d s' % (end - start)
    f.write(cost_time+'\n')
    print(cost_time)
    f.close()
    #从磁盘讲输入图像读取并画图最可能的预测值
    print(imagefile)
    # orig = cv2.imread(imagefile)
    # (imagenetID, label, prob) = P[0][0]
    # cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
    # (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # cv2.imshow("Classification", orig)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()