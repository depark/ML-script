#使用keras实现inception模型
import keras
from keras.layers import Conv2D,MaxPooling2D,Input
from keras.applications import ResNet50

#定义输入尺寸
input_img = Input(shape=(256,256,3))

#定义第一个分支
tower_1 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
tower_1 = Conv2D(64,(3,3),padding='same',activation='relu')(tower_1)

#定义第二个分支,第二个分支使用的是输入
tower_2 = Conv2D(64,(1,1),padding='same',activation='relu')(input_img)
tower_2 = Conv2D(64,(3,3),padding='same',activation='relu')(tower_2)


#定义第三个分支,第三个分支使用的是输入
tower_3 = MaxPooling2D((3,3),strides=(1,1),padding='same')(input_img)
tower_3 = Conv2D(64,(1,1),padding='same',activation='relu')(tower_3)

#将三个分支通过concatenate 拼接在一起
output = keras.layers.concatenate([tower_1,tower_2,tower_3],axis=1)