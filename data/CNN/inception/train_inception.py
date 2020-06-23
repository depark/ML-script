#使用谷歌inception_v3进行训练 只训练全连接层
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import logging
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3


#日志地址
LOG_FILE = '/opt/inceptionv3.log'

#处理好的数据
INPUT_DATA = '/root/ML/data/change_data/flower/flower_process_data.npy'

#保存训练好的模型路径
TRAIN_FILE = '/root/ML/MODEL/inception_v3/inceptionv3.ckpt'

#谷歌提供的训练好的模型位置
CKPT_FILE = '/root/ML/data/third_code/inception_v3.ckpt'

#定义训练中的参数
LEARNING_RATE = 0.001
STEPS = 3000
BATCH = 100
N_CLASS = 5

#不需要再谷歌训练好的模型中加载的数据,只需要训练最后一个全连接层,所以全连接层的参数不需要加载,而需要训练
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

#需要训练的网络层参数名
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'


class InceptionV3():
    def __init__(self):
        logging.basicConfig(filename=LOG_FILE,level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        self.config = tf.ConfigProto(
            device_count={"CPU": 30},
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0)

    #获取所有从谷歌模型中需要加载的参数
    def get_tuned_variables(self):
        exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

        variables_to_restore = []

        #列举所有参数,过滤掉最后一层参数
        for var in slim.get_model_variables():
            exclude = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    exclude = True
                    break
            if not exclude:
                variables_to_restore.append(var)
        return variables_to_restore

    def get_tuned_all_variables(self):
        variables_to_restore = []
        for var in slim.get_model_variables():
            variables_to_restore.append(var)
        return variables_to_restore

    # 获取需要训练的变量列表
    def get_trainable_variables(self):
        scopes = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
        variable_to_train = []
        for scope in scopes:
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,scope
            )
            variable_to_train.append(variables)
        return variable_to_train

    def full_main(self):
        #全连接层参数训练
        #加载预处理好的数据
        process_data = np.load(INPUT_DATA,allow_pickle=True)
        training_images = process_data[0]
        n_training_examples = len(training_images)
        training_labels = process_data[1]
        validation_images = process_data[2]
        validation_labels = process_data[3]
        testing_images = process_data[4]
        testing_labels = process_data[5]

        self.logger.info("%d 训练数据 %d 验证数据 %d 测试数据" % (n_training_examples,len(validation_images),len(testing_images)))

        #定义输入 和输出标签
        images = tf.placeholder(tf.float32,[None,299,299,3],name='image_images')

        labels = tf.placeholder(tf.int64,[None],name='labels')

        #定义incepition_v3模型，inception_v3前向传播
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits,_ = inception_v3.inception_v3(images,num_classes=5)
        #获取要训练的变量
        # trainable_variables = get_trainable_variables()

        #交叉熵损失,模型中已经加入正则损失
        tf.losses.softmax_cross_entropy(tf.one_hot(labels,N_CLASS), logits, weights=1.0)

        #定义训练过程,反向传播算法
        train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

        #正确率
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(logits,1),labels)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  #计算所有的平均正确率
        #定义加载模型的函数
        # slim.get_variables_to_restore(include=None,exclude=None)
        load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE,self.get_tuned_variables(),ignore_missing_vars=True)

        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session(config=self.config) as sess:
            #初始化变量
            init = tf.global_variables_initializer()
            sess.run(init)

            self.logger.info('加载模型')
            load_fn(sess)

            start = 0
            end = BATCH
            for i in range(300):
                # self.logger.info('全连接层训练 第 %d 轮训练' % i)
                sess.run(train_step,feed_dict={
                    images: training_images[start:end],
                    labels: training_labels[start:end]
                })

                #每训练30次或者结束输出日志并保存模型
                if i % 10 == 0 or i + 1 == STEPS:
                    validation_accuracy = sess.run(evaluation_step, feed_dict={
                        images: validation_images,labels: validation_labels
                    })
                    test_accuracy = sess.run(evaluation_step, feed_dict={
                        images: testing_images, labels: testing_labels
                    })
                    self.logger.info("最终测试数据上正确率为 %.2f %%" % (test_accuracy * 100))
                    saver.save(sess,TRAIN_FILE,global_step=i)
                    self.logger.info("全连接层训练 经过 %d 轮训练,正确率为 %.2f%%" % (i,validation_accuracy * 100.0))
                start = end
                if start == len(training_images):
                    start = 0

                end = start + BATCH
                if end > len(training_images):
                    end = len(training_images)



    def all_training(self):
        # 全连接层参数训练
        # 加载预处理好的数据
        process_data = np.load(INPUT_DATA, allow_pickle=True)
        training_images = process_data[0]
        n_training_examples = len(training_images)
        training_labels = process_data[1]
        validation_images = process_data[2]
        validation_labels = process_data[3]
        testing_images = process_data[4]
        testing_labels = process_data[5]

        self.logger.info("%d 训练数据 %d 验证数据 %d 测试数据" % (n_training_examples, len(validation_images), len(testing_images)))

        # 定义输入 和输出标签
        images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='image_images')

        labels = tf.placeholder(tf.int64, [None], name='labels')

        # 定义incepition_v3模型，inception_v3前向传播
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, _ = inception_v3.inception_v3(images, num_classes=5)
        # 获取要训练的变量
        # trainable_variables = get_trainable_variables()

        # 交叉熵损失,模型中已经加入正则损失
        tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASS), logits, weights=1.0)

        # 定义训练过程,反向传播算法
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

        # 正确率
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算所有的平均正确率


        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            # 初始化变量
            init = tf.global_variables_initializer()
            sess.run(init)

            start = 0
            end = BATCH
            for i in range(STEPS):
                # self.logger.info('第 %d 轮训练' % i)
                sess.run(train_step, feed_dict={
                    images: training_images[start:end],
                    labels: training_labels[start:end]
                })

                # 每训练30次或者结束输出日志并保存模型
                if i % 100 == 0 or i + 1 == STEPS:
                    validation_accuracy = sess.run(evaluation_step, feed_dict={
                        images: validation_images, labels: validation_labels
                    })
                    saver.save(sess, TRAIN_FILE, global_step=i)
                    self.logger.info("全网络训练>>经过 %d 轮训练,正确率为 %.2f%%" % (i, validation_accuracy * 100.0))
                start = end
                if start == len(training_images):
                    start = 0

                end = start + BATCH
                if end > len(training_images):
                    end = len(training_images)

            # 最后测试数据上验证正确率
            test_accuracy = sess.run(evaluation_step, feed_dict={
                images: testing_images, labels: testing_labels
            })
            self.logger.info("最终测试数据上正确率为 %.2f %%" % (test_accuracy * 100))

if __name__ == "__main__":
    inception = InceptionV3()
    inception.full_main()