#使用estimator实现全连接

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data



def full_conn():
    #日志输出到屏幕
    mnis = input_data.read_data_sets('/root/ML/data/ini_data/mnis_data',one_hot=False)

    #指定神经王的输入层,所有指定的输入都会拼接在一起作为整个神经网络的输入
    feature_columns = [tf.feature_column.numeric_column("image",shape=[784])]

    #定义多层全连接网络
    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,        # 定义输入层数据
        hidden_units=[500],                     # 指定每一层隐藏层节点数
        n_classes=10,                           # 指定总共菲律数量
        optimizer=tf.train.AdamOptimizer(),  # 指定优化函数
        model_dir="/tmp/log/estim_full"         #将训练过程指标保存的目录,可以用来可视化监控指标
    )

    #定义数据输入 ,x 为训练输入的数据(对应feature_columns中一个images)  y为每一个x对应的答案转化为正整数
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": mnis.train.images},
        y=mnis.train.labels.astype(np.int32),
        num_epochs=None,    #指定循环的轮数
        batch_size=128,
        shuffle=True        # 进行打算
    )

    #训练模型，DNNClassifier定义的模型会使用交叉熵作为损失函数
    estimator.train(input_fn=train_input_fn, steps=10000)

    #定义测试数据输入
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": mnis.test.images},
        y=mnis.test.labels.astype(np.int32),
        num_epochs=1,    #指定循环的轮数
        batch_size=128,
        shuffle=False        # 进行打算
    )


    #通过evaluate评测训练好的模型效果
    accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
    print("测试数据正确率: %g %%" % (accuracy_score*100))

#自定义使用卷积网络
class conv_conn:
    def __init__(self):
        self.model_params = {"learning_rate": 0.01}
        self.mnis = input_data.read_data_sets('/root/ML/data/ini_data/mnis_data', one_hot=False)

    #定义LETnet5网络结构
    def lenet(self,x,is_training):
        #转化输入
        x = tf.reshape(x,shape=[-1,28,28,1])

        #构建网络
        #卷积层
        net = tf.layers.conv2d(x,32,5,activation=tf.nn.relu)
        #池化层
        net = tf.layers.max_pooling2d(net,2,2)
        net = tf.layers.conv2d(net,64,3,activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(net,2,2)
        #转化为全连接层输入
        net = tf.contrib.layers.flatten(net)
        #1024个隐藏节点全连接层
        net = tf.layers.dense(net,1024)
        net = tf.layers.dropout(net,rate=0.4,training=is_training)
        return tf.layers.dense(net,10)

    #定义训练过程 损失函数 评测
    def model_fn(self,features,labels,mode,params):
        #定义前向传播 判断是否训练过程
        predict = self.lenet(features['image'],mode == tf.estimator.ModeKeys.TRAIN)

        #如果预测过程,直接将结果返回
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,predictions={"result": tf.argmax(predict,1)})

        #定义损失函数
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=predict,labels=labels
            )
        )

        #定义优化函数
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params['learning_rate']
        )

        #定义训练过程
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

        #定义测评标准, 运行evaluate会计算这里定义的所有测评标准
        eval_metric_ops = {
            "metric": tf.metrics.accuracy(
                tf.argmax(predict,1),labels
            )
        }

        #返回模型训练过程，定义的模型,损失函数，训练过程 测评方法
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

    def run(self):
        estimator = tf.estimator.Estimator(model_fn=self.model_fn,params=self.model_params,model_dir='/tmp/estimator-conn/')

        #定义输入
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"image": self.mnis.train.images},
            y=self.mnis.train.labels.astype(np.int32),
            num_epochs=None,
            batch_size=128,
            shuffle=True
        )
        estimator.train(input_fn=train_input_fn,steps=30000)

        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"image": self.mnis.test.images},
            y=self.mnis.test.labels.astype(np.int32),
            num_epochs=1,
            batch_size=128,
            shuffle=False
        )

        test_result = estimator.evaluate(input_fn=test_input_fn)

        #metric 就是model定义的指标
        accuracy_score = test_result["metric"]
        print('测试正确率: %g %%' % (accuracy_score*100))

        #使用训练好的模型在新数据上预测结果
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"image": self.mnis.test.images[:10]},
            num_epochs=1,
            shuffle=False
        )
        predictions = estimator.predict(input_fn=predict_input_fn)

        print('=========预测数据============')
        for i,p in enumerate(predictions):
            print('预测结果 %s: %s' % (i+1,p["result"]))


#使用dataset 测试iris的数据流处理 (每条数据四个特征,三个分类)
class dataset_estima():
    # 定义输入,是数据集每次被调用获得一个batch的数据(包括输入层数据和正确答案)
    def my_input_fn(self,file_path,perform_shuffle=False,repeat_count=1):
        #由于数据是csv格式,需要定义解析csv一行方法
        def decode_csv(line):
            #前4个为特征 最后一个为答案
            parsed_line = tf.decode_csv(line,[[0.],[0.],[0.],[0.],[0.]])
            #返回的格式跟feature_columns 格式匹配
            return {"x":parsed_line[:-1]}, parsed_line[-1:]

        # 使用数据集处理输入数据
        dataset = (tf.contrib.data.TextLineDataset(file_path).skip(1).map(decode_csv))

        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.batch(32)
        iterator = dataset.make_one_shot_iterator()
        #返回一个batch的输入数据
        batch_features,batch_labels = iterator.get_next()
        return batch_features,batch_labels


    def run(self,iris_train_file,iris_test_file):
        #定义神经网络2个隐藏层，定义输入数据
        feature_columns = [tf.feature_column.numeric_column("x",shape=[4])]
        classifiter = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[10,10],
            n_classes=3
        )

        #使用lambda将训练的相关信息传入自定义的数据处理函数,生成所需要的数据格式
        classifiter.train(input_fn=lambda: self.my_input_fn(iris_train_file,True,10))

        test_result = classifiter.evaluate(
            input_fn=lambda: self.my_input_fn(iris_test_file,False,1)
        )["accuracy"]
        print("测试正确率 %g %%" % (test_result*100))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    conv_conn().run()