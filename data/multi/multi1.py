import tensorflow as tf



#定义计算集群
cluster = tf.train.ClusterSpec({
    "worker": [
        "172.17.3.55:1234",    #需要使用定义/job:worker/task:0
        "172.17.3.200:1234"     #需要使用定义/job:worker/task:1
    ],
    "ps": [
        "172.17.3.157:1234"   #需要使用定义/job:ps/task:1
    ]
})


tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=2)

tf.estimator.Estimator()