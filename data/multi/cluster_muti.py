import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('/root/ML')
from data.FULL import mnist_interface

def train1():
    c1=tf.constant('hello from server1')

    c2=tf.constant('hello from server2')

    cluser = tf.train.ClusterSpec(
        {"local":["localhost:2222","localhost:2223"]}
    )

    server = tf.train.Server(cluser,job_name="local",task_index=0)

    sess=tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True))

    print(sess.run(c1))
    server.join()

# def train2():
#异步模式分布式训练过程
#神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZITION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

#模型保存路径
MODEL_SAVE_PATH = "log/log_async"

#MNISTS数据路径
DATA_PATH = "mnis/data"

#通过flags指定运行参数
FLAGS = tf.app.flags.FLAGS

#指定运行的服务器类型
tf.app.flags.DEFINE_string('job_name','worker',' "ps" or "worker" ')
#指定集群中参数服务器地址
tf.app.flags.DEFINE_string(
    'ps_hosts','tf-ps0:1111,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server'
    'job. e.g. "tf-ps0:1111,tf-ps1:1111"'
)

#指定计算服务器地址
tf.app.flags.DEFINE_string(
    'worker_hosts','tf-worker0:1111,tf-worker1:1111',
    'Comma-separated list of hostname:port for the parameter server'
    'job. e.g. "tf-worker:1111,tf-worker1:1111"'
)

#指定当前程序的任务ID   tf会根据参数/计算服务器列表中端口号启动服务，编号都是从0开始
tf.app.flags.DEFINE_integer(
    'task_id',0,'Task Id of the worker/replica running the training'
)
#定义tf计算图，返回每一轮迭代需要运行的操作
def build_model(x,y_,is_chief):
    #定义损失函数和前向传播算法
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZITION_RATE)
    y = mnist_interface.interface(x, regularizer)
    global_step = tf.contrib.framework.get_or_create_global_step()

    #损失函数和反向传播过程
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        60000 / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #定义每一轮需要运行的操作
    if is_chief:
        #计算变量滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step
        )
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([variable_averages_op, train_op]):
            train_op = tf.no_op()
    return global_step, loss, train_op

def main(argv=None):
    #解析flags 并配置集群
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec(
        {"ps": ps_hosts,"worker": worker_hosts}
    )
    #定义创建任务
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_id)
    #参数管理器只需要管理td中的变量,不需要执行训练的过程
    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()
    is_chief = (FLAGS.task_id == 0)
    mnis = input_data.read_data_sets(DATA_PATH, one_hot=True)
    #指定每一个运算的设备
    device_setter = tf.train.replica_device_setter(
        worker_device="/jobLworker/task:%d" % FLAGS.task_id,
        cluster=cluster
    )

    with tf.device(device_setter):
        x = tf.placeholder(tf.float32, [None,mnist_interface.INPUT_NODE],name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_interface.OUTPUT_NODE],name='y-input')

        global_step, loss, train_op = build_model(x,y_,is_chief)

        hooks = [tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
        sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)

        #管理训练深度学习模型的通用功能
        with tf.train.MonitoredTrainingSession(
            master=server.target,
            is_chief=is_chief,
            checkpoint_dir=MODEL_SAVE_PATH,
            hooks=hooks,
            save_checkpoint_secs=60,
            config=sess_config
        ) as mon_sess:
            print('session started')
            step = 0
            start_time = time.time()

            #执行迭代过程，MonitoredTrainingSession会自动完成初始化 加载训练模型 输出日志并保存模型，判断是否需要推出
            while not mon_sess.should_stop():
                xs, ys = mnis.train.next_batch(BATCH_SIZE)
                _, loss_value, global_step_value = mon_sess.run(
                    [train_op,loss,global_step], feed_dict={ x:xs, y_:ys}
                )
                #每隔一段时间输出训练信息, 不同的计算服务器都会更新全局训练轮数,
                if step > 0 and step % 100 == 0:
                    duration = time.time() - start_time
                    spec_per_batch = duration / global_step_value
                    format_str = "After %d training steps %d global steps, loss on trainnig batch is %g. %.3f sec/batch"
                    print(format_str % (step,global_step_value,loss_value, spec_per_batch))
                step += 1

if __name__ == "__main__":
    tf.app.run()