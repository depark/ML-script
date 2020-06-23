import tensorflow as tf

#　　其中，输入数据  i1=0.05，i2=0.10;

#　　　　　输出数据 o1=0.01,o2=0.99;


LAYER1_NODE = 2


x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y = tf.placeholder(tf.float32,shape=(2),name='y-output')

weight1 = tf.get_variable("weights1",shape=[2,LAYER1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
biases1 = tf.get_variable("biases1", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))

layer1 = tf.nn.sigmoid(tf.matmul(x,weight1)+biases1)

weight2 = tf.get_variable("weights2",shape=[LAYER1_NODE,2],initializer=tf.truncated_normal_initializer(stddev=0.1))
biases2 = tf.get_variable("biases2", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
output = tf.nn.sigmoid(tf.matmul(layer1,weight2)+biases2)

losses = tf.reduce_mean(tf.square(y-output))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(losses)


with tf.Session() as sess:
    for i in range(10000):
        tf.global_variables_initializer().run()

        _,loss,out = sess.run([train_op,losses,output],feed_dict={x:[[0.05,0.10]],y:[0.01,0.99]})
        print('loss is %s' % loss)
        print('output is %s' % out)