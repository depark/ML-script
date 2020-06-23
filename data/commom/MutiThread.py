import tensorflow as tf
import numpy as np
import time
import threading

def MyLoop(coord,workid):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("来自线程%d 的停止线程信号" % workid)
            coord.request_stop()
        else:
            print('当前运行线程: %d' % workid)
            time.sleep(1)

coord = tf.train.Coordinator()
threads = [
    threading.Thread(target=MyLoop,args=(coord, i)) for i in range(5)
]
for t in threads:t.start()
coord.join()

tf.train.QueueRunner