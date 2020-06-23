import time
from progressbar import *

def task(cost=0.5,epoch=10, name='',_sub_task=None):
    def _sub():
        bar = ProgressBar(maxval=epoch)
        bar.start()
        for _ in range(epoch):
            time.sleep(0.5)
            if _sub_task is not None:
                _sub_task()
            bar.update()
    return _sub


task(name='Task1',_sub_task=task(name='Task2',_sub_task=task(name="Task3")))()
