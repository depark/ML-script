#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cv2
import random
from time import sleep
import os
cap=cv2.VideoCapture(0)

path='d:\pic\pic'
frame_cnt = 0
num = 0
if not os.path.exists(path):
    os.makedirs(path)
content = ['你','是','猪','头','看','猪','头']

while True:
   ret,fram = cap.read()
   cv2.imshow('haha',fram) 
   if cv2.waitKey(1) == ord('q'):
     cv2.destroyAllWindows()
     cap.release()
     break
   if frame_cnt % 15 == 0:    
       cv2.imencode('.jpg',fram)[1].tofile(r'%s\%s-%s.jpg' % (path,content[num%7],random.randint(1000,2000)))
       num+=1 
   frame_cnt+=1
    


# In[ ]:




