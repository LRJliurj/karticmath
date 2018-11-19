import os
import random

train = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\ImageSets\\Main\\train.txt"
jpg = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\JPEGImages\\"
files_p = open(train)
files = files_p.readlines()
out_path = "D:\\opt\\data\\ssd\\ssd_train\\"
for file in files:
    if "fire" in file:
        file = file.strip("\n")
        # print (file)
        open(out_path+file+".jpg", "wb").write(open(jpg+file+".jpg", "rb").read())