#-*-coding:utf-8-*- 
__author__ = 'liurj'

# 滑动窗口函数

#给定一个图像，返回一个从左向右滑动的窗口（滑动步长可以指定）
def sliding_window(img,stepSize,windowSize):
    for y in xrange(0,img.shape[0],stepSize):
        for x in xrange(0,img.shape[1],stepSize):
            yield (x,y,img[y:y+windowSize[1],x:x+windowSize[0]])


