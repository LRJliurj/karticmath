#-*-coding:utf-8-*- 
__author__ = 'liurj'

# 图像金字塔


import cv2

#通过指定的因子来调整图像的大小
def resize(img,scaleFactor):
    return cv2.resize(img,(int(img.shape[1] * (1/scaleFactor)),int(img.shape[0] * (1/scaleFactor))),interpolation = cv2.INTER_AREA)

#建立图像金字塔，返回被调整过大小的图像直到宽度和高度都达到所规定的最小约束
def pyramid (img,scale=1.5,minSize=(200,80)):
    yield img
    while True :
        img = resize(img,scale)
        if (img.shape[0] < minSize[1] or img.shape[1] < minSize[0]):
            break

    # 注意这里以yield 关键字返回图像，该函数是一个生成器，与return返回有区别
    yield img

