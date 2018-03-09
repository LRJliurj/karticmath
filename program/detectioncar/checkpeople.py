#-*-coding:utf-8-*- 
__author__ = 'liurj'

# 使用opencv 的HOGDescriptor 函数来检测人

import cv2
import numpy as np

def is_inside(o,i):
    ox,oy,ow,oh = o
    ix,iy,iw,ih = i
    return ox>ix and oy > iy and ox + ow < ix + iw and oy +oh < iy +ih

def draw_person(img,person):
    x,y,w,h = person
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)

img = cv2.imread("./data/images/people.jpg")
print (img.shape)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 该检测方法将返回一个与矩形相关的数组，可用该数组在图像上绘制形状
found,w = hog.detectMultiScale(img)

print ("found")
print (found)
print ("w")
print (w)
found_filtered = []
for ri,r in enumerate(found):
    for qi,q in enumerate(found):
        if ri != qi and is_inside(r,q):
            break
        else:
            found_filtered.append(r)
for person in found_filtered:
    draw_person(img,person)

cv2.imshow("people detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()