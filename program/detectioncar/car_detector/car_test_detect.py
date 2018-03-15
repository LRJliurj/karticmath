#-*-coding:utf-8-*- 
__author__ = 'liurj'
import cv2
import numpy as np

from detector import car_detector,bow_features,load_model_svm_extracter
from pyramid import pyramid
from non_maximum import non_max_suppression_fast as nms
from sliding_window import sliding_window

def in_range(number,test,thresh=0.2):
    return abs(number-test) < thresh

img_tst='./images/car.jpg'

#训练svm检测模型 只做一次
# svm,extractor = car_detector()

svm,extractor = load_model_svm_extracter()
detect = cv2.xfeatures2d.SIFT_create()
w,h = 100,40
img = cv2.imread(img_tst)

rectangles = []
counter = 1
scaleFactor = 1.25
#图像的缩减尺度，图像金字塔尺度变换的比例
scale = 1
font = cv2.FONT_HERSHEY_PLAIN

#图像金字塔 ，按比例缩减图像
for resized in pyramid(img,scaleFactor):
    scale = float(img.shape[1]) / float(resized.shape[1])
    #滑动窗口 在图像金字塔每一层做滑动检测
    for (x,y,roi) in sliding_window(resized,20,(w,h)):
        if roi.shape[1] != w or roi.shape[0] != h:
            continue

        try:
            bf = bow_features(roi,extractor,detect)
            #result存储分类
            _,result = svm.predict(bf)
            #使用flag参数，res 表示返回预测的得分
            a,res = svm.predict(bf,flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            print ("类：%d, 得分：%f" % (result[0][0],res[0][0]))
            score = res[0][0]
            if result[0][0] == 1:
                if score < -0.4 : #得分越小，代表检测到的窗口越准确，置信度越高  可以在测试集中调整这个值，选取最优的阈值
                    rx,ry,rx2,ry2 = int(x*scale),int(y*scale),int((x+w)*scale),int((y+h)*scale)
                    rectangles.append([rx,ry,rx2,ry2,abs(score)])
        except:
            pass

        counter += 1

windows = np.array(rectangles)
boxes = nms(windows,0.25)
print ("boxes")
print (boxes)
for (x,y,x2,y2,score) in boxes:
    print ("滑动窗口 及 评分：")
    print (x,y,x2,y2,score)
    cv2.rectangle(img,(int(x),int(y)),(int(x2),int(y2)),(0,255,0),1)
    cv2.putText(img,"%f" % score,(int(x),int(y)),font,1,(0,255,0))

cv2.imshow("img",img)
cv2.waitKey(0)




