#-*-coding:utf-8-*- 
__author__ = 'liurj'

#训练数据集 Stanford Univercity : http://ai.stanford.edu/~jkrause/cars/car_dataset.html

import cv2
import numpy as np
from os.path import join

datapath = ''
#路径函数 pos-0.pgm neg-0.pgm
def path(cls,i):
    return "%s/%s%d.pgm" % (datapath,cls,i+1)
pos,neg = "pos-","neg-"
#提取关键点的SIFT实例
detect = cv2.xfeatures2d.SIFT_create()
#提取特征的SIFT实例
extract = cv2.xfeatures2d.SIFT_create()
#基于flann匹配器的实例
#algorithm=1 表示要使用FLANN_INDEX_KDTREE 算法
flann_params = dict(algorithm=1,trees =5)
flann = cv2.FlannBasedMatcher(flann_params,{})

#创建BOW 词袋模型的训练器 并指定簇数40
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
#初始化BOW提取器（extractor） extract（图片特征）视觉词汇作为BOW类的输入，在测试图像中会检测这些视觉词汇
extract_bow = cv2.BOWImgDescriptorExtractor(extract,flann)

# 为使用SIFT特征，需从图像中获取描述符
def extract_sift(fn):
    #以灰度读取图像
    im = cv2.imread(fn,0)
    return extract.compute(im,detect.detect(im))[1]

#每个类都差数据集中读取8个图像 （8个pos正样本，8个neg负样本）
for i in range(8):
    bow_kmeans_trainer.add(extract_sift(path(pos,i)))
    bow_kmeans_trainer.add(extract_sift(path(neg,i)))
#调用cluster() 函数，执行kmeans（）分类并返回词汇
voc = bow_kmeans_trainer.cluster()
# 添加词汇到bow 提取器词汇表中
extract_bow.setVocabulary(voc)

#获取基于BOW的描述符提取器计算得到的描述符
def bow_features(fn):
    im = cv2.imread(fn,0)
    return extract_bow.compute(im,detect.detect(im))

#添加训练集
traindata,trainlabels = [],[]
for i in range(20):
    traindata.extend(bow_features(path(pos,i)))
    trainlabels.append(1)
    traindata.extend(bow_features(path(neg,i)))
    trainlabels.append(-1)

#创建opencv 自带的SVM分类器
svm = cv2.ml.SVM_create()
svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))

#预测函数 fn原始图像
def predict(fn):
    f = bow_features(fn)
    p = svm.predict(f)
    print (fn,"\t",p[1][0][0])
    return p

car,notcar = "./data/images/car.jpg","./data/images/notcar.jpg"
car_img = cv2.imread(car)
notcar_img = cv2.imread(notcar)
car_predict = predict(car)
not_car_predict = predict(notcar)
font = cv2.FONT_HERSHEY_SIMPLEX

if (car_predict[1][0][0] == 1.0):
    cv2.putText(car_img,'Car Detected',(10,30),font,1,(0,255,0),cv2.LINE_AA)

if(not_car_predict[1][0][0] == -1.0):
    cv2.putText(notcar_img,'Car Not Detected',(10,30),font,1,(0,0,255),2,cv2.LINE_AA)

cv2.imshow('BOW + SVM Success',car_img)
cv2.imshow('BOW + SVM Failure',notcar_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
