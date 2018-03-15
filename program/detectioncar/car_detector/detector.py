#-*-coding:utf-8-*-
__author__ = 'liurj'

import cv2
import numpy as np

datapath = './TrainImages/'

SAMPLES = 450

def path(cls,i):
    return "%s/%s%d.pgm" % (datapath,cls,i+1)

def get_flann_matcher():
    flann_params = dict(algorithm = 1, trees = 5)
    return cv2.FlannBasedMatcher(flann_params,{})

def get_bow_extractor(extract,flann):
    return cv2.BOWImgDescriptorExtractor(extract,flann)

def get_extract_detect():
    return cv2.xfeatures2d.SIFT_create(),cv2.xfeatures2d.SIFT_create()

#返回一组数组 图像特征
def extract_sift(fn,extractor,detector):
    im = cv2.imread(fn,0)
    return extractor.compute(im,detector.detect(im))[1]

#返回BOW特征
def bow_features(img,extractor_bow,detector):
    return extractor_bow.compute(img,detector.detect(img))

def car_detector():
    pos,neg = "pos-","neg-"
    detect,extract = get_extract_detect()
    flann_matcher = get_flann_matcher()
    print ("构建词袋模型训练器....")
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(2000)
    extract_bow = cv2.BOWImgDescriptorExtractor(extract,flann_matcher)

    print ("添加特征到训练器....")
    for i in range(SAMPLES):
        bow_kmeans_trainer.add(extract_sift(path(pos,i),extract,detect))
        bow_kmeans_trainer.add(extract_sift(path(neg,i),extract,detect))

    #执行聚类  返回图像特征词汇 voc.shape=(1000,128)
    voc = bow_kmeans_trainer.cluster()
    np.save('./voc/voc.npy',voc)
    extract_bow.setVocabulary(voc)

    traindata,trainlabels = [],[]
    print ("添加训练样本数据....")
    for i in range(SAMPLES):
        traindata.extend(bow_features(cv2.imread(path(pos,i),0),extract_bow,detect))
        trainlabels.append(1)
        traindata.extend(bow_features(cv2.imread(path(neg,i),0),extract_bow,detect))
        trainlabels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)

    # C 此参数可以决定分类器的训练误差和预测误差 。 其值越大，误判的可能性越小，但训练精度会降低。 另外，值太低可能会导致过拟合，从而使预测精度降低
    svm.setC(30)
    # kernel : 该参数确定分类器的性质 ，SVM_LINEAR 说明分类器为线性超平面，这在实际应用中非常适用于二分类，而SVM_RBF使用高斯函数来对数据进行分类，这意味着
    # 数据被分到由这些函数定义的核中。 当训练SVM来分类超过两个的类时 ，必须使用RBF
    svm.setKernel(cv2.ml.SVM_RBF)

    svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))
    svm.save('./model/svm_carxml')
    # svm.load('./model/svm_carxml')
    return svm,extract_bow



def load_model_svm_extracter():
    svm = cv2.ml.SVM_create()
    svm = svm.load('./model/svm_carxml')
    detect,extract = get_extract_detect()
    flann_matcher = get_flann_matcher()
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann_matcher)
    voc = np.load('./voc/voc.npy')
    extract_bow.setVocabulary(voc)
    return svm,extract_bow

if __name__=='__main__':
    car_detector()