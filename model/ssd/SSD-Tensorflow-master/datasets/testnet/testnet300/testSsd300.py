import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
slim = tf.contrib.slim
import sys

sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
ckpt_filename = 'D:\\opt\\data\\ssd\\train_1\\model.ckpt-1659'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

numClasses = 5

def  predictImg(img,select_threshold=0.7, nms_threshold=0.1, net_shape=(300, 300)):

    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=numClasses, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# 决策  返回5个最高得分中 有1个 smoke / fire
def check(imgName,img,select_threshold):
    rclasses, rscores, rbboxes = predictImg(img,select_threshold)
    print (rclasses)
    print ("-----------")
    print (rscores)
    print("-----------")
    print(rbboxes)
    nums = len(rclasses)
    # for i in range(0,nums):
    #     tupe4l = (str(imgName),str(rclasses[i]),float(rscores[i]),str(rbboxes[i]))


if __name__=='__main__':
    fileImg = "D:\\opt\\data\\fire\\voc2007\\fire\\fire_1 (352).jpg"
    img = cv2.imdecode(np.fromfile((fileImg), dtype=np.uint8), cv2.IMREAD_COLOR)
    # img = cv2.resize(img, net_shape, interpolation=cv2.INTER_CUBIC)
    check("fire_1 (352)",img,0.70)