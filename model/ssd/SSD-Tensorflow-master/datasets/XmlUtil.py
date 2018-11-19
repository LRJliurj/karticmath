#!/usr/bin/evn python
#coding:utf-8
import os

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys

PATH = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\ImageSets\\Main\\"

PATH_XML = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\Annotations\\"

path_file_train=PATH+"train.txt"
path_file_test=PATH+"test.txt"
path_file_trainval=PATH+"trainval.txt"
path_file_val=PATH+"val.txt"

files = []
files.append(path_file_train)
files.append(path_file_test)
files.append(path_file_trainval)
files.append(path_file_val)

files_dict = {}
for file in files:
    file_srx = open(file)  #其中包含所有待计算的文件名
    lines = file_srx.readlines()
    obkect_dict = {}
    obkect_dict['fire'] = 0
    obkect_dict['smoke'] = 0
    obkect_dict['cloud'] = 0
    obkect_dict['lamp'] = 0
    obkect_dict['head'] = 0
    for line in lines:
        f = line[:-1]    # 除去末尾的换行符
        f = PATH_XML+f+".xml"
        tree = ET.parse(f)     #打开xml文档
        root = tree.getroot()         #获得root节点
        filename = root.find('filename').text
        filename = filename[:-4]
        for object in root.findall('object'): #找到root节点下的所有object节点
            name = object.find('name').text   #子节点下节点name的值
            obkect_dict[name] += 1
    file_name = file.split("\\")[-1]
    files_dict[file_name] = str(obkect_dict)
print (files_dict)

