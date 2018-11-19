#!/usr/bin/evn python
#coding:utf-8
import os

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys


PATH_XML = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\Annotations\\"



files_dict = {}
files = os.listdir(PATH_XML)
obkect_dict={}
obkect_dict['fire'] = 0
obkect_dict['smoke'] = 0
obkect_dict['cloud'] = 0
obkect_dict['lamp'] = 0
obkect_dict['head'] = 0
for file in files:
    f = PATH_XML+file
    tree = ET.parse(f)     #打开xml文档
    root = tree.getroot()         #获得root节点
    filename = root.find('filename').text
    filename = filename[:-4]
    for object in root.findall('object'): #找到root节点下的所有object节点
        name = object.find('name').text   #子节点下节点name的值
        if name == 'head':
            print (file)
        obkect_dict[name] += 1

print (str(obkect_dict))

