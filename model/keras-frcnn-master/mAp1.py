import demjson
import os
import numpy as np
import cv2
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

labels = ['head','helmet','neg','fire_extinguisher','oxygen_tank','acetylene_bottle']

xmlTestPath ="D:\\opt\\data\\fire\\VOCdevkit\\my20181018\\Voc_Test_20181018\\Annotations\\"
jpgTestPath = "D:\\opt\\data\\fire\\VOCdevkit\\my20181018\\Voc_Test_20181018\\JPEGImages\\"

def sumXMlObjects(xmlTestPath):
    labels_dict = {}
    for label in labels:
        labels_dict[label] = 0
    files = os.listdir(xmlTestPath)
    for file in files:
        f = xmlTestPath + file
        tree = ET.parse(f)  # 打开xml文档
        root = tree.getroot()  # 获得root节点
        filename = root.find('filename').text
        filename = filename[:-4]
        for object in root.findall('object'):  # 找到root节点下的所有object节点
            name = object.find('name').text  # 子节点下节点name的值
            labels_dict[name] += 1
    return labels_dict


def xmlObjects(file,object_dict):
    f = file
    tree = ET.parse(f)  # 打开xml文档
    root = tree.getroot()  # 获得root节点
    filename = root.find('filename').text
    filename = filename[:-4]
    for object in root.findall('object'):  # 找到root节点下的所有object节点
        name = object.find('name').text  # 子节点下节点name的值
        object_dict[name] += 1


def xmlObjectsLocations(f):

    tree = ET.parse(f)  # 打开xml文档
    root = tree.getroot()  # 获得root节点
    filename = root.find('filename').text
    filename = filename[:-4]
    object_list = []
    for object in root.findall('object'):# 找到root节点下的所有object节点
        object_dict = {}
        name = object.find('name').text  # 子节点下节点name的值
        bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        location=[]
        location.append(int(xmin))
        location.append(int(ymin))
        location.append(int(xmax))
        location.append(int(ymax))

        print (location)
        print (name)
        object_dict[name] = location
        object_list.append(object_dict)


    xmlObjectLocation = {}
    for ob_dict in object_list:
        for ob in ob_dict:
            xmlObjectLocation[ob]=[]
    for ob_dict in object_list:
        for ob in ob_dict:
            location = ob_dict[ob]
            xmlObjectLocation[ob].append(location)
    return xmlObjectLocation



def IOU (predictBox,Box,overthroud=0.5):
    cx1 = predictBox[0]
    cy1 = predictBox[1]
    cx2 = predictBox[2]
    cy2 = predictBox[3]

    print (predictBox)
    print (Box)
    gx1 = Box[0]
    gy1 = Box[1]
    gx2 = Box[2]
    gy2 = Box[3]

    carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = float(area) / (carea + garea - area)
    print (iou)
    if iou > overthroud :
        return True
    else :
        return False

#predictDict 预测字典 [{'fire':[165,165,165,165]},{'fire':[165,165,165,165]}]
#xmlDict xml文件的字典 [{'fire':[165,165,165,165]},{'fire':[165,165,165,165]}]
def classPrecesion(predictDict,xmlDict,object_Dict):
    classPrecesion_dict = {}
    print (predictDict)
    print (xmlDict)
    for predict in predictDict:
        preBoxs = predictDict[predict]
        xmlBoxs = []
        if predict in xmlDict.keys():
            xmlBoxs = xmlDict[predict]
        xmlObjects = len(xmlBoxs)
        if xmlObjects != 0:
            for preBox in preBoxs :
                for xmlBox in xmlBoxs:
                    bFalg = IOU(list(preBox),list(xmlBox))
                    if bFalg:
                        object_Dict[predict] +=1 #TP

            TP = object_Dict[predict]
            # classPrecesion = TP/float(xmlObjects)
            classPrecesion = TP
            classPrecesion_dict[predict] = classPrecesion
    return classPrecesion_dict


def averagePrecision(xmlTestPath,wfile):
    listPre=[]
    with open(wfile, 'rb') as f:
        line = f.readline()
        line = demjson.decode(line)
        listPre = list(line)

    classPrecesions_dict = {}
    for infoP in listPre:
        fileName = str(infoP['fileName']).split("jpg")[0]+"jpg"
        img = cv2.imdecode(np.fromfile(jpgTestPath+fileName, dtype=np.uint8), cv2.IMREAD_COLOR)
        w = img.shape[0]
        h = img.shape[1]
        predictInfo = list(demjson.decode(str(infoP['predictInfo'])))
        rclasses = []
        rscores = []
        rbboxes = []
        for i in range(len(predictInfo)):
            info = dict(predictInfo[i])
            rclasses.append(str(info['rclasses']))
            rscores.append(float(info['rscores']))
            locat= str(info['rbboxes']).strip().strip('()')
            locat_list = locat.split(",")
            locat_list1 = [int(locat_list[0].strip()),int(locat_list[1].strip()),int(locat_list[2]),int(locat_list[3].strip())]
            rbboxes.append(list(locat_list1))

        predictDict = {}
        for i in range(len(rclasses)):
            predictDict[rclasses[i]] = rbboxes

        xmlFileName = xmlTestPath+str(fileName.split(".")[0])+".xml"
        xmlFileDict = xmlObjectsLocations(xmlFileName)

        classP_dict = {}
        for label in labels:
            classP_dict[label] = 0
        classPrecesion_dict = classPrecesion(predictDict,xmlFileDict,classP_dict)
        classPrecesions_dict[fileName] = classPrecesion_dict

    print ("cp"+str(classPrecesions_dict))
    fileC_dict = {}
    for label in labels:
        fileC_dict[label] = 0
    fileC_dict = getFileDict(xmlTestPath,fileC_dict)
    classAveragePrecesion_dict = {}
    for key in fileC_dict:
        fileC = fileC_dict[key]
        cp = 0.
        for keyCp in classPrecesions_dict:
            keyCpP = dict(classPrecesions_dict[keyCp])
            if key in keyCpP:
                cp+=keyCpP[key]
        classAveragePrecesion = cp / fileC
        classAveragePrecesion_dict[key] = classAveragePrecesion
    return classAveragePrecesion_dict

def getFileDict(xmlPath,fileC_dict):
    for file in os.listdir(xmlPath):
        for key in fileC_dict:
            obj_list = []
            tree = ET.parse(xmlPath+file)  # 打开xml文档
            root = tree.getroot()  # 获得root节点
            filename = root.find('filename').text
            filename = filename[:-4]
            for object in root.findall('object'):  # 找到root节点下的所有object节点
                name = object.find('name').text  # 子节点下节点name的值
                if key == name :
                    fileC_dict[key]+=1
                    # break
    return fileC_dict






def meanAveragePrecesion(classAveragePrecesion_dict):
    print ("cAp"+str(classAveragePrecesion_dict))
    classNums = 0
    classAP = 0.
    for key in classAveragePrecesion_dict:
        classNums+=1
        classAP += classAveragePrecesion_dict[key]
    mAP = (classAP / classNums)
    print ("mAP"+str(mAP))
    return mAP





if __name__=='__main__':
    classAveragePrecesion_dict = averagePrecision(xmlTestPath,"./data/wPreFile.txt")

    meanAveragePrecesion(classAveragePrecesion_dict)
    



