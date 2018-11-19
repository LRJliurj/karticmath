import os
import random

test = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\ImageSets\\Main\\test.txt"
train = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\ImageSets\\Main\\train.txt"
trainval = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\ImageSets\\Main\\trainval.txt"
val = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\ImageSets\\Main\\val.txt"

def splitFiles(path):
    fileNames = os.listdir(path)
    fileShuff = []
    for file in fileNames:
        fileName = file.split(".")[0]
        fileShuff.append(fileName)

    random.shuffle(fileShuff)
    print (fileShuff)
    sizeFile = int(len(fileShuff) / 4)
    print (sizeFile)
    testF = fileShuff[0:sizeFile*2]
    trainF = fileShuff[sizeFile*2:sizeFile*3]
    trainValF = fileShuff[sizeFile*2:]
    valF = fileShuff[sizeFile*3:]

    with open (test,"w") as f :
        for fileName in testF:
            f.write(fileName+"\n")

    with open(train, "w") as f:
        for fileName in trainF:
            f.write(fileName + "\n")
    with open(trainval, "w") as f:
        for fileName in trainValF:
            f.write(fileName + "\n")
    with open(val, "w") as f:
        for fileName in valF:
            f.write(fileName + "\n")


if __name__=="__main__":
    path = "D:\\opt\\codework\\python3code\\ssd\\SSD-Tensorflow-master\\VOC2007\\Annotations\\"
    splitFiles(path)