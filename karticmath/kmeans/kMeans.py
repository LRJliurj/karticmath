#-*-coding:utf-8-*- 
__author__ = 'liurj'

from numpy import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

#计算两个向量的欧式距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#该函数为给定数据集构建一个包含K个随机质心的集合。 随机质心必须要在整个数据集的边界之内
#这可以通过找到数据集每一维的最小和最大值来完成。然后生成0到1.0之间的随机数并通过取值范围和最小值
#，以确保随机点在数据的边界之内。
def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        #random.rand(k,1) 生成k个 0-1之间的浮点数 组成数组
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# K-均值算法， 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
# 这个过程重复数次， 直到数据点的簇分配结果不再改变为止

#kMeans()函数一开始确定数据集中的数据点的总数，然后创建一个矩阵来存储每个点的簇分配结果。
#簇分配结果矩阵clusterAssment包含两列：1列记录簇所引值，第2列存储误差（当前点到簇质心的距离）。
# 后面会使用误差来评价聚类的效果

#按照上述方式（计算质心-分配-重新计算）反复迭代，直到所有数据点的簇分配结果不再改变为止。


#参数：数据集，簇，距离函数引用，初始质心函数引用

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    # 簇分配结果是否改变的标志
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                # 寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            # 如果任意一点簇分配结果发生改变，则更新clusterChanged的标志
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print ("质心发生改变")
        print  (centroids)
        # 更新质心的位置
        # 遍历所有质心并更新它们的取值
        for cent in range(k):#recalculate centroids
            # 通过数组过滤来获得给定 簇的所有点；然后计算所有点的均值，选项axis = 0 ,表示沿矩阵的列方向进行均值计算 .A 把矩阵转numpy数组
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment


# kMeans 函数有可能质心会收敛到局部最小值
#biKeans 函数质心的位置是全局最小值

def  biKmeans(dataSet,k,distMeas = distEclud):
    m = shape(dataSet)[0]
    #用于存储数据集中每个点的簇分配结果及平方误差
    clusterAssment = mat(zeros((m,2)))

    #创建一个初始簇，计算整个数据集的质心
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    print ("1:")
    print (mean(dataSet,axis=0))
    #使用列表保存所有的质心
    centList = [centroid0]

    #遍历数据集，计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:]) ** 2

    print ("2:")
    print (clusterAssment[:,0])

    #循环对簇进行切分,直到得到需要的k个簇数目为止
    while (len(centList) < k ):
        lowestSSE = inf
        for i in range(len(centList)):
            # 尝试划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            print ("3-ptsInCurrCluster")
            print (ptsInCurrCluster)
            # 质心矩阵，质心索引及点到质心索引的误差值矩阵
            centroidMat , splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)
            #计算被拆分的簇中的误差
            sseSplit = sum(splitClustAss[:,1])
            #计算未必拆分簇的误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            print ("sseSplit , and notSplit:",sseSplit,sseNotSplit)
            #比较总误差是否在减小
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                #新簇质心矩阵
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        #更新簇的分配结果 , 当使用kMeans() 函数并制定簇数2时，会得到编号为0和1的结果簇。
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit


        print ("the bestCentToSplit is :",bestCentToSplit)
        print ("the len of bestClustAss is ",len(bestClustAss))
        print ("5-bestNewCents:")
        print (bestNewCents)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment


import urllib
import json


def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print yahooApi
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep


def massPlaceFind(fileName):
    fw = open('./data/places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print "error fetching"
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib
import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('./data/places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('./data/Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__=='__main__':
    dataMat = mat(loadDataSet("./data/testSet.txt"))
    print (min(dataMat[:,0]))
    #生成min 到 max之间的值
    print (random.rand(2,1))
    centroids = randCent(dataMat,2)
    print (centroids)
    print  (distEclud(dataMat[0],dataMat[1]))
    myCentroids,clustAssing = kMeans(dataMat,4)
    print ("myCentroids")
    print (myCentroids)
    print ("clustAssing")
    print (clustAssing)

    dataMat3 = mat(loadDataSet("./data/testSet2.txt"))
    centList,myNewAssments = biKmeans(dataMat3,3)
    print ("二分kmeans聚类， 质心：")
    print (centList)

    #地理位置做聚类
    clusterClubs(5)