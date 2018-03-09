#-*-coding:utf-8-*- 
__author__ = 'liurj'

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]


#构建所有商品的候选项集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #frozenset 是指被“冰冻”的集合，不能修改该集合。 这里将C1集合作为字典键值使用
    return map(frozenset,C1)

# D:数据集 Ck:候选集合的列表 minSupport:感兴趣的项集的最小支持度
# 该函数用于从C1生成L1(筛选出满足最小支持度的候选项集)
def scanD(D,Ck,minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    #保存满足最小支持度要求的集合
    retList = []
    #用于存储频繁项集的支持度字典 key是频繁项集 value是其支持度
    supportData = {}
    for key in ssCnt:
        #计算项集的支持度
        support = ssCnt[key] / numItems
        if support >= minSupport:
            # 在首部插入满足支持度的候选集合
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

# Lk 频繁项集 k:项集元素个数
def aprioriGen (Lk,k):
    retList  = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # 前k-2个项相同时，将两个集合合并
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set,dataSet)
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0 ):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData


if __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    print ("所有候选项集,单个物品项")
    print (C1)
    D=map(set,dataSet)
    print ("交易物品集")
    print (D)
    L1,supportData0 = scanD(D,C1,0.5)
    print ("满足支持度的候选集")
    print (L1)
    print ("满足支持度的候选集的支持度字典")
    print (supportData0)

    # 结论： 1，2,3,5 构成了满足支持度的频繁项集L1 . 物品4没有达到最小支持度， 没在L1中，通过去掉这件物品， 减少了查找两物品项集的工作量



