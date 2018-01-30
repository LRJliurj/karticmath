#-*-coding:utf-8-*- 
__author__ = 'liurj'
from numpy import *
from Tkinter import *
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        # 将每行映射成浮点数
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

# 数据拆分 参数： dataSet 待切分的数据集， feature 待切分的特征，value 待切分的特征值
def binSplitDataSet(dataSet,feature,value):
    #书中bug
    # mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    # mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1


#回归树的切分函数


#负责生成叶子节点， 当chooseBestSplit() 函数确定不再对数据进行切分时，将调用该regLeaf()
# 函数来得到叶节点的模型。 在回归树中， 该模型其实就是目标变量的均值
#生成叶子节点
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

#在给定数据上计算目标变量的平方误差。（也可以先计算出均值，然后计算每个差值再平方）
# 但这里直接调用均方差函数var() 更加方便。 因为这里需要返回的是总方差， 所以要用均方差
#乘以数据集中样本的个数
# 误差估计函数
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

# 该函数首先尝试将数据集分成两个部分， 切分由函数chooseBestSplit() 完成（未给出函数实现）
# 如果满足停止条件，chooseBestSplit() 将返回None和某类模型的值， 如果构建的是回归树，该模型是一个常数
# 如果是模型树， 其模型是一个线性方程。后面会看到停止条件的作用方式。 如果不满足停止条件， chooseBestSplit
#()将返回一个新的Python字典并将数据集分成两份， 在这两份数据集上将分别继续递归调用createTree(0 h函数

# 函数参数： 数据集 ， 其它3可选参数， leafType给出建立叶节点的函数 ； errType代表误差计算函数；
#ops 是一个包含树构建所需其它参数的元组
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops = (1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


# 它是回归树构建的核心函数，该函数的目的是找到数据的最佳二元切分方式。 如果找不到一个
#“好”的二元切分，该函数返回None并同时调用createTree() 来产生叶节点，叶节点的值也返回None
#在函数chooseBestSplit() 中有三种情况不会切分， 而是直接创建叶节点。 如果找到了一个好的切分方式， 则返回特征编号和切分特征值

#chooseBestSplit() 一开始为ops设定了tolS和tolN 这两个值。 它们是用户指定的参数， 用于控制函数的停止时机
#其中变量tolS是容许的误差下降值， tolN是切分的最少样本数。 接下来通过对当前所有目标变量建立一个集合，函数
#chooseBestSplit() 会统计不同剩余特征值的数目。 如果该数目为1 ， 那么就不需要在切分而直接返回。然后函数计算了当前数据集的
#大小和误差。 该误差S将用于与新切分误差进行对比， 来检测新切分能否降低误差。

#这样， 用于找到最佳切分的几个变量就被建立和初始化了。 下面就将所有可能的特征及其可能取值上遍历，找到最简单
#的切分方式 ，最佳切分也就是使得切分后能达到最低误差的切分。 如果切分数据集后效果提升不够大， 那么就不应进行切分操作
#而直接创建叶节点。 另外还需要检查两个切分后的子集大小， 如果某个子集的大小小于用户定义的参数tolN， 哪儿也不应
#切分。 最后，如果这些提前终止条件都不满足， 那么就返回切分特征和特征值

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    # 如果所有值相等则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) ==1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        #python2
        # for splitVal in set(dataSet[:,featIndex]):
        #python3
        for splitVal in set((dataSet[:, featIndex].T.A.tolist())[0]):
            mat0 , mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减少不大则退出
    if (S-bestS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    # 如果切分出的数据集很小则退出
    if (shape(mat0)[0]< tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue


#回归树剪枝函数
# isTree() 判断是否是一颗树， 用于判断当前处理的节点是否是叶子节点
def  isTree(obj):
    return (type(obj).__name__=='dict')
# getMean() 递归函数，它从上往下遍历树直到叶节点为止。 如果找到两个叶节点则计算它们的平均值。
# 该函数对树进行塌陷处理（即返回树的平均值）， 在prune() 函数中调用该函数时应明确这点。
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']) / 2.0

#参数： tree,待剪枝的树 testData 剪枝所需要的测试数据
# prune() 首先需要确认测试集是否为空。 一旦非空， 则反复递归调用函数prune() 对测试数据进行切分。
# 因为树是由其他数据集（训练集）生成的，所以测试集上会有一些样本与原数据集样本的取值范围不同。
# 一旦出现这种情况应该怎么办？数据发送过拟合应该进行剪枝么？或者模型正确不需要任何剪枝？
#这里假设发生了过拟合， 从而对树进行剪枝。

#接下来要检查某个分支到底是子树还是节点。 如果是子树， 就调用函数prune()来对该子树进行剪枝。 在
#对左右两个分支完成剪枝之后，还需要检查它们是否仍然还是子树。 如果两个分支已经不再是子树， 那么就可以进行合并。
#具体做法是对合并前后的误差进行比较。 如果合并后的误差比不合并的误差小就进行合并操作，反之则不合并直接返回
def prune(tree,testData):
    # 没有测试数据则对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] -tree['right'],2))
        treeMean = (tree['left']+tree['right']) / 2.0
        errorMerge = sum(power(testData[:,-1] - treeMean ,2 ))
        if errorMerge < errorNoMerge:
            print ("merging")
            return treeMean
        else:
            return tree
    else :
        return tree


# 模型树的叶节点生成函数

#主要功能是将数据集格式化成目标变量Y和自变量X , 与第8章类似。 X和Y用于执行监督的线性回归。 另外这个函数中
# 也应该注意， 如果矩阵的逆不存在也会造成程序异常
def linearSolve (dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0 :
        raise NameError ('This matrix is singular ,cannot do inverse , try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws ,X ,Y

# 函数modeLeaf() 与程序清单9-2里的函数regLeaf() 类似， 当数据不再需要切分的时候，
#它负责生产叶节点的模型。 该函数在数据集上调用linearSolve() 并返回回归系数ws
def modelLeaf (dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

# 函数modelErr() 可以在给定的数据集上计算误差。 它与程序清单9-2 的函数regErr（）类似，会被chooseBestSplit()
#调用来找到最佳的切分。 该函数在数据集上调用linearSolve()  之后返回yHat和Y之间的平方误差。
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    # 数组的元素分别求2次方
    return sum(power(Y-yHat,2))

#9-6用树回归进行预测的代码
#要对回归树叶节点进行预测， 就调用函数regTreeEval()
#要对模型树节点进行预测，就调用modelTreeEval()
#他们会对输入数据进行格式化 处理，在原数据矩阵上增加第0列。然后计算并返回预测值
#为了与函数modelTreeEval() 保持一致， 尽管regTreeEval() 只使用一个输入， 但仍保留了两个
#输入参数

#对回归树进行预测的函数
def regTreeEval(model,inDat):
    return float(model)

#对模型树进行预测的函数
def modelTreeEval(model , inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)


#d对于输入的单个数据点或行向量，函数treeForeCast 会返回一个浮点值
#在给定树结构的情况下，对于单个数据点， 该函数会给出一个预测值。
#调用函数treeForeCast() 时需要指定树的类型，以便在叶节点上能够调用合适的模型。
#参数modelEval是对叶节点数据进行预测的函数的引用。
#函数treeForeCast() 自顶向下遍历整棵树， 直到命中叶节点为止。
#  一旦到达叶节点， 它就会在输入数据上调用modelEval(),

def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

# 它会多次调用treeForeCast() 函数。 由于它能够以向量形式返回一组预测值， 因此该函数
#在对整个测试集进行预测时非常有用
def createForeCast(tree,testData,modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat



if __name__=='__main__':
    testMat = mat(eye(4))
    # 对角单位矩阵
    print (testMat)
    # 切分数据集
    mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    print (mat0)
    print (mat1)

    myDat=loadDataSet("./data/ex00.txt")
    myMat = mat(myDat)
    tree = createTree(myMat)
    print (tree)

    myDat1 = loadDataSet("./data/ex0.txt")
    myMat1 = mat(myDat1)
    tree1 = createTree(myMat1)
    print(tree1)

    myDat2 = loadDataSet("./data/ex2.txt")
    myMat2 = mat(myDat2)
    tree2 = createTree(myMat2)
    print(tree2)

    # 树剪枝 对过拟合树进行优化
    tree3 = createTree(myMat2,ops=(1,4))
    print (tree3)
    myTestData = loadDataSet("./data/ex2test.txt")
    myMatTest = mat(myTestData)
    newTree3 = prune(tree3,myMatTest)
    print (newTree3)

    # 模型树
    # {'spInd': 0, 'spVal': 0.285477, 'right': matrix([[ 3.46877936],
    #  [ 1.18521743]]), 'left': matrix([[  1.69855694e-03],
    #   [  1.19647739e+01]])}
    # 生成的模型树 ： y=3.468 + 1.1852x  和 y = 0.00169 + 11.96477x
    myMat4 = mat(loadDataSet("./data/exp2.txt"))
    modelTree = createTree(myMat4,modelLeaf,modelErr,(1,10))
    print (modelTree)

    # 9-6比较 回归树和模型树的好坏 利用智商和骑自行车数据
    trainMat = mat(loadDataSet("./data/bikeSpeedVsIq_train.txt"))
    testMat = mat(loadDataSet("./data/bikeSpeedVsIq_test.txt"))
    # 创建一个回归树
    myTree4 = createTree(trainMat,ops=(1,20))
    #使用回归树进行预测
    yHat = createForeCast(myTree4,testMat[:,0])
    #用于比较两组数据的拟合程度函数 值越接近1 ， 拟合的效果越好
    resultValue1=corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print ("使用回归树拟合效果 %.8f" % resultValue1)

    #创建模型树
    myTree5 = createTree(trainMat,modelLeaf,modelErr,(1,20))
    yHat = createForeCast(myTree5,testMat[:,0],modelTreeEval)
    resultValue2 = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print ("使用模型树拟合效果 %.8f" % resultValue2)

    #使用线性模型预测
    ws,X,Y = linearSolve(trainMat)
    print (ws)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0] * ws[1,0] + ws[0,0]
    resultValue3 = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print ("使用线性拟合效果 %.8f" % resultValue3)




