#-*-coding:utf-8-*- 
__author__ = 'Administrator'
from Tkinter import *
from numpy import *
import regTrees
import matplotlib
# 设置后端为 TkAgg
matplotlib.use('TkAgg')
# 将TkAgg 和 Matplotlib 图链接起来的声明导入
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



#Tkinter的GUI由一些小部件（Widget）组成。 所谓的小部件， 指的是文本框（Text Box），按钮框(Button)
# 标签（Label） 和 复选按钮（Check Button）等对象。 在例子中， 标签myLabel就是其中一个唯一的小部件。
# 当调用myLabel的.grid() 方法时，就等于把myLabel的位置告知了布局管理器（Geometry Manager）
# Tkinter中提供了几种不同的布局管理器， 其中的.grid()方法会把小部件安排在一个二维的表格里
# 用户可以设定每个小部件所在的行列位置。 这里没有做任何设定，mylabel会默认在0行0列
def tstTkinter():
    root=Tk()
    myLabel = Label(root,text='hello world')
    myLabel.grid()
    #启动事件循环，使该窗口在众多事件中可以响应鼠标点击 按键 重会等动作
    root.mainloop()


#Matplotlib 的构建程序包含一个前端， 也就是面向用户的一些代码， 如plot（）和 Scatter()
#方法等。 它同时创建一个后端，用于实现绘图和不同应用之间的接口



# 绘制树
def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN=2
        myTree = regTrees.createTree(reDraw.rawDat,regTrees.modelLeaf,regTrees.modelErr,(tolS,tolN))
        yHat = regTrees.createForeCast(myTree,reDraw.testDat,regTrees.modelTreeEval)
    else :
        myTree = regTrees.createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat = regTrees.createForeCast(myTree,reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0],reDraw.rawDat[:,1],s=5)
    reDraw.a.plot(reDraw.testDat,yHat,linewidth=2.0)
    reDraw.canvas.show()



def drawNewTree():
    tolN,tolS = getInputs()
    reDraw(tolS,tolN)


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print ("enter Intger for tolN")
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print "enter Float for tolS"
        tolSentry.delete(0,END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS


#9-6 代码建立了一组Tkinter模块，并用网络布局管理器安排了它们的位置，这里还给出了两个绘制占位符
#（plot placeholder）函数， 函数的内容在后面补充。 这里所使用的代码的格式与前面的例子一致，即首先创建一个Tk
#类型的根部件然后插入标签。 读者可以使用.grid() 方法设定行和列的位置。
# 另外，也可以通过设定columnspan 和 rowspan 的值来告诉布局管理器是否允许一个小部件跨行或跨列。
# 除此之外还有其他的设置项可供使用

#还有一些新的小部件暂时未使用到， 这些小部件包括文本输入框（Entry），复选按钮(CheckButton)和按钮整数值(IntVar)等
#其中Entry部件是一个允许单行文本输入的文本框。 Checkbutton 和 IntVar 的功能显而易见： 为了读取CheckBButton 的状态需要
#创建一个变量，也就是IntVar

#最后初始化一些与reDraw() 关联的全局变量， 这些变量会在后面用到。
root = Tk()


reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root,text="Plot Place  Holder").grid(row=0,columnspan=3)
Label(root,text='tolN').grid(row=1,column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1,column = 1)
tolNentry.insert(0,'10')
Label(root,text='tolS').grid(row=2,column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')
Button(root,text='ReDraw',command=drawNewTree).grid(row=1,column=2,rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root,text='Model Tree',variable = chkBtnVar)
chkBtn.grid(row=3,column = 0 , columnspan = 2)
reDraw.rawDat = mat(regTrees.loadDataSet("./data/sine.txt"))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0,10)
# 退出按钮框
# Button(root,text='Quit',fg='black',command=root.quit).grid(row=1,column=2)

root.mainloop()


#将matplotlib应用到GUI上，下面首先介绍‘后端’的概念，然后通过修改
#Matplotlib 后端（仅在我们的GUI上）达到在Tkinter的GUI上绘图的目的




