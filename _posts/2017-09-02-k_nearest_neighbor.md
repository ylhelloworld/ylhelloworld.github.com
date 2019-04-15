---
layout:     post
title:      "最近邻算法"
subtitle:   "k-Nearest Neighbor"
date:       2017-07-20 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 机器学习
---
## KNN  *k-Nearest Neighbor* *最近邻算法*
### Refrence  
> k-nearest neighbors algorithm
 https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm  
 
 ![image](https://gss2.bdstatic.com/9fo3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=8fabe2c8d5c8a786aa27425c0660a258/03087bf40ad162d95867202e15dfa9ec8a13cd73.jpg)  
 所谓K近邻算法，即是给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的K个实例（也就是上面所说的K个邻居）， 这K个实例的多数属于某个类，就把该输入实例分类到这个类中。
 
#### 基本步骤：
- 计算测试数据与各个训练数据之间的距离  
- 按照距离的递增关系进行排序；
- 选取距离最小的K个点；
- 确定前K个点所在类别的出现频率；
- 返回前K个点中出现频率最高的类别作为测试数据的预测分类。
 
> 距离的计算一般为曼哈顿或欧式距离  

#### 优缺点
优点
- 思想简单，理论成熟，既可以用来做分类也可以用来做回归； 
- 可用于非线性分类； 
- 无时序性问题； 

缺点 
- 计算量大； 
- 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）； 
- 需要大量的内存；

## Solution  


### 01.优化约会网站的配对效果

海伦使用约会网站寻找约会对象。经过一段时间之后，她发现曾交往过三种类型的人:
* 不喜欢的人
* 魅力一般的人
* 极具魅力的人

她希望：
1. 工作日与魅力一般的人约会
2. 周末与极具魅力的人约会
3. 不喜欢的人则直接排除掉

现在她收集到了一些约会网站未曾记录的数据信息，这更有助于匹配对象的归类。

开发流程

```
收集数据：提供文本文件
准备数据：使用 Python 解析文本文件
分析数据：使用 Matplotlib 画二维散点图
训练算法：此步骤不适用于 k-近邻算法
测试算法：使用海伦提供的部分数据作为测试样本。
     测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。
```

- **收集数据**：提供文本文件

海伦把这些约会对象的数据存放在文本文件 [datingTestSet2.txt](../input/2.KNN/datingTestSet2.txt) 中，总共有 1000 行。海伦约会的对象主要包含以下 3 种特征：

* 每年获得的飞行常客里程数
* 玩视频游戏所耗时间百分比
* 每周消费的冰淇淋公升数

文本文件数据格式如下：
```
40920	8.326976	0.953952	3
14488	7.153469	1.673904	2
26052	1.441871	0.805124	1
75136	13.147394	0.428964	1
38344	1.669788	0.134296	1
```
- **准备数据**：使用 Python 解析文本文件

将文本记录转换为 NumPy 的解析程序
 
 ```python
def file2matrix(filename):
    """
    Desc:
        导入训练数据
    parameters:
        filename: 数据文件路径
    return: 
        数据矩阵 returnMat 和对应的类别 classLabelVector
    """
    fr = open(filename)
    # 获得文件中的数据行的行数
    numberOfLines = len(fr.readlines())
    # 生成对应的空矩阵
    # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0 
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回已移除字符串头尾指定字符所生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector
```

- **分析数据**：使用 Matplotlib 画二维散点图

```python
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
```

下图中采用矩阵的第一和第二列属性得到很好的展示效果，清晰地标识了三个不同的样本分类区域，具有不同爱好的人其类别区域也不同。

![Matplotlib 散点图](http://ml.apachecn.org/images/2.KNN/knn_matplotlib_2.png)

* 归一化数据 （归一化是一个让权重变为统一的过程，更多细节请参考： https://www.zhihu.com/question/19951858 ）

| 序号 | 玩视频游戏所耗时间百分比 | 每年获得的飞行常客里程数  | 每周消费的冰淇淋公升数  | 样本分类 |
| ------------- |:-------------:| -----:| -----:| -----:|
| 1 | 0.8 | 400     | 0.5 | 1 |
| 2 | 12  | 134 000 | 0.9 | 3 |
| 3 | 0   | 20 000  | 1.1 | 2 |
| 4 | 67  | 32 000  | 0.1 | 2 |

样本3和样本4的距离：
$$\sqrt{(0-67)^2 + (20000-32000)^2 + (1.1-0.1)^2 }$$

归一化特征值，消除特征之间量级不同导致的影响

**归一化定义：** 我是这样认为的，归一化就是要把你需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。首先归一化是为了后面数据处理的方便，其次是保正程序运行时收敛加快。 方法有如下：

1) 线性函数转换，表达式如下：　　
    
    y=(x-MinValue)/(MaxValue-MinValue)　　

    说明：x、y分别为转换前、后的值，MaxValue、MinValue分别为样本的最大值和最小值。　　

2) 对数函数转换，表达式如下：　　

    y=log10(x)　　

    说明：以10为底的对数函数转换。

    如图：
    
    ![对数函数图像](http://ml.apachecn.org/images/2.KNN/knn_1.png)

3) 反余切函数转换，表达式如下：

    y=arctan(x)*2/PI　

    如图：
    
    ![反余切函数图像](http://ml.apachecn.org/images/2.KNN/knn_2.jpg)

4) 式(1)将输入值换算为[-1,1]区间的值，在输出层用式(2)换算回初始值，其中和分别表示训练样本集中负荷的最大值和最小值。　
    

在统计学中，归一化的具体作用是归纳统一样本的统计分布性。归一化在0-1之间是统计的概率分布，归一化在-1--+1之间是统计的坐标分布。


```python
def autoNorm(dataSet):
    """
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals
```

- **训练算法**：此步骤不适用于 k-近邻算法

因为测试数据每一次都要与全量的训练数据进行比较，所以这个过程是没有必要的。

kNN 算法伪代码：

    对于每一个在数据集中的数据点：
        计算目标的数据点（需要分类的数据点）与该数据点的距离
        将距离排序：从小到大
        选取前K个最短距离
        选取这K个中最多的分类类别
        返回该类别来作为目标数据点的预测值
```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #距离度量 度量公式为欧氏距离
    diffMat = tile(inX, (dataSetSize,1)) – dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    #将距离排序：从小到大
    sortedDistIndicies = distances.argsort()
    #选取前K个最短距离， 选取这K个中最多的分类类别
    classCount={}
    for i in range(k)：
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```


- **测试算法**：使用海伦提供的部分数据作为测试样本。如果预测分类与实际类别不同，则标记为一个错误。

kNN 分类器针对约会网站的测试代码

```python
def datingClassTest():
    """
    Desc:
        对约会网站的测试方法
    parameters:
        none
    return:
        错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('input/2.KNN/datingTestSet2.txt')  # load data setfrom file
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print 'numTestVecs=', numTestVecs
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print errorCount
```

- **使用算法**：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

约会网站预测函数

```python
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games ?"))
    ffMiles = float(raw_input("frequent filer miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]
```

实际运行效果如下: 

```python
>>> classifyPerson()
percentage of time spent playing video games?10
frequent flier miles earned per year?10000
liters of ice cream consumed per year?0.5
You will probably like this person: in small doses
```

### 02.手写数字识别系统

构造一个能识别数字 0 到 9 的基于 KNN 分类器的手写数字识别系统。
需要识别的数字是存储在文本文件中的具有相同的色彩和大小：宽高是 32 像素 * 32 像素的黑白图像。

开发流程

```
收集数据：提供文本文件。
准备数据：编写函数 img2vector(), 将图像格式转换为分类器使用的向量格式
分析数据：在 Python 命令提示符中检查数据，确保它符合要求
训练算法：此步骤不适用于 KNN
测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的
         区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，
         则标记为一个错误
使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取
         数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统
```

- **收集数据**: 提供文本文件

目录中包含了大约 2000 个例子，每个数字大约有 200 个样本；目录中包含了大约 900 个测试数据。
 

- **准备数据**: 编写函数 img2vector(), 将图像文本数据转换为分类器使用的向量

将图像文本数据转换为向量

```python
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
```

- **分析数据**：在 Python 命令提示符中检查数据，确保它符合要求

在 Python 命令行中输入下列命令测试 img2vector 函数，然后与文本编辑器打开的文件进行比较: 

```python
>>> testVector = kNN.img2vector('testDigits/0_13.txt')
>>> testVector[0,0:32]
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
>>> testVector[0,32:64]
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

- **训练算法**：此步骤不适用于 KNN

因为测试数据每一次都要与全量的训练数据进行比较，所以这个过程是没有必要的。

- **测试算法**：编写函数使用提供的部分数据集作为测试样本，如果预测分类与实际类别不同，则标记为一个错误

```python
def handwritingClassTest():
    # 1. 导入训练数据
    hwLabels = []
    trainingFileList = listdir('input/2.KNN/trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('input/2.KNN/trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('input/2.KNN/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('input/2.KNN/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))
```

- **使用算法**：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统。


### Programing   
#### Base Example

```python
from numpy import *
import operate

#准备导入数据  
def createDataSet()
    group=arry([[1.0,1.1),[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
#分类算法
    #inX:分类的输入向量
    #dataSet、labels：训练样本集及对应的标签
    #k:选择最近相邻的数目

def classfiy(inX,dataSet,labels,k)
    #欧式距离计算 
    dataSetSize=dataSet.shape()
    diffMath=tile(inX,(dataSetSize,1)) -dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMath.sum(axis=1)
    distances=sqlDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    
    #选择距离最小的k个点
    for i in range(k):
        votelabel=labels[sortedDistIndicies[i]]
        classCount[voteIlbel]=classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount=soted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


```

#### Sklearn 
```python
#Import Library
from sklearn.neighbors import KNeighborsClassifier
 
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object model 
KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
 
# Train the model using the training sets and check score
model.fit(X, y)
 
#Predict Output
predicted= model.predict(x_test)
```