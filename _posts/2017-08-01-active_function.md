---
layout:     post
title:      "激活函数"
subtitle:   "Active Function"
date:       2017-07-08 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 机器学习
---
#### Sigmode Function
![image](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

$$  
 φ(x) = \dfrac{1}{ {1-e^{ \ -x}}} 
$$   

```python
# Required Python Package
import numpy as np

def sigmoid(inputs):
    """
    Calculate the sigmoid for the give inputs (array)
    :param inputs:
    :return:
    """
    sigmoid_scores = [1 / float(1 + np.exp(- x)) for x in inputs]
    return sigmoid_scores


sigmoid_inputs = [2, 3, 5, 6]
print "Sigmoid Function Output :: {}".format(sigmoid(sigmoid_inputs))
```

- 常用于二进制的分类
- 
#### Softmax Function

 $$
φ(x) = \dfrac{e^{z_j}}{\sum_{k=1}^Ke^{z_k}}
$$ 


--计算0~1 的概率

```python
# Required Python Package
import numpy as np

def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))

softmax_inputs = [2, 3, 5, 6]
print "Softmax Function Output :: {}".format(softmax(softmax_inputs))
``` 


#### Relu Function
![Image](http://upload-images.jianshu.io/upload_images/2256672-0ac9923bebd3c9dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/640)

 $$
f(x)= max(0,x)
$$ 

##### 优势
- **速度快**  和sigmoid函数需要计算指数和倒数相比，relu函数其实就是一个max(0,x)，计算代价小很多
- **减轻梯度消失问题** 回忆一下计算梯度的公式$\nabla=\sigma'\delta x$。其中，$\sigma'$是sigmoid函数的导数。在使用反向传播算法进行梯度计算时，每经过一层sigmoid神经元，梯度就要乘上一个$\sigma'$。从下图可以看出，$\sigma'$函数最大值是1/4。因此，乘一个$\sigma'$会导致梯度越来越小，这对于深层网络的训练是个很大的问题。而relu函数的导数是1，不会导致梯度变小。当然，激活函数仅仅是导致梯度减小的一个因素，但无论如何在这方面relu的表现强于sigmoid。使用relu激活函数可以让你训练更深的网络。  
![image](http://upload-images.jianshu.io/upload_images/2256672-ad98d6b22f1a66ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/360)
- **稀疏性** 通过对大脑的研究发现，大脑在工作的时候只有大约5%的神经元是激活的，而采用sigmoid激活函数的人工神经网络，其激活率大约是50%。有论文声称人工神经网络在15%-30%的激活率时是比较理想的。因为relu函数在输入小于0时是完全不激活的，因此可以获得一个更低的激活率
##### Reference  
>  Wikipedia Activation function *https://en.wikipedia.org/wiki/Activation_function*