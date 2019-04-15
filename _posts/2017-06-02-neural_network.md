---
layout:     post
title:      " 神经网络"
subtitle:   "神经网络介绍 & 神经网络训练"
date:       2017-06-02 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 神经网络
---  
## 神经单元 *Neural Unit*

![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180128002-nn.gif)  
- 一个神经单元可以接受多个输入：**x1,x2...xn**，和一个偏执项**b**
- 神经单元的输出计算公式为：$f(x)=w^Tx+b$
- 最后是一个激活函数，这里使用Sigmod函数  

 $$
y=sigmode(z)=\frac{1}{1+e^{-z}}= \frac{1}{1+e^{-(w^Tx+b)}}
$$ 

##  全连接神经网络 *Full-Connection Neural Network*
![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180128003-nn.png)  
计算隐藏层第一层


 $$
\begin{aligned}
a_4&=sigmoid(\vec{w}^T\centerdot\vec{x})\\
&=sigmoid(w_{41}x_1+w_{42}x_2+w_{43}x_3+w_{4b})
\end{aligned}
$$ 


 $$
 \begin{aligned}
a_4&=sigmoid(w_{41}x_1+w_{42}x_2+w_{43}x_3+w_{4b}) \\
a_5&=sigmoid(w_{51}x_1+w_{52}x_2+w_{53}x_3+w_{5b})\\
a_6&=sigmoid(w_{61}x_1+w_{62}x_2+w_{63}x_3+w_{6b}) \\
a_7&=sigmoid(w_{71}x_1+w_{72}x_2+w_{73}x_3+w_{7b}) 
\end{aligned}
$$ 

计算输出层

 $$
\begin{aligned}
y_1&=sigmoid(\vec{w}^T\centerdot\vec{a})\\
&=sigmoid(w_{84}a_4+w_{85}a_5+w_{86}a_6+w_{87}a_7+w_{8b})
\end{aligned}
$$ 

##  反向传播 *Back Propagetion*
![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180128001-NN.png)  
#### 已有数据 *Given Data*

 $$
D=\{(X_1,Y_1),(X_2,Y_2),...,(X_n,Y_n)\}

$$ 

矩阵表示 *matrix indicate*

 $$
X=\begin{bmatrix}
1&x_1^1&x_2^1&...x_n^1 \\
1&x_1^2&x_2^2&...x_n^2 \\
..... \\ 
1&x_1^n&x_2^n&...x_n^n
\end{bmatrix},
Y=\begin{bmatrix} 
y_1\\
y_2\\
.....\\
y_n
\end{bmatrix}
$$ 

#### 损失函数  *Cost Function*

 $$
E_d\equiv\frac{1}{2}\sum_{i\in outputs}(t_i-y_i)^2
$$ 

#### 计算 $\mathbf{\Large {\frac{\Delta{E_d}}{\Delta{w}}}}$
$w_{ji}$只可能通过j层影像其他部分，设是节点$net_j$的加权输入

 $$
\begin{aligned}
net_j&=\vec{w_j}\centerdot\vec{x_j}\\
&=\sum_{i}{w_{ji}}x_{ji}
\end{aligned}
$$ 


$E_d$是$net_j$的函数，$net_j$而是$w_{ji}$的函数。根据**链式求导法则**:

 $$
\begin{aligned}
\frac{\Delta{E_d}}{\Delta{w_{ji}}}&=\frac{\Delta{E_d}}{\Delta{net_j}}\frac{\Delta{net_j}}{\Delta{w_{ji}}}\\
&=\frac{\Delta{E_d}}{\Delta{net_j}}\frac{\Delta{\sum_{i}{w_{ji}}x_{ji}}}{\Delta{w_{ji}}}\\
&=\frac{\Delta{E_d}}{\Delta{net_j}}x_{ji}
\end{aligned}
$$ 

#### 计算 $\mathbf{ \Large {\frac{\Delta{E_d}}{\Delta{net_j}}} }$
现在只需要做$\frac{\Delta{E_d}}{\Delta{net_j}}$的推导：
##### 输出层时  
- 输出层,$net_j$ 只能通过节点j的输出值$y_j$来影响其他部分
- 也就是$E_d$是$y_j$的函数，而$y_j$是$net_j$的函数
- 其中$y_j=sigmoid(net_j)$
- 可以使用**链式求导法则**:

 $$
\begin{aligned}
\frac{\Delta{E_d}}{\Delta{net_j}}&=\frac{\Delta{E_d}}{\Delta{y_j}}\frac{\Delta{y_j}}{\Delta{net_j}}   [\mathbf{EX1}] \\  
\end{aligned}
$$ 



 $$
\begin{aligned}
\frac{\Delta{E_d}}{\Delta{y_j}}&=\frac{\Delta}{\Delta{y_j}}\frac{1}{2}\sum_{i\in outputs}(t_i-y_i)^2\\
&=\frac{\Delta}{\Delta{y_j}}\frac{1}{2}(t_j-y_j)^2\\
&=-(t_j-y_j)  \  [\mathbf{EX2}] \\   
\end{aligned}
$$ 



 $$
\begin{aligned}
\frac{\Delta{y_j}}{\Delta{net_j}}&=\frac{\Delta sigmoid(net_j)}{\Delta{net_j}}\\
&=y_j(1-y_j)   \  [\mathbf{EX3}] \\   
\end{aligned}
$$ 


 $$
\frac{\Delta{E_d}}{\Delta{net_j}}=-(t_j-y_j)y_j(1-y_j)  \ [\mathbf{EX4}]  [\mathbf{USING \ EX1,EX2,EX3}]
$$ 

声明临时变量

 $$
\delta_j=-\frac{\Delta{E_d}}{\Delta{net_j}}=(t_j-y_j)y_j(1-y_j)   \ [\mathbf{EX5}] 

$$ 

利用EX5进行递归操作

 $$
\begin{aligned}
w_{ji}&\gets w_{ji}-\eta\frac{\Delta{E_d}}{\Delta{w_{ji}}}x_{ji}\\
&=w_{ji}+\eta(t_j-y_j)y_j(1-y_j)x_{ji}\\
&=w_{ji}+\eta\delta_jx_{ji}  [\mathbf{USING \ EX5}]
\end{aligned}
$$ 


##### 隐藏层时
- 首先定义节点$j$的所有直接下游节点集合$DownStream(j)$
- 所有$net_j$只能通过影响$DownStream(j)$再影响$E_d$
- 若$net_k$为$j$的下游节点输入
- 则$E_d$是$net_k$的函数，$net_k$是$net_j$的函数
- $a_i$是节点的输出值
- 因为$net_k$有多个，使用**全导数公式**进行推导：

 $$
\begin{aligned}
\frac{\Delta{E_d}}{\Delta{net_j}}&=\sum_{k\in Downstream(j)}\frac{\Delta{E_d}}{\Delta{net_k}}\frac{\Delta{net_k}}{\Delta{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_k\frac{\Delta{net_k}}{\Delta{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_k\frac{\Delta{net_k}}{\Delta{a_j}}\frac{\Delta{a_j}}{\Delta{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_kw_{kj}\frac{\Delta{a_j}}{\Delta{net_j}}\\
&=\sum_{k\in Downstream(j)}-\delta_kw_{kj}a_j(1-a_j)\\
&=-a_j(1-a_j)\sum_{k\in Downstream(j)}\delta_kw_{kj}   \ [\mathbf{EX8}] 
\end{aligned}
$$ 

声明临时变量

 $$
\delta_j=-\frac{\Delta{E_d}}{\Delta{net_j}}  [\mathbf{EX9}] 
$$ 

最后得到

 $$
\delta_j=a_j(1-a_j)\sum_{k\in Downstream(j)}\delta_kw_{kj}    \ [\mathbf{USING \ EX8,EX9}] 
$$ 


####  梯度下降 *Grendient Dscent*

 $$
\begin{aligned}
&Repeat\{  \\
&w_{ji}:= w_{ji}-\eta\frac{\Delta {E_d}}{\Delta {w_{ji}}} \\
&\}
\end{aligned}
$$ 



### 程序实现  
#### 使用`numpy`实现双层神经网络
```python  
# -*- coding: utf-8 -*-
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
``` 

> 代码参考 *https://github.com/hanbt/learn_dl/blob/master/bp.py*
