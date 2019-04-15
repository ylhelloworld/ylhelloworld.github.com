---
layout:     post
title:      " 神经单元"
subtitle:   "线性神经单元 & 非线性神经单元"
date:       2017-06-01 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 神经网络
---  



**线性神经单元和非线性神经单元的区别在激活函数的不同**

## 线性神经单元
![Image](http://upload-images.jianshu.io/upload_images/2256672-801d65e79bfc3162.png)


#### 激活函数
- 激活函数使用阶跃函数

 $$
f(z)=\begin{cases}
 & 1\qquad z>0\\ 
 &0\qquad otherwise
\end{cases}
$$ 


#### 输出函数


 $$
y=f(\mathrm{w}\bullet\mathrm{x}+b)\qquad 
$$ 


> 代码实现参考 *https://github.com/hanbt/learn_dl/blob/master/perceptron.py*

## 非线性神经单元
![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180128002-nn.gif) 

#### 激活函数
- 激活函数使用Sigmod函数

 $$
y=sigmode(z)=\frac{1}{1+e^{-z}}
$$ 


#### 输出函数

 $$
y=sigmode(z)=\frac{1}{1+e^{-(w^Tx+b)}}
$$ 

>  代码实现参考  *https://github.com/hanbt/learn_dl/blob/master/linear_unit.py *

## 梯度下降

#### 优化的目标函数

 $$
E(\mathrm{w})=\frac{1}{2}\sum_{i=1}^{n}(\mathrm{y^{(i)}-\bar{y}^{(i)}})^2
$$ 

#### 梯度下降

 $$
\mathrm{w}_{new}=\mathrm{w}_{old}-\eta\nabla{E(\mathrm{w})}
$$ 


#### 目标函数的梯度$\nabla{E(\mathrm{w})}$

 $$
\nabla{E(\mathrm{w})}=-\sum_{i=1}^{n}(y^{(i)}-\bar{y}^{(i)})\mathrm{x}^{(i)}
$$ 


 $$
\begin{bmatrix}
w_0 \\
w_1 \\
w_2 \\
... \\
w_m \\
\end{bmatrix}_{new}=
\begin{bmatrix}
w_0 \\
w_1 \\
w_2 \\
... \\
w_m \\
\end{bmatrix}_{old}+\eta\sum_{i=1}^{n}(y^{(i)}-\bar{y}^{(i)})
\begin{bmatrix}
1 \\
x_1^{(i)} \\
x_2^{(i)} \\
... \\
x_m^{(i)} \\
\end{bmatrix}
$$ 

#### 链式推导发进行推导$\nabla{E(\mathrm{w})}$


 $$
\begin{aligned}
\nabla{E(\mathrm{w})}&=\frac{\partial}{\partial\mathrm{w}}E(\mathrm{w})\\
&=\frac{\partial}{\partial\mathrm{w}}\frac{1}{2}\sum_{i=1}^{n}(y^{(i)}-\bar{y}^{(i)})^2\\
&=\frac{1}{2}\sum_{i=1}^{n}\frac{\partial}{\partial\mathrm{w}}(y^{(i)}-\bar{y}^{(i)})^2\\
\end{aligned}

$$ 

- 先求累加的部分，便于推导

 $$
\begin{aligned}
&\frac{\partial}{\partial\mathrm{w}}(y^{(i)}-\bar{y}^{(i)})^2\\
=&\frac{\partial}{\partial\mathrm{w}}(y^{(i)2}-2\bar{y}^{(i)}y^{(i)}+\bar{y}^{(i)2})\\
\end{aligned}
$$ 

- 偏导数的推导

 $$
\frac{\partial{E(\mathrm{w})}}{\partial\mathrm{w}}=\frac{\partial{E(\bar{y})}}{\partial\bar{y}}\frac{\partial{\bar{y}}}{\partial\mathrm{w}}
$$ 

- 第一部分的推导

 $$
\begin{aligned}
\frac{\partial{E(\mathrm{w})}}{\partial\bar{y}}=
&\frac{\partial}{\partial\bar{y}}(y^{(i)2}-2\bar{y}^{(i)}y^{(i)}+\bar{y}^{(i)2})\\
=&-2y^{(i)}+2\bar{y}^{(i)}\\ 
\end{aligned}
$$ 

- 第二部分的推导
 
 $$
\begin{aligned}
\frac{\partial{\bar{y}}}{\partial\mathrm{w}}=
&\frac{\partial}{\partial\mathrm{w}}\mathrm{w}^T\mathrm{x}\\
=&\mathrm{x} \\ 
\end{aligned}
$$ 

- 前后两部分进行合并

 $$
\begin{aligned}
\nabla{E(\mathrm{w})}&=\frac{1}{2}\sum_{i=1}^{n}\frac{\partial}{\partial\mathrm{w}}(y^{(i)}-\bar{y}^{(i)})^2\\
&=\frac{1}{2}\sum_{i=1}^{n}2(-y^{(i)}+\bar{y}^{(i)})\mathrm{x}\\
&=-\sum_{i=1}^{n}(y^{(i)}-\bar{y}^{(i)})\mathrm{x}
\end{aligned}
$$ 
