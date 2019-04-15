---
layout:     post
title:      "递归神经网络-整理中..."
subtitle:   "Basic Recurisive Neural Network-ing...."
date:       2017-06-20 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 神经网络
---

### Basic Recursive Nerual Network  
![Image](http://upload-images.jianshu.io/upload_images/2256672-f2ea8885320110a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


 $$
\mathbf{p} = tanh(W\begin{bmatrix}\mathbf{c}_1\\\mathbf{c}_2\end{bmatrix}+\mathbf{b})\qquad
$$ 


### Train Recursive Nerual Network
![image](http://upload-images.jianshu.io/upload_images/2256672-9ab001431eb2f2a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### Define Param
- $\mathbf{net}_p$为父节点的加权输入

 $$
\mathbf{net}_p=W\begin{bmatrix}\mathbf{c}_1\\\mathbf{c}_2\end{bmatrix}+\mathbf{b}
$$ 

- 加权输入展开如下

 $$
\begin{aligned}
\begin{bmatrix}
net_{p_1}\\
net_{p_2}\\
...\\
net_{p_n}
\end{bmatrix}&=
\begin{bmatrix}
w_{p_1c_{11}}&w_{p_1c_{12}}&...&w_{p_1c_{1n}}&w_{p_1c_{21}}&w_{p_1c_{22}}&...&w_{p_1c_{2n}}\\
w_{p_2c_{11}}&w_{p_2c_{12}}&...&w_{p_2c_{1n}}&w_{p_2c_{21}}&w_{p_2c_{22}}&...&w_{p_2c_{2n}}\\
...\\
w_{p_nc_{11}}&w_{p_nc_{12}}&...&w_{p_nc_{1n}}&w_{p_nc_{21}}&w_{p_nc_{22}}&...&w_{p_nc_{2n}}\\
\end{bmatrix}
\begin{bmatrix}
c_{11}\\
c_{12}\\
...\\
c_{1n}\\
c_{21}\\
c_{22}\\
...\\
c_{2n}
\end{bmatrix}+\begin{bmatrix}\\
b_1\\
b_2\\
...\\
b_n\\
\end{bmatrix}
\end{aligned}
$$ 


- $\delta_p$为误差函数相对于父节点p的加权输入$\mathbf{net}_p$的导数

 $$
\delta_p\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_p}}
$$ 


-计算

 $$
\begin{aligned}
\frac{\partial{E}}{\partial{c_{jk}}}&=\sum_i{\frac{\partial{E}}{\partial{net_{p_i}}}}\frac{\partial{net_{p_i}}}{\partial{c_{jk}}}\\
&=\sum_i{\delta_{p_i}}w_{p_ic_{jk}}
\end{aligned}
$$ 
