---
layout:     post
title:      "长短时序神经网络"
subtitle:   "基础神经网络&双向神经网络"
date:       2017-06-05 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 神经网络
---


## Introduction Long Short Term Memory Network

与传统的RNN相比，LSTM多出一个细胞状态链用于存储细胞状态，LSTM中包含了Gate的概念，新增3个门来控制细胞状态，包括遗忘门（Forget Gate)、输入门（Input Gate）、输出门（Output Gate）
- LSTM 多出一个Cell状态链
![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180122017-LSTM3-C-line.png)

- 传统的RNN只包含一tanh层

![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180122015-LSTM3-SimpleRNN.png)


- LSTM 含有四个交互层
![IMAGE](http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180122016-LSTM2-notation.png)

- Forget Gate 控制丢弃的信息

![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180122019-LSTM3-focus-f.png)

 $$
\mathbf{f}_t=\sigma(W_f\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f)\qquad\quad
$$ 

- Input Gate 控制更新的信息


![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180122020-LSTM3-focus-i.png)

 $$
\mathbf{i}_t=\sigma(W_i\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i)\qquad\quad 

\mathbf{\tilde{C}}_t=\tanh(W_c\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c)\qquad\quad
$$ 



- 更新细胞状态

![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180122021-LSTM3-focus-C.png)

 $$
\mathbf{c}_t=f_t\circ{\mathbf{c}_{t-1}}+i_t\circ{\mathbf{\tilde{c}}_t}\qquad\quad
$$ 

- Output Gate控制输出的信息
![IMAGE](https://raw.githubusercontent.com/ylhelloworld/resource/master/Image/20180122022-LSTM3-focus-o.png)

 $$
\mathbf{o}_t=\sigma(W_o\cdot[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o)\qquad\quad 

\mathbf{h}_t=\mathbf{o}_t\circ \tanh(\mathbf{c}_t)\qquad\quad 
$$ 

## Train LSTM
- 向前计算每个神经单元的值，包括$\mathbf{f}_t$、$\mathbf{i}_t$、$\mathbf{c}_t$、$\mathbf{o}_t$、$\mathbf{h}_t$  
- 反向计算每个神经元的误差$\delta$
- 计算每个权重的梯度

#### Define Value
- 定义t时刻的误差

 $$
\delta_t\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{h}_t}}
$$ 

- 定义各层的值和误差项

 $$
\begin{aligned}
\mathbf{net}_{f,t}&=W_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f\\
&=W_{fh}\mathbf{h}_{t-1}+W_{fx}\mathbf{x}_t+\mathbf{b}_f\\
\mathbf{net}_{i,t}&=W_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i\\
&=W_{ih}\mathbf{h}_{t-1}+W_{ix}\mathbf{x}_t+\mathbf{b}_i\\
\mathbf{net}_{\tilde{c},t}&=W_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c\\
&=W_{ch}\mathbf{h}_{t-1}+W_{cx}\mathbf{x}_t+\mathbf{b}_c\\
\mathbf{net}_{o,t}&=W_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o\\
&=W_{oh}\mathbf{h}_{t-1}+W_{ox}\mathbf{x}_t+\mathbf{b}_o\\
\delta_{f,t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{f,t}}}\\
\delta_{i,t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{i,t}}}\\
\delta_{\tilde{c},t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{\tilde{c},t}}}\\
\delta_{o,t}&\overset{def}{=}\frac{\partial{E}}{\partial{\mathbf{net}_{o,t}}}\\
\end{aligned}

$$ 

#### Calculate $\mathbf{  \Large { \frac{\partial{E}}{\partial{w}}} }$

 $$
\begin{aligned}
\frac{\partial{E}}{\partial{w}}=&\frac{\partial{E}}{\partial{net_j^t}}\frac{\partial{net_j^t}}{\partial{w}}\\
=&\delta_j^ts_i^{t-1}   
\end{aligned}
$$ 

- $\delta_j^t=\frac{\partial{E}}{\partial{net_j^t}}$  (refrence Calculate $\mathbf{ \delta_k^T }$)
- $h_i^{t-1} =\frac{\partial{net_j^t}}{\partial{w_{ji}}}$

#### Calculate $\mathbf{ \large{\delta_k^T }}$

 $$
\delta_k^T=\prod_{j=k}^{t-1}\delta_{o,j}^TW_{oh}
+\delta_{f,j}^TW_{fh}
+\delta_{i,j}^TW_{ih}
+\delta_{\tilde{c},j}^TW_{ch}\qquad\quad
$$ 


#### Calculate $\mathbf{ \large{\frac{\partial{E}}{\partial{\mathbf{net}_t^{l-1}}}}}$

 $$
\begin{aligned}
\frac{\partial{E}}{\partial{\mathbf{net}_t^{l-1}}}=(\delta_{f,t}^TW_{fx}+\delta_{i,t}^TW_{ix}+\delta_{\tilde{c},t}^TW_{cx}+\delta_{o,t}^TW_{ox})\circ f'(\mathbf{net}_t^{l-1})\qquad\quad
\end{aligned}
$$ 

#### Calculate $\mathbf{ \large{ \frac{\partial{E}}{\partial{w}}}}$

 $$
\begin{aligned}
\frac{\partial{E}}{\partial{\mathbf{b}_o}}&=\sum_{j=1}^t\delta_{o,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_i}}&=\sum_{j=1}^t\delta_{i,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_f}}&=\sum_{j=1}^t\delta_{f,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_c}}&=\sum_{j=1}^t\delta_{\tilde{c},j}\\
\end{aligned}

$$ 


 $$
\begin{aligned}
\frac{\partial{E}}{\partial{\mathbf{b}_o}}&=\sum_{j=1}^t\delta_{o,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_i}}&=\sum_{j=1}^t\delta_{i,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_f}}&=\sum_{j=1}^t\delta_{f,j}\\
\frac{\partial{E}}{\partial{\mathbf{b}_c}}&=\sum_{j=1}^t\delta_{\tilde{c},j}\\
\end{aligned}
$$ 


## Refrence Math Formula
#### Active Function Sigmod & Derivative 

 $$
\begin{aligned}
\sigma(z)&=y=\frac{1}{1+e^{-z}}\\
\sigma'(z)&=y(1-y)\\ 
\end{aligned}
$$ 

#### Active Function Tanx & Derivative

 $$
\begin{aligned} 
\tanh(z)&=y=\frac{e^z-e^{-z}}{e^z+e^{-z}}\\
\tanh'(z)&=1-y^2
\end{aligned}
$$ 


#### Vector $\circ$ Vector

 $$
\mathbf{a}\circ\mathbf{b}=\begin{bmatrix}
a_1\\a_2\\a_3\\...\\a_n
\end{bmatrix}\circ\begin{bmatrix}
b_1\\b_2\\b_3\\...\\b_n
\end{bmatrix}=\begin{bmatrix}
a_1b_1\\a_2b_2\\a_3b_3\\...\\a_nb_n
\end{bmatrix}
$$ 

#### Vector $\circ$ Matrix

 $$
\begin{aligned}
\mathbf{a}\circ X&=\begin{bmatrix}
a_1\\a_2\\a_3\\...\\a_n
\end{bmatrix}\circ\begin{bmatrix}
x_{11} & x_{12} & x_{13} & ... & x_{1n}\\
x_{21} & x_{22} & x_{23} & ... & x_{2n}\\
x_{31} & x_{32} & x_{33} & ... & x_{3n}\\
& & ...\\
x_{n1} & x_{n2} & x_{n3} & ... & x_{nn}\\
\end{bmatrix}\\
&=\begin{bmatrix}
a_1x_{11} & a_1x_{12} & a_1x_{13} & ... & a_1x_{1n}\\
a_2x_{21} & a_2x_{22} & a_2x_{23} & ... & a_2x_{2n}\\
a_3x_{31} & a_3x_{32} & a_3x_{33} & ... & a_3x_{3n}\\
& & ...\\
a_nx_{n1} & a_nx_{n2} & a_nx_{n3} & ... & a_nx_{nn}\\
\end{bmatrix}
\end{aligned}
$$ 

