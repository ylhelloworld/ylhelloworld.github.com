---
layout:     post
title:      "逻辑回归"
subtitle:   "Logstic Regression"
date:       2017-06-20 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 机器学习
---
## Logistic Regression *逻辑回归*   

### Theory
#### Given Data *已有数据*

 $$
D=\{(X_1,Y_1),(X_2,Y_2),...,(X_n,Y_n)\}
$$


matrix indicate

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


#### Hypothesis Function *期望函数*
the sigmoid function is:  
*when z -> ∞ then f(z)-> 1  ,when  z -> -∞ then f(z)-> 0*

 $$
f(z)=\frac{1}{1+e^{-z}} 
$$ 




the hypothesis function is:  
*期望函数为：*

 $$
h_w(x)=f(w^Tx)=\frac{1}{1+e^{w^Tx}}    [\mathbf{EX0}]
$$ 

#### Cost Function 
the direcvtive of the sigmoid function:  
*sigmoid 函数的导数为：*

 $$
\begin{aligned}
f'(z)&=\frac{\Delta}{\Delta z} \left( \frac{1}{1+e^-z} \right) \\
&=\frac{1}{(1+e^-z)^2}e^{-z} \\
&=\frac{1}{1+e^{-z}}\left ( 1- \frac{1}{1+e^{-z}} \right) \\
&=f(z)(1-f(z)) [\mathbf{EX1}]
\end{aligned}
$$ 

let calculate Probabity  
*计算概率*


 $$
\begin{aligned}
&P(y=1|x;w)=h_w(x)\\
&P(y=0|x;w)=1-h_w(x)
\end{aligned}
$$ 

writen in compact notaion as  
*缩写形式为*

 $$
P(y|x;w)=(h_w(x)^y(1-h_w(x))^{1-y}

$$ 

the likelyhood of the parameter w can be written as   
*w的可能性可以缩写为*

 $$
L(w)=P(y|X:w)
$$ 

assuming that the taining examples are all indepdent  
*假设测设数据都是独立的事件*


 $$
\begin{aligned}
L(w)&=\prod_{i=1}^{m}P(y^i|x^i;w)  \\
&=\prod_{i=1}^{m}(h_w(x^i))^{y^i}(1-h_w(x^i))^{1-y^i}
\end{aligned}
$$ 

Let G(z) continues incresing functiomn,put z=L(w),L(w) is maximum then G(z) will also be maximum,maximise G(z)=log(z)  
*G(z)也是一个递增函数，若z=L(w),若 L(w)最大化也是G(z)最大化*

 $$
\begin{aligned} 
\varphi(w)&=log(L(w)) \\
&=log(\prod_{i=1}^{m}(h_w(x^i))^{y^i}(1-h_w(x^i))^{1-y^i}) \\
&=\sum_{i=1}^m(y^ilog(h_w(x^i)+(1-y^i)log(1-h(x^i))) [\mathbf{EX2}]
\end{aligned}

$$ 

#### Maximise Cost 
Maximise the cost function **$\varphi (x)$**   
*最大化cost function* 


 $$
 \begin{aligned}
&\frac{\Delta\varphi(w)}{\Delta w_j}=\left(y^i \frac{1}{f(w^Tx^j)}-(1-y^i)\frac{1}{1-f(w^Tx^j)} \right)\frac{\Delta f(w^Tx^j)}{\Delta w_j}   \ [\mathbf{Using \  EX0,EX2}]   \\

&\frac{\Delta\varphi(w)}{\Delta w_j}=\left(y^i \frac{1}{f(w^Tx^j)}-(1-y^i)\frac{1}{1-f(w^Tx^j)} \right)f(w^Tx^j)(1-f(w^Tx^j))\frac{\Delta w^Tx^j}{\Delta w_j} \ [\mathbf{ \ Using EX1}] s
\end{aligned}
$$ 

solving about 
*解方程得到结果：* 

 $$
\begin{aligned}
&=(y^i(1-f(w^Tx^i)-(1-y^i)(w^Tx^i)x_j^i \\
&=(y^i-f(w^Tx^i)x^i_j \   [\mathbf{EX3}]
\end{aligned}

$$ 


#### Gradient Ascent
loop gradient ascent to update w  
*循环剃度上升更新参数w*


 $$
\begin{aligned}
&Repeat\{  \\
&w:=w+\eta \Delta_w \varphi(w) \\
&w_j:=w_j+ \eta (y^i-f(w_Tx)x_j^i \\
&w_j:=w_j+ \eta (y^i-h(x^i))x_j^i \   [\mathbf{Using \ EX1}] \\
&\}
\end{aligned}
$$ 


### Exapmple
#### Exam Pass Probability
有一组20个学生话费0~6不同的时间学习后去考试，预估学习的时间数对考试通过率的影响？  
训练数据：  

Hours | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 1.75 | 2.20 | 2.25 | 2.50 |
---|---|---|---|---|---|---|---|---|---|---|
Pass|0|0|0|0|0|0|1|1|0|1|0|1|0|


Hours | 2.75 | 3.00 | 3.25 | 3.50 | 4.00 | 4.25 | 4.50 | 4.75 | 5.00 | 5.50 |
---|---|---|---|---|---|---|---|---|---|---|
Pass|1|0|1|0|1|1|1|1|1|1|

期望函数 $t=wx+b$ (x为学习的时间)
概率函数 $p(y=1|x,w)=\frac{1}{1+e^-t}$  
初始化参数为b=1,w=-1,则初始期望函数为$t_0=1+x$  
循环执行3000次来进行递归递增：    
```python  
 using DataFrames;
 dtf = readtable("/Users/parag/Downloads/data.txt")
 b = fill(1, length(dtf[1]))
 X = hcat(b, dtf[1])
 Y = dtf[2]

function linearEq(theta, x)
    return theta'*x
end

function sigmoid(x)
    return 1/(1 + exp(-x))
end 

function init_theta()
	return [1.0; 1.0]
end

immutable EachRow{T<:AbstractMatrix}
    A::T
end

function gradient_assent(results, X)
	theta = init_theta()
	alpha = 0.0001
	for j=1:300000
		index = 1
		for i=1:size(X,1)
		x = X[i,:]
		y = results[index]
		index = index + 1
		eq = linearEq(theta, x)
		evaluated_y = sigmoid(eq)
		error = y - evaluated_y
		theta_index = 1
			for i=1:size(theta)[1]
				theta_j = theta[i,:]
				theta_j = theta_j + alpha*error*x[theta_index]
				theta[theta_index] = theta_j[1]
				theta_index = theta_index + 1
			end
		end
	end
	print(theta)
end
```

循环训练后得到的参数值为w=1.4738,b=-3.98244  
最后训练后的得到的期望概率函数为  

 $$
P(y=1|x)=\frac{1}{1+e^{1.474x-3.982}}
$$ 

用训练的模型计算通过率  
当学习时数为1，x=1时，考试通过率为0.082  

 $$
P(y=1|x=1)=\frac{1}{1+e^{-1.474*1-3.982}}=sigmoid(1.474*1-3.982))=0.082
$$ 

当学习时数为5，x=5时，考试通过率为0.967

 $$
P(y=1|x=1)=\frac{1}{1+e^{-1.474*1-3.982}}=sigmoid(1.474*1-3.982))=0.967
$$ 

### Programing
#### Tensorflow
- 001  

```python 
"""Simple tutorial using code from the TensorFlow example for Regression.
Parag K. Mital, Jan. 2016"""
# pip3 install --upgrade
# https://storage.googleapis.com/tensorflow/mac/tensorflow-0.6.0-py3-none-any.whl
# %%
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt


# %%
# get the classic mnist dataset
# one-hot means a sparse vector for every observation where only
# the class label is 1, and every other class is 0.
# more info here:
# https://www.tensorflow.org/versions/0.6.0/tutorials/mnist/download/index.html#dataset-object
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# %%
# mnist is now a DataSet with accessors for:
# 'train', 'test', and 'validation'.
# within each, we can access:
# images, labels, and num_examples
print(mnist.train.num_examples,
      mnist.test.num_examples,
      mnist.validation.num_examples)

# %% the images are stored as:
# n_observations x n_features tensor (n-dim array)
# the labels are stored as n_observations x n_labels,
# where each observation is a one-hot vector.
print(mnist.train.images.shape, mnist.train.labels.shape)

# %% the range of the values of the images is from 0-1
print(np.min(mnist.train.images), np.max(mnist.train.images))

# %% we can visualize any one of the images by reshaping it to a 28x28 image
plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')

# %% We can create a container for an input image using tensorflow's graph:
# We allow the first dimension to be None, since this will eventually
# represent our mini-batches, or how many images we feed into a network
# at a time during training/validation/testing.
# The second dimension is the number of features that the image has.
n_input = 784
n_output = 10
net_input = tf.placeholder(tf.float32, [None, n_input])

# %% We can write a simple regression (y = W*x + b) as:
W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))
net_output = tf.nn.softmax(tf.matmul(net_input, W) + b)

# %% We'll create a placeholder for the true output of the network
y_true = tf.placeholder(tf.float32, [None, 10])

# %% And then write our loss function:
cross_entropy = -tf.reduce_sum(y_true * tf.log(net_output))

# %% This would equate each label in our one-hot vector between the
# prediction and actual using the argmax as the predicted label
correct_prediction = tf.equal(
    tf.argmax(net_output, 1), tf.argmax(y_true, 1))

# %% And now we can look at the mean of our network's correct guesses
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# %% We can tell the tensorflow graph to train w/ gradient descent using
# our loss function and an input learning rate
optimizer = tf.train.GradientDescentOptimizer(
    0.01).minimize(cross_entropy)

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# %% Now actually do some training:
batch_size = 100
n_epochs = 10
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            net_input: batch_xs,
            y_true: batch_ys
        })
    print(sess.run(accuracy,
                   feed_dict={
                       net_input: mnist.validation.images,
                       y_true: mnist.validation.labels
                   }))

# %% Print final test accuracy:
print(sess.run(accuracy,
               feed_dict={
                   net_input: mnist.test.images,
                   y_true: mnist.test.labels
               }))

# %%
"""
# We could do the same thing w/ Keras like so:
from keras.models import Sequential
model = Sequential()
from keras.layers.core import Dense, Activation
model.add(Dense(output_dim=10, input_dim=784, init='zero'))
model.add(Activation("softmax"))
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', 
    optimizer=SGD(lr=learning_rate))
model.fit(mnist.train.images, mnist.train.labels, nb_epoch=n_epochs,
          batch_size=batch_size, show_accuracy=True)
objective_score = model.evaluate(mnist.test.images, mnist.test.labels,
                                 batch_size=100, show_accuracy=True)
"""
```


- 002  
  
```python 

#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
sess.run(predict_op, feed_dict={X: teX})))
```
