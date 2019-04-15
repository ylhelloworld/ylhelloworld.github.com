---
layout:     post
title:      "自动解码"
subtitle:   "AutoEncoder"
date:       2017-10-02 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 机器学习
---
#### Summary
- 自动编码器基于这样一个事实：原始input（设为x）经过加权（W、b)、映射（Sigmoid）之后得到y，再对y反向加权映射回来成为z
- 通过反复迭代训练两组（W、b），使得误差函数最小，即尽可能保证z近似于x，即完美重构了x。
- 那么可以说正向第一组权（W、b）是成功的，很好的学习了input中的关键特征，不然也不会重构得如此完美
- The autoencoder tries to learn a function `$\textstyle h_{W,b}(x) \approx x$`

#### TensorFlow Demo

```python
#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
import tensorflow as tf 
import numpy as np 

def model(x, w1, w2, b1, b2): 
    a = tf.matmul(x, w1) 
    b = tf.add(a,b1) 
    c = tf.sigmoid(b) 
    hidden = tf.sigmoid(tf.add(tf.matmul(x, w1), b1)) 
    out = tf.nn.softmax(tf.add(tf.matmul(hidden, w2), b2)) 
    return out 

x = tf.placeholder("float", [4, 4]) 

w1 = tf.Variable(tf.random_normal([4,2]), name = 'w1') 
w2 = tf.Variable(tf.random_normal([2,4]), name = 'w2') 
b1 = tf.Variable(tf.random_normal([2]), name = 'b1') 
b2 = tf.Variable(tf.random_normal([4]), name = 'b2') 

pred = model(x, w1, w2, b1, b2) 
cost = tf.reduce_sum(tf.pow(tf.sub(pred, x), 2)) 
optimizer = tf.train.AdamOptimizer().minimize(cost) 

with tf.Session() as sess: 
    init = tf.initialize_all_variables() 
    sess.run(init) 

    input_data = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], float) 
    for i in xrange(10000): 
        sess.run(optimizer, feed_dict = {x: input_data}) 

    res = sess.run(pred, feed_dict = {x: input_data}) 
    index = np.argmax(res, 1) 
    for i in xrange(4): 
        tmp = np.zeros((4,)) 
        tmp[index[i]] = 1. 
        print res[i] 
        print tmp

```