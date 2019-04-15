---
layout:     post
title:      "距离计算"
subtitle:   "Manhattan Distance & Minkowski Distance & Minkowski Distance"
date:       2017-10-15 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post04.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 机器学习
---
### Manhattan Distance *曼哈顿距离*
曼哈顿距离又称计程车几何距离或方格线距离，是由十九世纪的赫尔曼·闵可夫斯基所创词汇 ，为欧几里得几何度量空间的几何学之用语，用以标明两个点上在标准坐标系上的绝对轴距之总和。曼哈顿距离的正式意义为L1-距离或城市区块距离，也就是在欧几里得空间的固定直角坐标系上两点所形成的线段对轴产生的投影的距离总和。例如在平面上，坐标(x1,y1)的点P1与坐标(x2,y2)的点P2的曼哈顿距离为 

  $$
d=|x_1 -x_2| + |y_1 -y_2|
$$ 

- 表现形式 

 $$
d=\sum_{i=1}^n|x_i-y_i|
$$ 


### Euclidean Metric *欧式距离*
欧几里得度量（euclidean metric）也称欧氏距离，在数学中，欧几里得距离或欧几里得度量是欧几里得空间中两点间“普通”（即直线）距离。在欧几里得空间中，点x=(x1,x2,...,xn)和 y=(y1,y2,...,yn)之间的欧氏距离为:

 $$
d(x,y)=\sqrt{(x_1-y_1)^2 + (x_2-y_2)^2 + ... + (x_i-y_i)^2}
$$ 


- 表现形式

 $$
d=\sqrt{\sum_{i=1}^n(x_i-y_i)^2}
$$ 

### Minkowski Distance *闵可夫斯基距离*

闵可夫斯基距离或闵氏距离（Minkowski Distance）：以俄罗斯数学家闵可夫斯基命名的距离；是欧式距离的推广，闵氏距离不是一种距离，而是一组距离的定义 

- 表现形式

 $$
d=\sqrt[p]{\sum_{i=1}^n|x_i-y_i|^p} 
$$ 


从上面公式可以看出：
- 当p=1时，就是曼哈顿距离
- 当p=2时，就是欧氏距离
- 当p→∞时，就是切比雪夫距离
### Refrence 
>  https://www.iteblog.com/archives/2317.html