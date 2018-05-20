---
layout:     post
title:      "Spark Summary"
subtitle:   "Tutorials & some base examples"
date:       2018-05-20 12:00:00
author:     "YangLong"
header-img: "img/post-bg-nextgen-web-pwa.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Spark
    - Big Data
    - Machine Learn
---


## Introduction 
### Why use Spark
Spark是一个大规模数据处理的快速通用引擎
- **速度** Spark在内存中运行比Hadoop快100倍以上，在硬盘上运行比Hadoop快10倍以上 
- **便捷** Spark提供超过80+的高阶操作去建立并行的应用，可以使用Python、R、Scale等高阶语言进行交互
- **通用** Spark提供了一系列的类库，包括SQL、DataFrame、用于机器学习的MLlib，Spark Streaming，可以在同一个应用中使用这些类库
- **易用** Spark可以在Hadoop、Mesos的单实例或云上运行，也可以使用不同的数据源如HDFS、Cassandra、Hbase、S3 

![IMAGE](http://spark.apache.org/images/spark-stack.png)


### Tutorials
#### 下载安装
使用Python开发时，可以使用PIP进行安装
```python
pip install pysparkd.
```
#### 基础示例
##### 分词示例  
使用此示例生成（字符串，数量）结果，保存进文件
```python
text_file = sc.textFile("hdfs://...")
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
counts.saveAsTextFile("hdfs://...")
```

##### π计算
Spark也可用于数值的并行计算
```python
def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1

count = sc.parallelize(xrange(0, NUM_SAMPLES)) \
             .filter(inside).count()
print "Pi is roughly %f" % (4.0 * count / NUM_SAMPLES)
```

##### 文本查询 *[DataFrame]*
在日志文件中查询特定的错误信息
```python
extFile = sc.textFile("hdfs://...")

# Creates a DataFrame having a single column named "line"
df = textFile.map(lambda r: Row(r)).toDF(["line"])
errors = df.filter(col("line").like("%ERROR%"))
# Counts all the errors
errors.count()
# Counts errors mentioning MySQL
errors.filter(col("line").like("%MySQL%")).count()
# Fetches the MySQL errors as an array of strings
errors.filter(col("line").like("%MySQL%")).collect()
```
##### 数值操作 *[DataFrame]* 
从数据库中读取数据计算人员的平均年龄，把计算记过按照JSON格式存储在S3数据库中，数据库中存在表people，含有两列name，age
```python
# Creates a DataFrame based on a table named "people"
# stored in a MySQL database.
url = \
  "jdbc:mysql://yourIP:yourPort/test?user=yourUsername;password=yourPassword"
df = sqlContext \
  .read \
  .format("jdbc") \
  .option("url", url) \
  .option("dbtable", "people") \
  .load()

# Looks the schema of this DataFrame.
df.printSchema()

# Counts people by age
countsByAge = df.groupBy("age").count()
countsByAge.show()

# Saves countsByAge to S3 in the JSON format.
countsByAge.write.format("json").save("s3a://...")
```

##### 线性回归计算 *[Machine Learn]*  
在这个示例中，我们有特征向量和对应标签的数据集，使用线性回归用特征向量来预测标签
```python
# Every record of this DataFrame contains the label and
# features represented by a vector.
df = sqlContext.createDataFrame(data, ["label", "features"])

# Set parameters for the algorithm.
# Here, we limit the number of iterations to 10.
lr = LogisticRegression(maxIter=10)

# Fit the model to the data.
model = lr.fit(df)

# Given a dataset, predict each point's label, and show the results.
model.transform(df).show()

```
