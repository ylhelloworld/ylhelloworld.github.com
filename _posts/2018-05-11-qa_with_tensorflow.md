---
layout:     post
title:      "基于Tensorflow的问答系统"
subtitle:   "Question answer with tensorflow"
date:       2018-05-11 12:00:00
author:     "YangLong"
header-img: "img/spiderman/post08.jpg"
header-mask: 0.3
catalog:    true 
multilingual: false  
tags:
    - 我的翻译 
---
## Question answer with tensorflow  

QA 系统用于回答自然语言提出的问题，QA系统从文本和图像提取信息后去回答问题，这种系统主要被分成两块：**open domain**回答的答案不限定于某个领域，**close domain**回答的问题限定于某个领域如医学、网点常用服务内容。   
本文主要使用动态记忆伸进网络作为QA的算法，主要使用Tensorflow作为开发的框架。


#### 环境准备  
- 安装python3
- 安装Tensorflow 1.2+ 
- Jupyter 
- Numpy 
- Matplotlib  

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import urllib
import sys
import os
import zipfile
import tarfile
import json 
import hashlib
import re
import itertools
```


#### bAbi数据集  
本文主要使用Facebook的bAbi作为数据集，bAbi除了ixie问题比较复杂外其他还算比较明确，每一个问题都有和回答这个问题相关的详细内容去回答每个问题，当然每个问题还有正确的答案。

bAbi中的数据集根据不同的技术被分为20种Task，每种Task都有是被独立的训练和测试，这些Task含有各种自然语言处理能力，包括时序因果分析[#Task14]和归纳逻辑[#Task16]，为了更好的理解，可以参考下方这个例子 



task5 | task other
---|---
![image](https://d3ansictanv2wj.cloudfront.net/questionex-5a47c6bdd8e5dcc944460495c0d7eeb9.png) |![image](https://d3ansictanv2wj.cloudfront.net/questionex-5a47c6bdd8e5dcc944460495c0d7eeb9.png) 

task[#5]测试神经网络去理解三个事物之间的关系，概括来说，通过直接事物和简介的食物去识别，在这个例子中，最后能否识别哪个人从Jeff那里拿来牛奶，神经网络要从第五个句子中：Bill为主题、Jeff为间接事物，在第六句中：Jeff变成了主体，当然，网络完全不知道训练的主题和物体是什么，而是通过训练中的测试数据去推断理解。

另一个小问题是系统必须通过数据集去理解同义词，Jeff[拿]牛奶给Bill，对于系统来说[拿]和[给],[接]很相似，当然，系统无须从同开始做词区分，我们可以借助词向量，词向量中存储词的定义和词相近的词语，近义词有相近的向量，对于词词向量来说，我们将使用斯坦福的单词解释词向量Glove，后面会做单独介绍。

很多Task有限制上下文中必须包含确切的词做回答，在这个例子中，答案[Bill]可以从上下文中找到，我们最后的答案可以在上下文中使用这个是特殊的优势，在其他QA中答案可能不能在上线文中发现。
> 去下载要使用的安装包可能需要花费一些时间，在运行下列代码之前，须先下载bAbi和Glove，然后解压好用于我们的神经网络  


```lua
glove_zip_file = "glove.6B.zip"
glove_vectors_file = "glove.6B.50d.txt"

# 15 MB
data_set_zip = "tasks_1-20_v1-2.tar.gz"

#Select "task 5"
train_set_file = "qa5_three-arg-relations_train.txt"
test_set_file = "qa5_three-arg-relations_test.txt"

train_set_post_file = "tasks_1-20_v1-2/en/"+train_set_file
test_set_post_file = "tasks_1-20_v1-2/en/"+test_set_file
```

```python
glove_zip_file = "glove.6B.zip"
glove_vectors_file = "glove.6B.50d.txt"

# 15 MB
data_set_zip = "tasks_1-20_v1-2.tar.gz"

#Select "task 5"
train_set_file = "qa5_three-arg-relations_train.txt"
test_set_file = "qa5_three-arg-relations_test.txt"

train_set_post_file = "tasks_1-20_v1-2/en/"+train_set_file
test_set_post_file = "tasks_1-20_v1-2/en/"+test_set_file
```

```python
def unzip_single_file(zip_file_name, output_file_name):
    """
        If the output file is already created, don't recreate
        If the output file does not exist, create it from the zipFile
    """
    if not os.path.isfile(output_file_name):
        with open(output_file_name, 'wb') as out_file:
            with zipfile.ZipFile(zip_file_name) as zipped:
                for info in zipped.infolist():
                    if output_file_name in info.filename:
                        with zipped.open(info) as requested_file:
                            out_file.write(requested_file.read())
                            return
def targz_unzip_single_file(zip_file_name, output_file_name, interior_relative_path):
    if not os.path.isfile(output_file_name):
        with tarfile.open(zip_file_name) as un_zipped:
            un_zipped.extract(interior_relative_path+output_file_name)    
unzip_single_file(glove_zip_file, glove_vectors_file)
targz_unzip_single_file(data_set_zip, train_set_file, "tasks_1-20_v1-2/en/")
targz_unzip_single_file(data_set_zip, test_set_file, "tasks_1-20_v1-2/en/")

```

#### 分析Glove & 操作未知特征   
将根据GloVe定义的映射来讨论这是一个将字符串转换为矩阵的函数。这个函数将字符串分成标记，这些标记大致相当于标点符号，单词或部分单词。例如，在“Bill  travled to the kitchen"中，有六个令牌：五个对应于每个单词，最后一个对应于最后一个时间点。每个令牌都会单独进行矢量化，从而生成与每个句子对应的矩阵，如下图将句子转换为矩阵的过程：

![image](https://d3ansictanv2wj.cloudfront.net/vectorize-e32f76ee1e29838979e0041428e9588a.png)

在bAbI中的某些任务中，系统将遇到不存在GloVe单词。为了网络能够处理这些未知的单词，我们需要保持这些单词的一致矢量化。通常的做法是用单个<UNK>矢量替换所有未知的标记，但是这样效果不太好。根据经验，我们可以随机为未知的单词生成向量。

当我们遇到一个新的单词时，我们只需从原始Glove矢量化的（高斯近似）分布中绘制一个新的矢量化，并将该向量添加回GloVe单词图。为了收集分布超参数，Numpy具有自动计算方差和均值的功能。如下函数fill_unk 为新遇到的单词分配一个新的向量
```python
# Deserialize GloVe vectors
glove_wordmap = {}
with open(glove_vectors_file, "r", encoding="utf8") as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")
```
```python
wvecs = []
for item in glove_wordmap.items():
    wvecs.append(item[1])
s = np.vstack(wvecs)

# Gather the distribution hyperparameters
v = np.var(s,0) 
m = np.mean(s,0) 
RS = np.random.RandomState()

def fill_unk(unk):
    global glove_wordmap
    glove_wordmap[unk] = RS.multivariate_normal(m,np.diag(v))
    return glove_wordmap[unk]
```



#### 已知&未知词语处理  
因为bAbI的词汇量有限，意味着即使不知道单词的含义，网络也可以学习单词之间的关系。然而，为了学习的速度，我们尽量选择单词已有的向量。为此，在Glove中进行全局搜索，如果单词不存在，创建一个新的向量来表示这个单词。
我们定义一个新的函数```sentence2sequence```来向量化单词：
```python
def sentence2sequence(sentence):
    """

    - Turns an input paragraph into an (m,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.

      TensorFlow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our 
      mapping does not need the computational power that
      TensorFlow provides. Normal Python suffices for this task.
    """
    tokens = sentence.strip('"(),-').lower().split(" ")
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
                continue
            else:
                i = i-1
            if i == 0:
                # word OOV
                # https://arxiv.org/pdf/1611.01436.pdf
                rows.append(fill_unk(token))
                words.append(token)
                break
    return np.array(rows), words

```
现在我们可以将每个问题所需的所有数据打包在一起，包括上下文的矢量化，问题和答案。在bAbI中，上下文由编号的句子序列定义，contextualize反序列化成与一个上下文关联的句子列表。问题和答案位于同一行，由制表符分隔，因此我们可以使用制表符作为特定行是否引用问题的标记。当编号重置时，未来的问题将引用新的上下文（注意，通常有多个问题对应于单个上下文）。答案还包含我们保留但不需要使用的另一条信息：与参考顺序中回答问题所需的句子相对应的数字。在我们的系统中，网络会自学自己需要哪些句子来回答问题。

```python
def contextualize(set_file):
    """
    Read in the dataset of questions and build question+answer -> context sets.
    Output is a list of data points, each of which is a 7-element tuple containing:
        The sentences in the context in vectorized form.
        The sentences in the context as a list of string tokens.
        The question in vectorized form.
        The question as a list of string tokens.
        The answer in vectorized form.
        The answer as a list of string tokens.
        A list of numbers for supporting statements, which is currently unused.
    """
    data = []
    context = []
    with open(set_file, "r", encoding="utf8") as train:
        for line in train:
            l, ine = tuple(line.split(" ", 1))
            # Split the line numbers from the sentences they refer to.
            if l is "1":
                # New contexts always start with 1, 
                # so this is a signal to reset the context.
                context = []
            if "\t" in ine: 
                # Tabs are the separator between questions and answers,
                # and are not present in context statements.
                question, answer, support = tuple(ine.split("\t"))
                data.append((tuple(zip(*context))+
                             sentence2sequence(question)+
                             sentence2sequence(answer)+
                             ([int(s) for s in support.split()],)))
                # Multiple questions may refer to the same context, so we don't reset it.
            else:
                # Context sentence.
                context.append(sentence2sequence(ine[:-1]))
    return data
train_data = contextualize(train_set_post_file)
test_data = contextualize(test_set_post_file)
```

```python
final_train_data = []
def finalize(data):
    """
    Prepares data generated by contextualize() for use in the network.
    """
    final_data = []
    for cqas in train_data:
        contextvs, contextws, qvs, qws, avs, aws, spt = cqas

        lengths = itertools.accumulate(len(cvec) for cvec in contextvs)
        context_vec = np.concatenate(contextvs)
        context_words = sum(contextws,[])

        # Location markers for the beginnings of new sentences.
        sentence_ends = np.array(list(lengths)) 
        final_data.append((context_vec, sentence_ends, qvs, spt, context_words, cqas, avs, aws))
    return np.array(final_data)
final_train_data = finalize(train_data)   
final_test_data = finalize(test_data)
```



####  定义超参数   

此时，我们已经准备好了我们的培训数据和我们的测试数据。接下来的任务是构建我们将用来理解数据的网络。我们首先清除TensorFlow默认图表，这样我们就可以再次运行网络，如果我们想要改变某些东西的话。

```python
tf.reset_default_graph()
```  
由于这是实际网络的开始，我们还要定义网络所需的所有常量。我们称之为“超参数”，因为它们定义了网络的外观和训练方式：
```python
# Hyperparameters

# The number of dimensions used to store data passed between recurrent layers in the network.
recurrent_cell_size = 128

# The number of dimensions in our word vectorizations.
D = 50 

# How quickly the network learns. Too high, and we may run into numeric instability 
# or other issues.
learning_rate = 0.005

# Dropout probabilities. For a description of dropout and what these probabilities are, 
# see Entailment with TensorFlow.
input_p, output_p = 0.5, 0.5

# How many questions we train on at a time.
batch_size = 128

# Number of passes in episodic memory. We'll get to this later.
passes = 4

# Feed Forward layer sizes: the number of dimensions used to store data passed from feed-forward layers.
ff_hidden_size = 256

weight_decay = 0.00000001
# The strength of our regularization. Increase to encourage sparsity in episodic memory, 
# but makes training slower. Don't make this larger than leraning_rate.

training_iterations_count = 400000
# How many questions the network trains on each time it is trained. 
# Some questions are counted multiple times.

display_step = 100
# How many iterations of training occur before each validation check.
```

#### 神经网络结构  
通过超参数，我们来描述网络结构。这个网络的结构被松散地分成四个模块，在Ask Me Anything：Dynamic Memory Networks for Natural Language Processing中有描述。

该网络的设计基于文本中的其他信息动态设置循环图层的内存，因此名称为动态内存网络（DMN）。DMN松散地基于对人类如何试图回答阅读理解型问题的理解。首先，这个人有机会阅读背景并创造内部事实的记忆。考虑到这些事实，他们然后阅读这个问题，重新审视具体寻找问题答案的背景，将问题与每个事实进行比较。

有时，一个事实引导我们到另一个事实。在bAbI数据集中，网络可能想要查找足球的位置。它可能会搜索有关足球的句子，发现约翰是最后一个接触足球的人，然后搜索关于约翰的句子，以发现约翰曾在卧室和走廊里。一旦它意识到John已经走到了走廊的最后，它就可以回答这个问题并自信地说足球在走廊里。

![image](https://d3ansictanv2wj.cloudfront.net/model-7b6a1ecc9d23af0100f29edb7930d3bf.png)   


##### 输入  
输入模块是动态存储器网络用来提出答案的四个模块中的第一个模块，它包含一个简单的输入端和门控循环单元（GRU）（TensorFlow的tf.contrib.nn.GRUCell）收集证据。每条证据或事实对应于上下文中的单个句子，并由该时间步的输出表示。这需要一些非TensorFlow预处理，因此我们可以收集句子末尾的位置并将其传递到TensorFlow中，以便在以后的模块中使用。

当我们接受培训时，我们会在稍后处理这些外部处理。我们可以使用TensorFlow的处理数据gather_nd来选择相应的输出。该函数gather_nd是一个非常有用的工具，我建议您查看API文档以了解它的工作原理。

```python
# Input Module

# Context: A [batch_size, maximum_context_length, word_vectorization_dimensions] tensor 
# that contains all the context information.
context = tf.placeholder(tf.float32, [None, None, D], "context")  
context_placeholder = context # I use context as a variable name later on

# input_sentence_endings: A [batch_size, maximum_sentence_count, 2] tensor that 
# contains the locations of the ends of sentences. 
input_sentence_endings = tf.placeholder(tf.int32, [None, None, 2], "sentence")

# recurrent_cell_size: the number of hidden units in recurrent layers.
input_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

# input_p: The probability of maintaining a specific hidden input unit.
# Likewise, output_p is the probability of maintaining a specific hidden output unit.
gru_drop = tf.contrib.rnn.DropoutWrapper(input_gru, input_p, output_p)

# dynamic_rnn also returns the final internal state. We don't need that, and can
# ignore the corresponding output (_). 
input_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, context, dtype=tf.float32, scope = "input_module")

# cs: the facts gathered from the context.
cs = tf.gather_nd(input_module_outputs, input_sentence_endings)
# to use every word as a fact, useful for tasks with one-sentence contexts
s = input_module_outputs
```

##### 问题  
问题模块是第二个模块，可以说是最简单的。它由另一个GRU通过，这次是问题文本。我们可以简单地传递最终状态，而不是一些证据，因为问题由数据集保证为一个句子长。
```python
# Question Module

# query: A [batch_size, maximum_question_length, word_vectorization_dimensions] tensor 
#  that contains all of the questions.

query = tf.placeholder(tf.float32, [None, None, D], "query")

# input_query_lengths: A [batch_size, 2] tensor that contains question length information. 
# input_query_lengths[:,1] has the actual lengths; input_query_lengths[:,0] is a simple range() 
# so that it plays nice with gather_nd.
input_query_lengths = tf.placeholder(tf.int32, [None, 2], "query_lengths")

question_module_outputs, _ = tf.nn.dynamic_rnn(gru_drop, query, dtype=tf.float32, 
                                               scope = tf.VariableScope(True, "input_module"))

# q: the question states. A [batch_size, recurrent_cell_size] tensor.
q = tf.gather_nd(question_module_outputs, input_query_lengths)

```
#####  情节记忆  
我们的第三个模块，情景记忆模块，是事情开始变得有趣的地方。它使用注意力来执行多次传递，每次传递由GRU组成，迭代输入。根据当时对相应事实的关注程度，每次传递中的每次迭代都会对当前内存进行加权更新。

神经网络中的注意力最初是为图像分析而设计的，特别是对于图像部分比其他图像更相关的情况。网络使用注意力来确定执行任务时进行进一步分析的最佳位置，例如查找图像中对象的位置，跟踪在图像间移动的对象，面部识别或从查找最相关信息图像中的任务。

主要的问题是注意力，或者至少是注意力集中在一个输入位置，不容易优化。与大多数其他神经网络一样，我们的优化方案是根据输入和权重计算损失函数的导数，由于其二元性质，硬性注意力根本不可区分。相反，我们不得不使用被称为“软性关注”的实值版本，该版本将所有可能使用的输入位置与使用某种形式的权重相结合。值得庆幸的是，权重是完全可以区分的，并且可以正常训练。虽然可以学习注意力，但它比软性注意要困难得多，有时表现得更差。因此，我们将坚持这种模式的软性关注。不要担心编码的衍生物; TensorFlow”

我们通过在每个事实，我们当前的记忆和原始问题之间构建相似度量来计算这个模型中的注意力。（请注意，这与正常关注不同，后者仅构建事实与当前记忆之间的相似性度量）。我们将结果通过双层前馈网络传递给每个事实的注意力常量。然后，我们通过在输入事实上使用GRU进行加权传递来修改记忆（由相应的注意力常数加权）。为了避免当上下文短于矩阵的全部长度时将不正确的信息添加到内存中，我们创建了一个掩码，其中存在事实并且根本不参加（即保留相同的内存）存在。

另一个值得注意的方面是关注蒙版几乎总是围绕图层使用的表示。对于图像来说，这个包裹最可能发生在卷积层（很可能是直接映射到图像中的位置），对于自然语言而言，包裹最可能发生在复发层周围。在技术上可能的情况下，在前馈层周围注意围绕前馈层通常是没有用的 - 至少不能通过其他前馈层更难以模拟的方式。

```python
# Episodic Memory

# make sure the current memory (i.e. the question vector) is broadcasted along the facts dimension
size = tf.stack([tf.constant(1),tf.shape(cs)[1], tf.constant(1)])
re_q = tf.tile(tf.reshape(q,[-1,1,recurrent_cell_size]),size)


# Final output for attention, needs to be 1 in order to create a mask
output_size = 1 

# Weights and biases
attend_init = tf.random_normal_initializer(stddev=0.1)
w_1 = tf.get_variable("attend_w1", [1,recurrent_cell_size*7, recurrent_cell_size], 
                      tf.float32, initializer = attend_init)
w_2 = tf.get_variable("attend_w2", [1,recurrent_cell_size, output_size], 
                      tf.float32, initializer = attend_init)

b_1 = tf.get_variable("attend_b1", [1, recurrent_cell_size], 
                      tf.float32, initializer = attend_init)
b_2 = tf.get_variable("attend_b2", [1, output_size], 
                      tf.float32, initializer = attend_init)

# Regulate all the weights and biases
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_1))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_1))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w_2))
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(b_2))

def attention(c, mem, existing_facts):
    """
    Custom attention mechanism.
    c: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor 
        that contains all the facts from the contexts.
    mem: A [batch_size, maximum_sentence_count, recurrent_cell_size] tensor that 
        contains the current memory. It should be the same memory for all facts for accurate results.
    existing_facts: A [batch_size, maximum_sentence_count, 1] tensor that 
        acts as a binary mask for which facts exist and which do not.

    """
    with tf.variable_scope("attending") as scope:
        # attending: The metrics by which we decide what to attend to.
        attending = tf.concat([c, mem, re_q, c * re_q,  c * mem, (c-re_q)**2, (c-mem)**2], 2)

        # m1: First layer of multiplied weights for the feed-forward network. 
        #     We tile the weights in order to manually broadcast, since tf.matmul does not
        #     automatically broadcast batch matrix multiplication as of TensorFlow 1.2.
        m1 = tf.matmul(attending * existing_facts, 
                       tf.tile(w_1, tf.stack([tf.shape(attending)[0],1,1]))) * existing_facts
        # bias_1: A masked version of the first feed-forward layer's bias
        #     over only existing facts.

        bias_1 = b_1 * existing_facts

        # tnhan: First nonlinearity. In the original paper, this is a tanh nonlinearity; 
        #        choosing relu was a design choice intended to avoid issues with 
        #        low gradient magnitude when the tanh returned values close to 1 or -1. 
        tnhan = tf.nn.relu(m1 + bias_1)

        # m2: Second layer of multiplied weights for the feed-forward network. 
        #     Still tiling weights for the same reason described in m1's comments.
        m2 = tf.matmul(tnhan, tf.tile(w_2, tf.stack([tf.shape(attending)[0],1,1])))

        # bias_2: A masked version of the second feed-forward layer's bias.
        bias_2 = b_2 * existing_facts

        # norm_m2: A normalized version of the second layer of weights, which is used 
        #     to help make sure the softmax nonlinearity doesn't saturate.
        norm_m2 = tf.nn.l2_normalize(m2 + bias_2, -1)

        # softmaxable: A hack in order to use sparse_softmax on an otherwise dense tensor. 
        #     We make norm_m2 a sparse tensor, then make it dense again after the operation.
        softmax_idx = tf.where(tf.not_equal(norm_m2, 0))[:,:-1]
        softmax_gather = tf.gather_nd(norm_m2[...,0], softmax_idx)
        softmax_shape = tf.shape(norm_m2, out_type=tf.int64)[:-1]
        softmaxable = tf.SparseTensor(softmax_idx, softmax_gather, softmax_shape)
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_softmax(softmaxable)),-1)

# facts_0s: a [batch_size, max_facts_length, 1] tensor 
#     whose values are 1 if the corresponding fact exists and 0 if not.
facts_0s = tf.cast(tf.count_nonzero(input_sentence_endings[:,:,-1:],-1,keep_dims=True),tf.float32)


with tf.variable_scope("Episodes") as scope:
    attention_gru = tf.contrib.rnn.GRUCell(recurrent_cell_size)

    # memory: A list of all tensors that are the (current or past) memory state 
    #   of the attention mechanism.
    memory = [q]

    # attends: A list of all tensors that represent what the network attends to.
    attends = []
    for a in range(passes):
        # attention mask
        attend_to = attention(cs, tf.tile(tf.reshape(memory[-1],[-1,1,recurrent_cell_size]),size),
                              facts_0s)

        # Inverse attention mask, for what's retained in the state.
        retain = 1-attend_to

        # GRU pass over the facts, according to the attention mask.
        while_valid_index = (lambda state, index: index < tf.shape(cs)[1])
        update_state = (lambda state, index: (attend_to[:,index,:] * 
                                                 attention_gru(cs[:,index,:], state)[0] + 
                                                 retain[:,index,:] * state))
        # start loop with most recent memory and at the first index
        memory.append(tuple(tf.while_loop(while_valid_index,
                          (lambda state, index: (update_state(state,index),index+1)),
                           loop_vars = [memory[-1], 0]))[0]) 

        attends.append(attend_to)

        # Reuse variables so the GRU pass uses the same variables every pass.
        scope.reuse_variables()

```



##### 回答  
最后一个模块是答案模块，它从问题和情节记忆模块的输出中使用完全连接的层到“最终结果”单词向量，并且距离结果最近的单词是我们的最终结果输出（以保证结果是一个实际的单词）。我们通过为每个单词创建一个“分数”来计算最接近的单词，它表示最终结果与单词的距离。虽然您可以设计可以返回多个单词的答案模块，但本文中尝试的bAbI任务不需要这些答案。

```python
# Answer Module

# a0: Final memory state. (Input to answer module)
a0 = tf.concat([memory[-1], q], -1)

# fc_init: Initializer for the final fully connected layer's weights.
fc_init = tf.random_normal_initializer(stddev=0.1) 

with tf.variable_scope("answer"):
    # w_answer: The final fully connected layer's weights.
    w_answer = tf.get_variable("weight", [recurrent_cell_size*2, D], 
                               tf.float32, initializer = fc_init)
    # Regulate the fully connected layer's weights
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 
                     tf.nn.l2_loss(w_answer)) 

    # The regressed word. This isn't an actual word yet; 
    #    we still have to find the closest match.
    logit = tf.expand_dims(tf.matmul(a0, w_answer),1)

    # Make a mask over which words exist.
    with tf.variable_scope("ending"):
        all_ends = tf.reshape(input_sentence_endings, [-1,2])
        range_ends = tf.range(tf.shape(all_ends)[0])
        ends_indices = tf.stack([all_ends[:,0],range_ends], axis=1)
        ind = tf.reduce_max(tf.scatter_nd(ends_indices, all_ends[:,1],
                                          [tf.shape(q)[0], tf.shape(all_ends)[0]]),
                            axis=-1)
        range_ind = tf.range(tf.shape(ind)[0])
        mask_ends = tf.cast(tf.scatter_nd(tf.stack([ind, range_ind], axis=1), 
                                          tf.ones_like(range_ind), [tf.reduce_max(ind)+1, 
                                                                    tf.shape(ind)[0]]), bool)
        # A bit of a trick. With the locations of the ends of the mask (the last periods in 
        #  each of the contexts) as 1 and the rest as 0, we can scan with exclusive or 
        #  (starting from all 1). For each context in the batch, this will result in 1s 
        #  up until the marker (the location of that last period) and 0s afterwards.
        mask = tf.scan(tf.logical_xor,mask_ends, tf.ones_like(range_ind, dtype=bool))

    # We score each possible word inversely with their Euclidean distance to the regressed word.
    #  The highest score (lowest distance) will correspond to the selected word.
    logits = -tf.reduce_sum(tf.square(context*tf.transpose(tf.expand_dims(
                    tf.cast(mask, tf.float32),-1),[1,0,2]) - logit), axis=-1)
```
#### 优化  
渐变下降是神经网络的默认优化器。其目标是减少网络的“损失”，这是衡量网络性能差的一个指标。它通过找到当前输入下每个权重的损失导数，然后“下降”权重来减少损失。大多数情况下，这样做的效果不错，但通常情况并不理想。有多种方案使用“动量”或其他更直接路径的近似值来达到最佳权重。这些优化方案中最有用的一种称为自适应矩估计，即Adam。

亚当通过计算过去迭代的梯度和平方梯度的指数衰减平均值来估计梯度的前两个时刻，其对应于这些梯度的估计平均值和估计方差。计算使用两个额外的超参数来指示平均值随着新信息的增加而衰减的速度。平均值初始化为零，这导致偏向零，特别是当那些超参数接近一时。

为了抵消这种偏见，亚当计算偏差纠正的时间估计值，其幅度大于原始值。校正后的估计值用于更新整个网络的权重。这些估算的结合使Adam成为整体优化的最佳选择之一，尤其是对于复杂网络。这适用于非常稀疏的数据，例如自然语言处理任务中常见的数据。

在TensorFlow中，我们可以使用Adam创建一个tf.train.AdamOptimizer。

```python
# Training

# gold_standard: The real answers.
gold_standard = tf.placeholder(tf.float32, [None, 1, D], "answer")
with tf.variable_scope('accuracy'):
    eq = tf.equal(context, gold_standard)
    corrbool = tf.reduce_all(eq,-1)
    logloc = tf.reduce_max(logits, -1, keep_dims = True)
    # locs: A boolean tensor that indicates where the score 
    #  matches the minimum score. This happens on multiple dimensions, 
    #  so in the off chance there's one or two indexes that match 
    #  we make sure it matches in all indexes.
    locs = tf.equal(logits, logloc)

    # correctsbool: A boolean tensor that indicates for which 
    #   words in the context the score always matches the minimum score.
    correctsbool = tf.reduce_any(tf.logical_and(locs, corrbool), -1)
    # corrects: A tensor that is simply correctsbool cast to floats.
    corrects = tf.where(correctsbool, tf.ones_like(correctsbool, dtype=tf.float32), 
                        tf.zeros_like(correctsbool,dtype=tf.float32))

    # corr: corrects, but for the right answer instead of our selected answer.
    corr = tf.where(corrbool, tf.ones_like(corrbool, dtype=tf.float32), 
                        tf.zeros_like(corrbool,dtype=tf.float32))
with tf.variable_scope("loss"):
    # Use sigmoid cross entropy as the base loss, 
    #  with our distances as the relative probabilities. There are
    #  multiple correct labels, for each location of the answer word within the context.
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = tf.nn.l2_normalize(logits,-1),
                                                   labels = corr)

    # Add regularization losses, weighted by weight_decay.
    total_loss = tf.reduce_mean(loss) + weight_decay * tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

# TensorFlow's default implementation of the Adam optimizer works. We can adjust more than 
#  just the learning rate, but it's not necessary to find a very good optimum.
optimizer = tf.train.AdamOptimizer(learning_rate)

# Once we have an optimizer, we ask it to minimize the loss 
#   in order to work towards the proper training.
opt_op = optimizer.minimize(total_loss)

```
```python
# Initialize variables
init = tf.global_variables_initializer()

# Launch the TensorFlow session
sess = tf.Session()
sess.run(init)
```
#### 训练网络  
随着一切准备就绪，我们可以开始批量培训数据来训练我们的网络！在系统正在接受培训的同时，我们应该检查网络在准确性方面的表现。我们使用验证集来完成此操作，该验证集取自测试数据，因此与训练数据没有重叠。

使用基于测试数据的验证集可以让我们更好地了解网络如何将其所学知识普遍化并将其应用于其他环境。如果我们对训练数据进行验证，网络可能会过度训练 - 换句话说，学习具体的例子并记住答案，这对网络回答新问题并无帮助。

如果您安装了TQDM，您可以使用它来记录网络已经训练了多长时间并接收训练结束时间的估计。通过中断Jupyter Notebook内核，您可以随时停止培训

```python 
def prep_batch(batch_data, more_data = False):
    """
        Prepare all the preproccessing that needs to be done on a batch-by-batch basis.
    """
    context_vec, sentence_ends, questionvs, spt, context_words, cqas, answervs, _ = zip(*batch_data)
    ends = list(sentence_ends)
    maxend = max(map(len, ends))
    aends = np.zeros((len(ends), maxend))
    for index, i in enumerate(ends):
        for indexj, x in enumerate(i):
            aends[index, indexj] = x-1
    new_ends = np.zeros(aends.shape+(2,))

    for index, x in np.ndenumerate(aends):
        new_ends[index+(0,)] = index[0]
        new_ends[index+(1,)] = x

    contexts = list(context_vec)
    max_context_length = max([len(x) for x in contexts])
    contextsize = list(np.array(contexts[0]).shape)
    contextsize[0] = max_context_length
    final_contexts = np.zeros([len(contexts)]+contextsize)

    contexts = [np.array(x) for x in contexts]
    for i, context in enumerate(contexts):
        final_contexts[i,0:len(context),:] = context
    max_query_length = max(len(x) for x in questionvs)
    querysize = list(np.array(questionvs[0]).shape)
    querysize[:1] = [len(questionvs),max_query_length]
    queries = np.zeros(querysize)
    querylengths = np.array(list(zip(range(len(questionvs)),[len(q)-1 for q in questionvs])))
    questions = [np.array(q) for q in questionvs]
    for i, question in enumerate(questions):
        queries[i,0:len(question),:] = question
    data = {context_placeholder: final_contexts, input_sentence_endings: new_ends, 
                            query:queries, input_query_lengths:querylengths, gold_standard: answervs}
    return (data, context_words, cqas) if more_data else data

```

```python
# Use TQDM if installed
tqdm_installed = False
try:
    from tqdm import tqdm
    tqdm_installed = True
except:
    pass


# Prepare validation set
batch = np.random.randint(final_test_data.shape[0], size=batch_size*10)
batch_data = final_test_data[batch]

validation_set, val_context_words, val_cqas = prep_batch(batch_data, True)

# training_iterations_count: The number of data pieces to train on in total
# batch_size: The number of data pieces per batch
def train(iterations, batch_size):
    training_iterations = range(0,iterations,batch_size)
    if tqdm_installed:
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)

    wordz = []
    for j in training_iterations:

        batch = np.random.randint(final_train_data.shape[0], size=batch_size)
        batch_data = final_train_data[batch]

        sess.run([opt_op], feed_dict=prep_batch(batch_data))
        if (j/batch_size) % display_step == 0:

            # Calculate batch accuracy
            acc, ccs, tmp_loss, log, con, cor, loc  = sess.run([corrects, cs, total_loss, logit,
                                                                context_placeholder,corr, locs], 
                                                               feed_dict=validation_set)
            # Display results
            print("Iter " + str(j/batch_size) + ", Minibatch Loss= ",tmp_loss,
                  "Accuracy= ", np.mean(acc))
train(30000,batch_size) # Small amount of training for preliminary results
```
经过一点训练后，让我们窥视一下，看看我们从网络中得到了什么样的答案。在下面的图表中，我们可以看到关于我们上下文中所有句子（列）的每一集（行）的注意力; 较暗的颜色代表该集中对该句子的更多关注。

至少在每个问题的两集之间你应该看到注意力的变化，但是有时候注意力能够在一个问题中找到答案，有时它会占用所有四集。如果注意力似乎是空白的，它可能会饱和并且立即关注所有事情。在这种情况下，你可以尝试更高的训练，weight_decay以防止发生这种情况。在培训后期，饱和度变得非常普遍。

```python
ancr = sess.run([corrbool,locs, total_loss, logits, facts_0s, w_1]+attends+
                [query, cs, question_module_outputs],feed_dict=validation_set)
a = ancr[0]
n = ancr[1]
cr = ancr[2]
attenders = np.array(ancr[6:-3]) 
faq = np.sum(ancr[4], axis=(-1,-2)) # Number of facts in each context

limit = 5
for question in range(min(limit, batch_size)):
    plt.yticks(range(passes,0,-1))
    plt.ylabel("Episode")
    plt.xlabel("Question "+str(question+1))
    pltdata = attenders[:,question,:int(faq[question]),0] 
    # Display only information about facts that actually exist, all others are 0
    pltdata = (pltdata - pltdata.mean()) / ((pltdata.max() - pltdata.min() + 0.001)) * 256
    plt.pcolor(pltdata, cmap=plt.cm.BuGn, alpha=0.7)
    plt.show()

#print(list(map((lambda x: x.shape),ancr[3:])), new_ends.shape)
```

col1 | col2
---|---
![image](https://d3ansictanv2wj.cloudfront.net/output_33_0-b584d319ba66250ce3349bdd6de8eca6.png)| ![image](https://d3ansictanv2wj.cloudfront.net/output_33_1-4b2d22984bd0a72b84af367bf3622319.png) 
![image](https://d3ansictanv2wj.cloudfront.net/output_33_2-53fc7e40413b86e2ca57d560d9f7ef0b.png) | ![image](https://d3ansictanv2wj.cloudfront.net/output_33_3-02a5508a2bc4bdcb933377168ae3ae90.png)  
 ![image](https://d3ansictanv2wj.cloudfront.net/output_33_4-5a3a9c768bf4723d579ba6a2aa4a4a44.png) | 
 
 

为了看到上述问题的答案是什么，我们可以在上下文中使用我们的距离分数的位置作为索引，并查看该索引处的单词。
```python
# Locations of responses within contexts
indices = np.argmax(n,axis=1)

# Locations of actual answers within contexts 
indicesc = np.argmax(a,axis=1)

for i,e,cw, cqa in list(zip(indices, indicesc, val_context_words, val_cqas))[:limit]:
    ccc = " ".join(cw)
    print("TEXT: ",ccc)
    print ("QUESTION: ", " ".join(cqa[3]))
    print ("RESPONSE: ", cw[i], ["Correct", "Incorrect"][i!=e])
    print("EXPECTED: ", cw[e])
    print()
```

```python
 TEXT:  mary travelled to the bedroom . mary journeyed to the bathroom . mary got the football there . mary passed the football to fred .
    QUESTION:  who received the football ?
    RESPONSE:  mary Incorrect
    EXPECTED:  fred

    TEXT:  bill grabbed the apple there . bill got the football there . jeff journeyed to the bathroom . bill handed the apple to jeff . jeff handed the apple to bill . bill handed the apple to jeff . jeff handed the apple to bill . bill handed the apple to jeff .
    QUESTION:  what did bill give to jeff ?
    RESPONSE:  apple Correct
    EXPECTED:  apple

    TEXT:  bill moved to the bathroom . mary went to the garden . mary picked up the apple there . bill moved to the kitchen . mary left the apple there . jeff got the football there . jeff went back to the kitchen . jeff gave the football to fred .
    QUESTION:  what did jeff give to fred ?
    RESPONSE:  apple Incorrect
    EXPECTED:  football

    TEXT:  jeff travelled to the bathroom . bill journeyed to the bedroom . jeff journeyed to the hallway . bill took the milk there . bill discarded the milk . mary moved to the bedroom . jeff went back to the bedroom . fred got the football there . bill grabbed the milk there . bill passed the milk to mary . mary gave the milk to bill . bill discarded the milk there . bill went to the kitchen . bill got the apple there .
    QUESTION:  who gave the milk to bill ?
    RESPONSE:  jeff Incorrect
    EXPECTED:  mary

    TEXT:  fred travelled to the bathroom . jeff went to the bathroom . mary went back to the bathroom . fred went back to the bedroom . fred moved to the office . mary went back to the bedroom . jeff got the milk there . bill journeyed to the garden . mary went back to the kitchen . fred went to the bedroom . mary journeyed to the bedroom . jeff put down the milk there . jeff picked up the milk there . bill went back to the office . mary went to the kitchen . jeff went back to the kitchen . jeff passed the milk to mary . mary gave the milk to jeff . jeff gave the milk to mary . mary got the football there . bill travelled to the bathroom . fred moved to the garden . fred got the apple there . mary handed the football to jeff . fred put down the apple . jeff left the football .
    QUESTION:  who received the football ?
    RESPONSE:  mary Incorrect
    EXPECTED:  jeff
```  
让我们继续训练！为了获得良好的效果，您可能需要长时间训练（在我的家用台式机上，大约需要12个小时），但您最终应该能够达到非常高的精度（超过90％）。有经验的Jupyter Notebook用户应该注意，只要你保持不变，你可以在任何时候中断培训并仍然保存网络已经取得的进展tf.Session; 如果您想要显示注意力并回答网络当前的提示，这很有用。

```python
train(training_iterations_count, batch_size)
```
```python
# Final testing accuracy
print(np.mean(sess.run([corrects], feed_dict= prep_batch(final_test_data))[0]))
```
```python
0.95
```
一旦我们完成了查看模型返回的内容，我们可以关闭会话以释放系统资源。

```python
sess.close()
```

#### 优化方向  

- **其他任务在bAbI** 我们只抽样了bAbI必须提供的许多任务。尝试更改预处理以适应其他任务，并查看我们的动态内存网络如何执行。当然，在尝试在新任务上运行网络之前，您可能想重新训练网络。如果任务不能保证答案出现在上下文中，则可能需要将输出与单词词典及其相应向量进行比较。（这些任务是6-10和17-20）。我建议尝试任务1或3，你可以通过改变值做test_set_file和train_set_file。

- **监督培训** 我们的关注机制是无人监督的，因为我们没有明确地指定应该关注哪些句子，而是让网络自己决定。尝试向网络添加损失，鼓励注意机制注意正确的句子

- **Coattention** 一些研究人员不是简单地参与输入句子，而是在他们称之为“动态交叉网络”的研究中取得了成功，该网络参加了一个矩阵，该矩阵代表两个序列中的两个位置同时出现。

- **备用矢量化方案和来源**。尝试在句子和向量之间进行更加智能的映射，或者使用不同的数据集。GloVe提供了高达840亿个不同标记的大型库，每个库有300个尺寸。

### Refrence
> https://www.oreilly.com/ideas/question-answering-with-tensorflow  
> [01. [arxiv.org]Ask Me Anything: Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/abs/1506.07285)