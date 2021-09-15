### Word Vectors

#### SVD Based Methods

- 首先遍历一个很大的数据集和统计词的共现计数矩阵 X，然后对矩阵 X 进行 SVD 分解得到  $USV^{T}$​。然后使用 U 的行来作为字典中所有词的词向量。
  - 两种共现矩阵：**Word-Document Matrix**，**Window based Co-occurrence Matrix**
- 问题
  - 矩阵的维度会经常发生改变（经常增加新的单词和语料库的大小会改变）。
  - 矩阵会非常的稀疏，因为很多词不会共现。
  - 矩阵维度一般会非常高 $≈10^6×10^6$
  - 基于 SVD 的方法的计算复杂度很高 ( m×nm×n 矩阵的计算成本是 O(mn2)O(mn2) )，并且很难合并新单词或文档
  - 需要在 X 上加入一些技巧处理来解决词频的极剧的不平衡

#### Word2vec

- Word2vec输入的是one-hot向量，神经网络训练得到的权重是词向量结果，一个是中心词向量，一个是周围词向量，更容易优化，最后都取平均值

- 两个算法
  - continuous bag-of-words（CBOW）CBOW 是根据中心词周围的上下文单词来预测该词的词向量
  - skip-gram 是根据中心词预测周围上下文的词的概率分布
- 两个训练方法
  - negative sampling 
    - 负采样所采用的指数为 ¾ 的 Unigram 模型
  - hierarchical softmax
    - **hierarchical softmax 对低频词往往表现得更好，负采样对高频词和较低维度向量表现得更好**。

#### GloVe

- window-based (word-word) co-occurrence matrix

- 300是一个很好的词向量维度
- 不对称上下文(只使用单侧的单词)不是很好，但是这在下游任务重可能不同
- window size 设为 8 对 Glove向量来说比较好

#### 词向量的特点

- CBOW只需要一次梯度下降(一个老师教导多个学生),Skip-gram会有n次(多个老师教一个学生),所以CBOW会更快
- 但是Skip-gram 会对低频词迭代更加充分
- GloVe是基于全局共现的词向量模型, word2vec是基于上下文的预测

