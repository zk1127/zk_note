

## FM(因子分解机)

### FM

#### 背景

- 预估CTR的模型, $CTR=\cfrac{\广告点击数}{曝光数} * 100\%$

- 一种基于矩阵分解的算法,为了解决大规模稀疏矩阵的特征组合问题
- 一般的线性模型是各个特征独立作用,实际上大量特征是关联的；而且高维稀疏矩阵计算量大,特征值更新缓慢
- FM的优点:
  - 特征组合,引入交叉特征,提高模型得分
  - FM是一个通用模型,能够用于任何特征为实值的模型
  - FM的时间复杂度是线性的

#### 推导

- 直接特征组合

  - $y(X)=w_0+\sum^Nw_ix_i+\sum^{N-1}\sum^{N}w_{ij}x_ix_j$
  - 由于矩阵十分稀疏,二次项的系数很难直接训练,每个参数wij的训练都需要大量xi 和xj都非零的样本, 这可能导致$w_{ij}$不够准确,严重影响模型性能

- FM的思想

  - 给每个特征分量$x_i$,引入一个辅助向量$v_i=(v_1,v_2,...,v_k)$

  - 研究证明k足够大,一定存在向量$V \in n \times k$,使得$W = VV^T \in n \times n$,但是对k进行限制可以增加泛化性能

  - 公式推导

  - $$
    \begin{align}
    \sum_{i=1}^{n-1}\sum_{j=i+1}^n <v_i,v_j>x_ix_j
    &= 1/2(\sum_{i=1}^n\sum_{j=1}^n<v_i,v_j>x_ix_j-\sum_{i=1}^n<v_i,v_j>x_ix_i)\\ 
    &=1/2(\sum_{i=1}^n\sum_{j=1}^n\sum_{f=1}^kv_{i,f}v_{i,f}x_ix_j-\sum_{i=1}^n\sum_{f=1}^kv_{i,f}v_{i,f}x_ix_i)\\
    &= 1/2 \sum^k_{f=1}((\sum_{i=1}^nv_{i,f}x_i)(\sum_{j=1}^nv_{j,f}x_j)-\sum_{i=1}^n v_{i,f}^2x_i^2)\\
    &= 1/2 \sum^k_{f=1}((\sum_{i=1}^nv_{i,f}x_i)^2-\sum_{i=1}^n v_{i,f}^2x_i^2)\\
    \end{align}
    $$

  - 从第一步推导到第二步,可以理解为矩阵对角线上的元素之和等于(矩阵所有元素-对角线元素之和)/2

  - 可以发现只要计算$\sum_{i=1}^nv_{i,f}x_i)$就好,复杂度为$O(kn)$

- 梯度计算

  - $\cfrac{\part y(X)}{\part v_{i,f}}=x_j\sum_{i=1}^nv_{i,f}x_i-v_{j,f}x_i^2$

### FFM

**Field Factor Machine**

- 公式
  - $y(X)=w_0+\sum^Nw_ix_i+\sum^{N-1}\sum^{N}(V_{j1,f_2},V_{j2,f1})x_ix_j$

- 简单理解, 比如有三个特征<Publisher, Advertiser, Gender>, 他们的取值分别为2,5,10维, 如果用one-hot表示将会有17维
- 在FM中每个特征只考虑一个隐向量,而FMM中就会考虑多个,还要考虑与特征所在字段的相互关系,这里就会维护三个隐向量

### DeepFM

- FM可以得到高阶特征关系,实际上只使用二阶特征,因为复杂度太高
- deepFM会同时考虑低阶和高阶特征

- 模型概况
- ![img](https://pica.zhimg.com/80/v2-1668503023802fcfd60e0313cd2a84a7_720w.jpg?source=1940ef5c)

- 输入是原始的one-hot向量,按照field进行区分
- 然后进入Embedding层,将特征压缩到有限的向量空间
- 接着进入FM层和DNN层. FM和DNN共用Embedding层
- FM layer 捕捉低阶特征,类似FM和FFM
- DNN捕捉高阶特征,最后预测$y=\sigma(y_{FM}+y_{DNN})$

