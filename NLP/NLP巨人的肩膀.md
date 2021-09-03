## NLP巨人肩膀

#### 从CV谈起

- transfer learning 的fine-tuning

#### 早期文本统一表示

- one-hot词袋模型
- tf-idf
- textrank
- LSA $\rightarrow$ pLSA $\rightarrow$ LDA

#### 语言模型

- 最基本的认识，建模$P(S_i)$: 语句$S_i$​是一句话的概率，由于文本量的限制，无法对这个概率直接建模。转变为对条件概率进行建模$P(x_0)P(X_1|X_0),...,P(x_T|x_0,x_1,...,x_{T-1})$​
  - 语言模型难以建模，且无法表征词语间的相似性

##### NNLM

- ![img](https://pic1.zhimg.com/80/v2-d5ed23f9bb52b164bef4f4fe3c4244d4_720w.jpg)
- 撇去正则化项，NNLM的极大目标函数对数似然函数，其本质上是个N-Gram的语言模型
- $L=\cfrac{1}{T}\sum_tlogP(w_t|w_{t-1},...,w_{t-n+1};\theta)+R(\theta)$
- NNLM已经是将词表示为分布式向量

##### word2vec

- CBOW：上下文预测核心词
- Skip-gram：核心词预测上下文
- 两个改进：
  - 负采样：如果不进行负采样，在计算softmax概率时候把所有其他词都当作负样例
  - 层次化树：保证词频较大的词处于相对比较浅的层

- GloVe

  - ![img](https://www.zhihu.com/equation?tex=J+%3D+%5Csum_%7Bi%3D1%7D%5EV+%5Csum_%7Bj%3D1%7D%5EV+%5C%3B+f%28X_%7Bij%7D%29+%28+w_i%5ET+w_j+%2B+b_i+%2B+b_j+-+%5Clog+X_%7Bij%7D%29%5E2%5C%5C)

  - ![img](https://pic2.zhimg.com/80/v2-d4de1bb69b83fe7b961d69bb1ffba419_720w.jpg)
  - GloVe对共现概率建模

##### fastText

- ![img](https://pic3.zhimg.com/80/v2-4b580820febc415276358886e7ba812a_720w.jpg)
- 类似CBOW的结构，有标注数据，进行文本分类

##### InferSent

- 在推理数据集上训练模型,将训练好的模型作为特征提取器
- ![img](https://pic4.zhimg.com/80/v2-41b28869cced9362ed7b856d1c9722f7_720w.jpg)_

##### ELMo

- 基本框架仍然是2-stacked biLSTM+Residual的结构

- 区别在于输入是char-based CNN, 可以缓解参数量过大和OOV问题

- ELMo的向量

  - $$
    ELMok^{task}=\gamma^{task}\sum_j=0^l s^{task}_jh_{k,j}
    $$

  - 浅层向量倾向于句法, 高层输出倾向于语义

  - 比较而言,word2vec和GloVe都没有考虑语序的影响

##### GPT

- Transformer架构的核心还是自注意力机制,使用全连接矩阵将输入转变为Q,K,V,通过QK计算得到权值加和到V上.学习了句子中某个词与其他词的语义联系,而多头机制是捕捉了不同侧重的语义联系.
- GPT只使用了Transformer的Decoder来训练语言模型
- GPT的目标函数:$L_3(C)=L_2(C)+\lambda L_1(c)$

##### BERT

- 双向的Transformer架构:自注意力机制天然的双向性质
- MLM语言模型
  - 所有词的15%用于mask, 其中80%选择mask, 10%不做mask, 避免输入不匹配问题, 10%随机替换, 进行文本纠错
- NSP任务
- 学习率的warm-up
- 