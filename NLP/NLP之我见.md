[TOC]



##  梳理

#### 自然语言处理有哪些任务

- 文本到类别
  - 分类
  - 序列标注
- 文本到文本
  - seq2seq

- POS tagging(词性标注)

- Word Segment (中文)

- Parsing

- Coreference Resolution(指代消歧)

- Summarization

- Machine Translation

- Sentiment Classification

- Stance Detection(support, deny, querying,  commenting)

- Nature Language Inference

- Slot Filling

#### BERT-style

##### 词向量说起

- word2vec 和 GloVe都是单词的向量
- fasttext其实是字向量

- 不能解决一词多义的问题

**ELMo和BERT等可以通过语境上下文，给同一个词不同的向量表示，从而解决一词多义问题**

##### 如何减少BERT的参数量

- 网络减层
- 知识蒸馏
- 参数量化
- 网络结构(减小自注意力机制的计算复杂度($O(n^2)$))
  - Reformer
  - Longformer

#### BERT的finetune

- finetunre可以让模型快速收敛

- Adaptor : pre-train部分参数不变，增加一个Adaptor层，只finetune这个部分的参数

- Weighted Feature: 将多层的输出权重化后，进行微调

#### pre-train 

- 自监督学习

##### Predict Next Token:

- LSTM: ELMo,ULMFiT
- Self-Attenttion: GPT, Turing NLG (**MASK Attention**)

##### MASK Input

- MLM: BERT(Token-Level)
- WWM: ERINE(Phrase-Level & Entity-Level)
- SpanBERT(设定一个概率来确定一次能MASK多长的Span，这种在T5的论文也被认为更好)

###### XLNet

- 打乱句子顺序
- 使用Transfomer-XL：在预测MASK时只选择部分Token的注意力来进行预测

###### MASS/BART

- seq2seq式的训练
- 通过decoder还原encoder来解码MASK，对于编码器有多种变化（MASK，Delete，permutation， rotation，text infilling）
- permutation， rotation表现不佳，text infilling表现最好

###### UniLM

- 同时包含encoder，decoder，seq2seq
- 同时进行三种训练，通过attention可视范围的设计

##### Sentence-Level

- Skip-throught: 预测下一句
- Quick-throught：

###### RoBERTa

- SOP： Sentence order predication(ALBERT)
  - ​						