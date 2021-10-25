[TOC]

## BERT的后续

#### T5(Transfer Text-to-Text Transformer)

- https://zhuanlan.zhihu.com/p/88438851

- 将所有的NLP任务都转变为Text-to-Text任务
  - 假设需要翻译"That is good"，那么先转换成 "translate English to German：That is good."，类似prompt的思路了
- 选择了encoder-decoder型的预训练结构:对于 Encoder 部分，输入可以看到全体，之后结果输给 Decoder，而 Decoder 因为输出方式只能看到之前的
- **预训练目标**
  - BERT-style
  - replace-span
  - 15%
  - 长度为3的span
- 结论
  - 模型越大越好

#### BERT的缺点

- 训练目标角度：
  - MLM:字级别的MASK，不如Phrase级别的MASK
  - NSP任务作用不大
- 效果角度：
  - NLG任务效果不佳
  - 得到的[CLS]向量作为句向量效果极差
  - 更大的参数，更多的文本量，更多次迭代能得到更好的效果(RoBERTa)
  - 长文章无法得到很好的建模(Transformer-XL)

#### Transformer-XL

- https://www.jianshu.com/p/422a54b18835

- 主要改进点在于transformer的注意力部分，这个部分也被用到XLNet上

- 两个创新点

- **Segment-Level Recurrence**

  - 普通transfomer在进行编码时，会将文档进行分段处理，段与段之间的注意力没有被捕捉。Transformer-XL在训练当前segment时会保存并使用上一个segment的信息，提高捕捉长期依赖能力
  - 训练时，前一个segment的输出，不参与反向传播
  - **Transformer-XL 可以支持的最长依赖近似于 O(NL)**
  - ![img](https://upload-images.jianshu.io/upload_images/20030902-4156f860e807e140.png?imageMogr2/auto-orient/strip|imageView2/2/w/808/format/webp)

- **Relative Positional Encodings**

  - 传统三角函数的位置向量无法区分两个segment位置相同的单词的位置

  - 他在计算embbeding的时候，将传统位置向量都替换为相对位置向量，**R** 使用三角函数公式计算

    ![img](https://upload-images.jianshu.io/upload_images/20030902-170c291368de654e.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

    ![img](https://upload-images.jianshu.io/upload_images/20030902-36850bd1ff172b80.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

#### Distill BERT

- 蒸馏模型，使用6层BERT蒸馏12层BERT，使用KL散度作为损失函数
- ![img](https://upload-images.jianshu.io/upload_images/20030902-e7ff7ec41ff4d60a.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

- DistilBERT 最终的损失函数由 **KL 散度 (蒸馏损失)** 和 **MLM (遮蔽语言建模)** 损失两部分线性组合得到
- DistilBERT 移除了 BERT 模型的 token 类型 embedding 和 NSP (下一句预测任务)，保留了 BERT 的其他机制，然后把 BERT 的层数减少为原来的 1/2。

- 整体上性能也达到了 BERT 的 97%，但是 DistilBERT 的参数量只有 BERT 的 60 %
