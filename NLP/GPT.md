[TOC]

### GPT

大规模语料集上的预训练语言模型GPT。目标是few-shot learning，不经过finetune，直接inference

#### （1）Transformer

Transformer里最为核心的机制是Self-attention，正是因为Self-attention的存在，才使得Transformer在做类似翻译问题的时候，可以让其Encoder不用做序列输入，而是将整个序列一次全输入，并且超长序列的输入也变得可能。

Encoder：在Transformer的Encoder中，还有一些其他的设计，比如加入position  embedding（因为Transformer的Encoder中不是时序输入词序列，因此position  embedding也是主要的位置信息）；Residual结构，使得模型的训练过程更为平稳，此外还有normalization层，接着便是feed  forward层（本质上是一个两层的全连接网络，中间加一个ReLu的激活函数）。Decoder的结构与此类似，只不过在进行decode的时候，会将Encoder这边的输出作为Decoder中Self-attention时候的K和V。

Decoder：在矩阵运算过程中，**Decoder中有许多的mask操作，参与运算的三个矩阵Q,K和V都要做许多的mask操作，主要有两方面的作用：一方面是消除输入句子本身长度之外的padding的影响，另一方面是decoder必须要求不能提前看到待生成的词。**除了mask操作，另外，值得注意的是，和Encoder中只有一种类型的Self-attention不同的，**Decoder的attention实际上包含两部分，第一部分是带有mask的Self-attention，通过mask的作用将decode阶段的attention限定只会attention到已经生成过的词上，因此叫做Mask  Self-attention；第二部分是普通的Self-attention操作，不过这个时候的K和V矩阵已经替换为Encoder的输出结果，所以本质上并不是一个Self-attention了。**

#### （2）GPT中是怎么用Transformer的

**GPT中使用的Transformer是只用了Decoder**，因为对于语言模型来讲，确实不需要Encoder的存在。GPT本质上就是用了语言模型的目标函数来优化和训练Transformer-Decoder，这个和上文提到过的语言模型保持一致。利用语言模型的目标函数预训练完成后，**紧接着便可以在具体的任务上进行finetune**。**GPT直接把这两个过程糅合到一个目标函数中**，如

![023](fig\023.png)

其中，$L_2$是task-specific的目标函数，$L_1$则是语言模型的目标函数。**分类问题**中，直接在原序列的开始和末尾添加表示开始和末尾的符号；在**文本蕴含**问题中（比如Natural Language  Inference），将Premise和Hypothesis通过一个中间分隔符“$”连接起来成为一个序列，尔后同样在开头和末尾添加标记符号；在**文本相似**问题中，因为序列1和序列2没有先后关系，因此将先后关系相反的两个序列作为输入；在**智能问答**中，将query和每一个候选的answer都分别连接成一个序列作为输入，最后按各自的打分进行排序。因此，这套输入的表示方法，基本可以使用同一个输入框架来表征许多文本问题（以至于后来的BERT直接借用了这套做法）。除此之外，在输出层，只需要接入一个很简单的全连接层或者MLP便可以，根本不需要非常复杂的模型设计。

正是因为有了输入层和输出层的这种通用化设计考虑，一旦中间的Transformer(当然，正如前文所说，这里的Transformer在使用语言模型进行预训练的时候只有Decoder部分，然而在将其当做文本特征提取器的时候，相应的也可以很便利的将其变成Encoder)表征能力足够强大，迁移学习在NLP任务中的威力也会变得更为强大。

### GPT1、GPT2、GPT3

#### （1）GPT1

GPT底层也基于Transformer模型，与针对翻译任务的Transformer模型不同的是：它只使用了多个Deocder层。

位置编码，基础Transformer使用正余弦函数构造位置信息，位置信息不需要训练相应的参数；而GPT将绝对位置信息作为编码。

GPT-1的训练分为无监督的预训练和有监督的模型微调。

模型参数

- 使用字节对编码（byte pair encoding，BPE），共有40,000个字节对；
- 词编码的长度为768；
- 位置编码也需要学习；
- 12 层的transformer，每个transformer块有12头；
- 位置编码的长度是3,072 ；
- Attention， 残差，Dropout等机制用来进行正则化，drop比例为0.1；
- 激活函数为GLEU；
- 训练的batchsize为64 

#### （2）GPT2

GPT-2旨在训练一个泛化能力更强的词向量模型。GPT-2主要针对zero-shot问题。它在解决多种无监督问题时有很大提升，但是对于有监督学习则差一些。

GPT-2的结构类似于GPT模型，仍然使用单向的Transformer模型，只做了一些**局部修改：如将归一化层移到Block的输入位置；在最后一个自注意力块之后加了一层归一化；增大词汇量**等等。

GPT-2的学习目标是**使用无监督的预训练模型做有监督的任务**。认为任何有监督任务都是语言模型的一个子集，当模型的容量非常大且数据量足够丰富时，仅仅靠训练语言模型的学习便可以完成其他有监督学习的任务。

GPT-2的最大贡献是验证了通过海量数据和大量参数训练出来的词向量模型有迁移到其它类别任务中而不需要额外的训练。

模型参数

- 同样使用了使用字节对编码构建字典，字典的大小为50,257 ；
- 滑动窗口的大小为1024 ；
- batchsize的大小为512；
- **Layer Normalization移动到了每一块的输入部分，在每个self-attention之后额外添加了一个Layer Normalization；**（LN通过把一部分不重要的复杂信息损失掉，以此来降低拟合难度以及过拟合的风险，从而加速了模型的收敛）
- 将残差层的初始化值用$1/\sqrt N$ 进行缩放，其中N 是残差层的个数。

#### （3）GPT3

GPT-3是目前最强大的语言模型，仅仅需要zero-shot或者few-shot，GPT-3就可以在下游任务表现的非常好。除了几个常见的NLP任务，GPT-3还在很多非常困难的任务上也有惊艳的表现，例如撰写人类难以判别的文章，甚至编写SQL查询语句，React或者JavaScript代码等。而这些强大能力的能力则依赖于GPT-3疯狂的1750亿的参数量，45TB的的训练数据以及高达1200万美元的训练费用。

模型

GPT-3沿用了GPT-2的结构，但是在网络容量上做了很大的提升，具体如下：

- GPT-3采用了96 层的多头transformer，头的个数为96 ；
- 词向量的长度是12,888 ；
- 上下文划窗的窗口大小提升至2,048 个token；
- 使用了alternating dense和locally banded sparse attention。

#### （4）总结

总的来说，GPT1,2,3都是 单向transformer decoder结构，训练语言模型，最主要的是训练数据量和模型大小的区别，越来越多，越来越大。

|            | GPT1                                                         | GPT2                                                         | GPT3                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| paper      | Improving Language Understanding by Generative Pre-Training [link](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | Language Models are Unsupervised Multitask Learners [link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Language Models are Few-Shot Learners [link](https://arxiv.org/pdf/2005.14165.pdf) |
| 学习目标   | 无监督语言模型（Pre-training），有监督fine-tune              | 多任务，P(output\|input, task) Zero Short Task Transfer      | few shot                                                     |
| 主要区别   |                                                              | 增加语料、层数、维度 LN前移，最后加LN，初始化scale           | 增加语料、层数、维度                                         |
| Dataset    | 7000 unpublished books，长文较多                             | WebText, 40GB, 8 million documents                           | Common Crawl, WebText2, Books1, Books2 and Wikipedia，共45TB |
| 模型结构   | 12-layer decoder，12 heads，dim 768，ff 3072                 | 48 layers，dim 1600                                          | 96 layers, 96 heads, dim 12888,                              |
| 训练参数   | 100 epochs，batch_size 64，sequence length of 512，lr 2.5e-4，BPE vocab 40,000， | vocab 50,257, batch_size 512, context window 1024            | context 2048, β_1=0.9, β_2=0.95, ε= 10^(-8)                  |
| 模型参数量 | 117M parameters（1.17亿）                                    | 117M (same as GPT-1), 345M, 762M and 1.5B (GPT-2) parameters（15亿） | 175 billion parameters（1750亿）                             |
