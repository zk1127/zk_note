### BERT & Transformer

#### 关于BERT微调

- ```python
  from ray import tune
  from ray.tune import track
  ```

- ```python
  search_space = {
          "batch": tune.grid_search([8,16]),
          "lr": tune.grid_search([1e-5,2e-5,3e-5,4e-5]),
          "lr_type": tune.grid_search(['constant','linear','cosine']),
          'maxlength':tune.grid_search([128,256])
      }
  ```

  

### Transformer

- Transformer的两个显著优势
  - Transformer能够利用分布式GPU进行并行训练，提升模型训练效率.     
  - 在分析预测更长的文本时, 捕捉间隔较长的语义关联效果更好.   
- 架构
- ![img](http://121.199.45.168:8001/img/4.png)

- Transformer总体架构可分为四个部分:
  - 输入部分
    - 源文本嵌入层及其位置编码器
    - 目标文本嵌入层及其位置编码器
  - 输出部分
    - 线性层
    - softmax层
  - 编码器部分
    - 由N个编码器层堆叠而成
    - 每个编码器层由两个子层连接结构组成
    - 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
    - 第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接
  - 解码器部分
    - 由N个解码器层堆叠而成
    - 每个解码器层由三个子层连接结构组成
    - 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
    - 第二个子层连接结构包括一个多头注意力子层和规范化层以及一个残差连接
    - 第三个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接

##### 为什么BERT会对15%的词执行mask操作，其中80%的词被mask，10%的词被替换，10%的单词保持不变？

##### 优点：

- mask操作是为了语言模型任务，来预测mask的单词
- mask中的替换操作，比如用任意词来替换正确词，相当于文本纠错任务，能够给予BERT模型一定的纠错能力
- mask中的不变操作，缓解fintune时与输入不匹配的问题（预训练时候输入句子当中有mask，而finetune时候输入是完整无缺的句子，即为输入不匹配问题）

##### 缺点：

- 针对两个或者两个以上的字组成的词，mask操作会使的割裂他们之间的语义相关性，使模型不太容易学到词的语义信息

##### 抽象理解 Transformer 中的Query,Key, Value?

比如机器翻译任务中，从中文“我喜欢打篮球 ”$\rightarrow$ "I like playing basketball"

query 是“I”， key 是“我,喜欢,打,篮球” value 就是这些词的隐含层向量 ，通过key捕捉到一些比较关键的字词来保证翻译的性能