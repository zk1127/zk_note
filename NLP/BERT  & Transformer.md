### BERT  & Transformer

##### 为什么BERT会对15%的词执行mask操作，其中80%的词被mask，10%的词被替换，10%的单词保持不变？

##### 优点：

- mask操作是为了语言模型任务，来预测mask的单词
- mask中的替换操作，比如用任意词来替换正确词，相当于文本纠错任务，能够给予BERT模型一定的纠错能力
- mask中的不变操作，缓解fintune时与输入不匹配的问题（预训练时候输入句子当中有mask，而finetune时候输入是完整无缺的句子，即为输入不匹配问题）

##### 缺点：

- 针对两个或者两个以上的字组成的词，mask操作会使的割裂他们之间的语义相关性，使模型不太容易学到词的语义信息

##### 抽象理解 Transformer 中的Query,Key, Value?

比如机器翻译任务中，从中文“我喜欢打篮球 ”$\rightarrow$ "I like playing basketball"

query 是“我喜欢打篮球”， key 是“我， 篮球”  value 就是"I like playing basketball"，通过key捕捉到一些比较关键的字词来保证翻译的性能

