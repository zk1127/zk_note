## RNN，LSTM，GRU

#### LSTM的公式和理解

- LSTM中的cell以及cell的状态和传递，cell能够保证LSTM中长期信号的存储和传递
- LSTM第二个设计在于 门，门的存在可以选择性的让信息通过或者阻塞，用于保护和控制信息

##### LSTM过程

![img](https://img-blog.csdn.net/20170228165331300?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVycl9feQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- 遗忘门 forget gate：控制上一个cell中信息输入到当前cell的比重
  - $$f_t=\sigma(W_t[h_{t-1},x_t]+b_f)$$
- 输入门 input gate:控制当前时间步的新信息能够输入到当前cell的比重
  - $$i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$$

- 当前时间步汇总的cell信息
  - $$\tilde{c_t}=tanh(W_c[h_{t-1},x_t]+b_c)$$
- 最终当前时间步的cell状态
  - $$c_t=f_t* c_{t-1} + i_t* \tilde{c_t}$$
  - 这里 * 代表矩阵对应位置的数相乘

- 输出门 output gate :控制当前时间步中cell信息输出到$h_t$的比重
  - $$o_t=\sigma(W[h_{t-1},x_t]+b_o)$$
- 最终隐状态的输出
  - $$h_t=o_t * tanh(c_t)$$

##### GRU

![img](https://img-blog.csdn.net/20170509215601173?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVycl9feQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

- 重置门 $r_t$ ,对于之前的隐藏状态的记忆程度
  - $$r_t = \sigma(W_r\cdot[h_{t-1},x_t])$$
- 更新门$z_t$,控制最后的输出
  - $$z_t=\sigma(W_z \cdot[h_{t-1},x_t])$$
- 隐藏状态的更新
  - $$\tilde{h_t}=tanh(W\cdot[r * h_{t-1},x_t])$$
  - $$h_t=(1-z_t)*h_{t-1}+z_t * \tilde{h_t}$$

##### GRU和LSTM的区别

- GRU少了一个门，并且少了cell状态
- 在LSTM中通过输入门和遗忘门来控制上一个状态信息的保留和输入，GRU通过重置门来控制上一个状态信息的输入，但是**不限制当前信息的流入** 
- LSTM中，得到新的cell状态后，需要输出门进行控制:$h_t=o_t * tanh(c_t)$,GRU中得到新的隐藏状态时，需要更新门进行控制：$h_t=(1-z_t)* h_{t-1}+z_t * h_t$

##### RNN的操作实例

1. raw text process

2. tokenize 分词
3. 词典映射
4. padding to fixed length l，每个句子将在RNN中经历 l个时间步
5. mapping token to embedding
6. feed into RNNs as input
7. get ouput 每个时间步可以得到隐状态$h_t$, 整体RNN输出是最后一个时间步的隐状态
8. further processing with the output

#### pytorch中LSTM的输入和输出

- 输入 $$input, (h_0,c_0)$$
  - input: 一个句子,维度为$(seq\_len,batch,input\_size)$
    - seq_len: 句子长度，即token数量，也是时间步
    - batch:传入的句子的数量
    - input_size:token的维度
  - $h_0$: 维度为(num_layers \* num_directions, batch, hidden_size)
  - $c_0$:维度形状为 (num_layers \* num_directions, batch, hidden_size)

- 输出 output, $h_n$,$c_n$
  - output, 维度为(seq_len, batch, num_directions * hidden_size)，LSTM中所有时间步的隐藏状态
  - $h_n$：(num_layers \* num_directions, batch, hidden_size)，所有LSTM层中最后一个时间步的隐藏状态
  - $ c_n$：(num_layers \* num_directions, batch, hidden_size)，所有LSTM层中最后一个时间步的cell状态

##### 为什么LSTM可以缓解RNN的梯度消失问题？

- RNN的梯度消失：在反向传播过程中，用于训练参数的loss信号随着距离的增加会呈指数级减小。这个问题的直接后果就是靠近输入层的一些层的参数很少会更新。对于RNN来说,它就没法捕捉这些长距离信号了
- 一定程度上类似残差连接, 而且这种连接形式是自己网络学习出来的,多条路径的梯度传递能够缓解梯度消失

#### GRU比LSTM好在哪里?

- GRU将遗忘门和输入门组合成一个“更新门”。并将cell state和hidden state合并，整个模型比LSTM简单。
- 参数量减少

#### LSTM的激活函数

- 一个取值为0~1的激活函数更符合门控的物理意义,所以在门控的函数选择sigmoid
- Tanh是一个输出在-1~1的激活函数, 大多数场景下特征分布是0中心的,所以Tanh函数在0附近相比Sigmoid函数有更大的梯度
- 在一些计算有限制的设备上, 直接将门控输出控制为硬输出

#### 在RNN中是否能使用ReLU?

- 因为每层RNN结构的W是重复使用, 所以ReLU会导致更严重的梯度消失和梯度爆炸问题
- 只有初始化权重为单位矩阵时, 才能使用ReLU

#### 