[TOC]

### 数据预处理

#### pickle循环读取

```python
test_data = []
with open('xx.pkl','rb') as f:
    while True:
        try:
            test_data.extend(pickle.load(f))
        except:
#                 print(i)
            break
```

#### 去除非ascii字符和链接

```python
def remove_non_ascii_chars(text):
    """
    return text after removing non-ascii characters i.e. characters with ascii value >= 128
    """
    return ''.join([w if ord(w) < 128 else ' ' for w in text])


def remove_hyperlinks(text):
    """
    return text after removing hyperlinks
    """
    return ' '.join([w for w in text.split(' ') if not 'http' in w])
```

#### 生成UUID

```python
str(uuid.uuid3(uuid.NAMESPACE_DNS,some_str)).replace("-","")
```



#### 语言测试

```python
def det(x):
    try:
        lang = detect(x)
    except:
        lang = 'Other'
    return lang

# pandas删除非英文行 
tweet_test = tweet_test[df.text.apply(det).eq('en')]
```

#### 按json格式存储dataframe

```python
test_json = test_df.to_json(orient="records",force_ascii=False)
import json
with open('test.json','w+') as f:
    json.dump(test_json,f,ensure_ascii=False)
```

#### pandas 画图代码

```python
ax = test_df.plot(kind='bar', rot='0')
fig = ax.get_figure()
fig.savefig('pretrain.eps',dpi=600,bbox_inches='tight')
```

### 正则表达式

#### 匹配推特hashtag和user_mention

```python
def get_hashtags(text):
    pattern = re.compile(r'\B#(\w*[A-Za-z_]+\w*)') # 由字母数字字符和下划线组成，只考虑了英文
    hahstag_result = pattern.findall(text)
    return hahstag_result

def get_mentions(text):
    pattern = re.compile(r'\B@(\w*[A-Za-z_]+\w*)')
    mention_result = pattern.findall(text)
    return mention_result
```

#### 匹配中文微博hashtag

```python
def get_hashtags(text):
    pattern = re.compile(r'#[^#]+#') # 双#匹配
    hahstag_result = pattern.findall(text)
    return hahstag_result
```

#### 匹配邮件地址

```python
pattern = re.compile(r'[\w]+@[\.\w]+')
```

#### 匹配中文

```python
pattern = re.compile(r'.*?([\u4E00-\u9FA5]+).*?')
```



### 自然语言处理

#### ROUGE计算

```python
def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target)
    metrics['main'] = (
            metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
            metrics['rouge-l'] * 0.4
    )
    return metrics


def compute_main_metric(source, target):
    """计算主要metric
    """
    return compute_metrics(source, target)['main']
```



### 深度学习

#### softmax

```python
def softmax(x, axis=-1):
    """numpy版softmax
    """
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)
```

#### sigmoid

```python
def sigmoid(x):
    try:
        return 1 / (1 + exp(-x))
    except:
        print('SIGMOID ERROR:', x)
        return 0
```



#### 快速比较一个向量与一个向量矩阵的相似度

```python
## pytorch
class CosineSimilarityModel(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityModel, self).__init__()

    def forward(self, sentence, matrix):
        '''
        :param sentense: 某一个句向量
        :param matrix: 总向量矩阵
        :return: s与m中每行相似度结果
        '''
        sentence = sentence.t()
        x = matrix.mm(sentence)

        matrix_frobenins = matrix.norm(dim=1).unsqueeze(0).t()
        sentence_frobenins = sentence.norm(dim=0).unsqueeze(0)
        x_frobenins = matrix_frobenins.mm(sentence_frobenins)

        final = x.mul(1 / x_frobenins)
        return final
```

### Linux命令

#### 随机抽取N行

```python
awk 'BEGIN{srand()} {print rand()"\t"$0}'  input_file | sort -nk 1 | head -n N 
```

#### 文件交并差集

```python
sort a.txt b.txt | uniq -d
sort a.txt b.txt | uniq
sort a.txt b.txt b.txt | uniq -u # a-b 
```

