## Pythorch 处理

- 生成一个batch_size=10，sequence_length=256，embedding_dim=768的随机tensor1？

- ```python
  tensor1 = torch.rand(10, 256, 768) #均匀分布
  tensor1 = torch.randn(10, 256, 768) #高斯分布
  ```

- 如何通过tensor1得到一个维度是（2560，768）的tensor2？

- ```python
  tensor2 = tensor1.reshape(2560, 768)
  tensor2 = tensor1.view(2560, 768)
  ```

- 如何通过tensor2得到一个维度是（768，2560）的tensor3？

- ```python
  tensor3 = tensor2.transpose(0, 1)
  tensor3 = tensor2.permute(0, 1）
  ```

- 如何通过tensor3得到一个维度为（1，768，2560）的tensor4？

- ```python
  tensor4 = torch.unsqueeze(tensor3,0) # 升维
  ```

- 如何通过tensor4的第一维扩展，变为（3，768，2560）的tensor5？

```python
tensor5 = tensor4.expand(3, -1, -1)
```

