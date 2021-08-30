## SVD分解

- 特征值和特征向量
  - $Ax=\lambda x$
  - 其中A是一个$n \times n$的实对称矩阵，*𝑥*是一个n维向量，则我们说*𝜆*是矩阵A的一个特征值，而x是矩阵A的特征值λ所对应的特征向量。

- SVD的定义
  - $A = U\Sigma V^T$
  - U是一个$m \times m$的矩阵, V是一个$n \times n$的矩阵, $\Sigma$是一个$m \times n$的矩阵
  - U和V都是酉矩阵 即 $U^TU=1$
- U矩阵的组成
  - $A^TA$的特征向量
- V矩阵的组成
  - $AA^T$的特征向量

- $\Sigma$对脚线为奇异值,其余取值为0 ,奇异值是$A^TA$的特征值开根号
- $A = U\Sigma V^T$  $\Rightarrow$  $m \times n = m \times m | m \times n | n \times n$

- SVD降维 $A_{m \times n} = U_{m \times m}\Sigma_{m \times n} V^T_{n \times n} \approx U_{m \times k}\Sigma_{k \times k} V^T_{k \times n}$