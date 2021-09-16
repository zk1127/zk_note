## K-means 

- K-means算法
  - 初始化聚类中心
  - 给聚类中心分配样本
  - 更新聚类中心
  - 停止移动

- 代码

- ```python
  import numpy as np
  
  def k_means(data, k=3, tolerance=0.0001, max_iter=300):
     center = {}
     for i in range(k): #初始化中心
         center[i] = data[k]
  
     for i in range(max_iter):
         clf = {}
  
         for x in data:
             distance = []
  
             for c in center: # 计算与当前中心的距离
                  distance.append(np.sqrt(np.sum((x-center[c])**2)))
  
             cluster = distance.index(min(distance)) # 确认当前数据的中心
             clf[cluster].append(x) 
  
         prev_center = center
         for c in clf:
             center[c] = np.average(clf[c], axis=0) # 更新中心
  
         optimized = True
         for c in center:
             ori_center = prev_center[c]
             cur_center = center[c]
  
             if np.sum(cur_center - ori_center) /ori_center * 100.0 > tolerance: # 判断是否稳定
                 optimized = False
  
             if optimized:
                 break
  ```

  