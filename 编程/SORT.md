### SORT

- 冒泡排序

```java
public void swap(int[] nums, int i ,int j){
	int temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}

public void bubbleSort(int[] nums){
    for(int i = 0;i < nums.length;i++){
        for(int j = 0;j < nums.length - i - 1;j++){
            if(nums[j + 1] < nums[j]){
				swap(nums,j + 1,j);
            }
        }
    }
}
```

- 选择排序

```java
public void selectSort(int[] nums){
    for(int i = 0;i < nums.length - 1;i++){
        int t = i;
        for(int j = i + 1;j < nums.length;j++){
            if(nums[j] < nums[t])t = j;
        }
        if(i != t)swap(nums,i,t);
    }
}
```

- 插入排序

```java
public void insertSort(int[] nums){
    for(int i = 1;i < nums.length;i++){
        for(int j = i;j >= 0;j--){
            if(nums[j] < nums[j - 1]) swap(nums, j, j - 1);
        }
    }
}
```

- 归并排序

```java
public void mergeSort(int[] nums){
    int l = 0, r = nums.length - 1;
    int[] copy = new int[nums.length];
    mergeSort(nums,copy,l,r);
}
public void mergeSort(int[] nums,int[] copy,int l,int r){
    if(l < r){
        int m = l + (r - l) / 2;
        mergeSort(nums,copy,l,m);
        mergeSort(nums,copy,m+1,r);
        merge(nums, copy,l ,m , r);
        copy(nums,copy,l,r);
    }
}
public void merge(int[] nums,int[] copy,int l, int m, int r){
    int i = l, j = m + 1,k = l;
    while(i <= m && j <= r){
        if(nums[i] < nums[j]) copy[k++] = nums[i++];
        else copy[k++] = nums[j++];
    }
    while(i <= m) copy[k++] = nums[i++];
    while(j <= r) copy[k++] = nums[j++];
}
```

- 快速排序

```java
public void quickSort(int[] nums,int start,int end){
    if(start < end){
	    int q = partitation(nums,start,end);
        quickSort(nums,start,q - 1);
        quickSort(nums,q + 1, end);
    }
}
public int partitation(int[] nums,int start,int end){
    int left = start + 1,right = end;
    while(true){
        while(left <= right && nums[left] < nums[start]) left++;
        while(right >= left && nums[right] > nums[start]) right--;
        if(right < left) break;
        swap(nums,left,right);
        left++;
        right--;
    }
    swap(nums,start,right);
    return right;
}
```

- 堆排序

```java
public void heapSort(int[] nums){
	PriorityQueue<Integer> pq = new PriorityQueue<>();
    for(int n:nums) pq.add(n);
	int i = 0;
    while(!pq.isEmpty())nums[i++] = pq.poll();
}
```

