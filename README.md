评估算法效率

使用数据集 [spam email](https://www.kaggle.com/code/balaka18/email-spam-classification/data)  [English](./README_ENGLISH.md)

数据集包含5172 行，每行对应一封电子邮件。有3002列，第一列代表邮件名称，最后一列代表是否俄日垃圾邮件，实际上包含3000个email常见单词。使用机器学习算法预测算法的准确度以及性能。

算法使用纯原生方式实现，硬件环境CPU AMD Ryzen 7 5800U

## 开始

因为每次执行都是会将所有数据随机打乱，但实际上准确度不变，这里会进行多次执行，也会对减少训练集内容进行测试准确度

### 第一次执行

| 算法       | 平均准确度 | 耗时                   |
| ---------- | ---------- | ---------------------- |
| K-NN       | 86.750%    | 2:16:26                |
| 朴素贝叶斯 | 70.986%    | 0:00:41.58             |
| 决策树     | -          | 10小时未完成第一折预测 |
| 逻辑回归   | 97.408%    | 0:20:08.54             |

下面是执行的输出

KNN

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 1
Start time: 2022-06-05 12:01:13.592018
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-05 14:17:40.090502
time comsumption: 2:16:26.498484
Scores: [85.88007736943906, 85.78336557059961, 87.9110251450677, 86.46034816247582, 87.71760154738878]
Mean Accuracy: 86.750%
```

Naive Bayes

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 2
Start time: 2022-06-05 12:01:55.528607
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-05 12:02:37.115811
time comsumption: 0:00:41.587204
Scores: [71.47001934235976, 70.50290135396519, 71.08317214700193, 70.6963249516441, 71.1798839458414]
Mean Accuracy: 70.986%
```

Decision Tree

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 3
Start time: 2022-06-05 09:47:29.257484
Start the 1st predicted.
Traceback (most recent call last):

KeyboardInterrupt
```

Logistic Regression

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 4
Start time: 2022-06-07 17:04:54.442080
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-07 17:25:02.989900
time comsumption: 0:20:08.547820
Scores: [97.38878143133462, 97.96905222437138, 97.29206963249516, 97.58220502901354, 96.80851063829788]
Mean Accuracy: 97.408%
```

### 第二次执行

| 算法       | 平均准确度 | 耗时                   |
| ---------- | ---------- | ---------------------- |
| K-NN       | 86.750%    | 2:10:41.08             |
| 朴素贝叶斯 | 71.006%    | 0:00:39.17             |
| 决策树     | -          | 10小时未完成第一折预测 |
| 逻辑回归   | 97.408%    | 0:21:32.28             |

KNN

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 1
Start time: 2022-06-05 09:47:19.848818
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-05 11:58:00.933139
time comsumption: 2:10:41.084321
Scores: [85.88007736943906, 85.78336557059961, 87.9110251450677, 86.46034816247582, 87.71760154738878]
Mean Accuracy: 86.750%
```

Naive Bayes

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 2
Start time: 2022-06-05 09:53:05.812818
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-05 09:53:44.991169
time comsumption: 0:00:39.178351
Scores: [73.11411992263056, 69.53578336557061, 72.24371373307544, 70.11605415860735, 70.01934235976789]
Mean Accuracy: 71.006%
```

Decision Tree

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 3
Start time: 2022-06-05 09:47:29.257484
Start the 1st predicted.
Traceback (most recent call last):

KeyboardInterrupt
```

Logistic Regression

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 4
Start time: 2022-06-07 17:54:10.797715
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-07 18:15:43.081137
time comsumption: 0:21:32.283422
Scores: [97.38878143133462, 97.96905222437138, 97.29206963249516, 97.58220502901354, 96.80851063829788]
Mean Accuracy: 97.408%
```

### 减少数据集进行测试

数据集数量为5000行3000列，这里将会减少为500行进行测试


| 算法       | 平均准确度 | 耗时       |
| ---------- | ---------- | ---------- |
| K-NN       | 85.200%    | 0:01:23.77 |
| 朴素贝叶斯 | 34.200%    | 0:00:03.10 |
| 决策树     | 84.800%    | 0:14:33.56 |
| 逻辑回归   | 97.408%    | 0:20:08.54 |

KNN

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 1
Start time: 2022-06-07 18:00:29.718125
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-07 18:01:53.490808
time comsumption: 0:01:23.772683
Scores: [81.0, 88.0, 86.0, 82.0, 89.0]
Mean Accuracy: 85.200%
```

Naive Bayes

```
Please select algorithms (tips: input int 1-4 to select，other key exit.): 2
Start time: 2022-06-07 18:00:37.838374
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-07 18:00:40.947373
time comsumption: 0:00:03.108999
Scores: [70.0, 24.0, 25.0, 21.0, 31.0]
Mean Accuracy: 34.200%
```

Decision Tree

```
Start time: 2022-06-07 18:00:50.302771
Start the 1st predicted.
Start the 2st predicted.
Start the 3st predicted.
Start the 4st predicted.
Start the 5st predicted.
end time: 2022-06-07 18:15:23.863681
time comsumption: 0:14:33.560910
Scores: [80.0, 82.0, 87.0, 90.0, 85.0]
Mean Accuracy: 84.800%
```

Logistic Regression

```

```