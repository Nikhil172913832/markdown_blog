---
title: Isolation Forest
source: https://medium.com/@corymaklin/isolation-forest-799fceacdda4
author:
  - "[[Cory Maklin]]"
published: 2022-07-15
created: 2025-02-05
description: Isolation Forest is an unsupervised machine learning algorithm for anomaly detection. As the name implies, Isolation Forest is an ensemble method (similar to random forest). In other words, it use…
tags:
  - clippings
  - ml
---


Isolation Forest is an unsupervised machine learning algorithm for anomaly detection. As the name implies, Isolation Forest is an ensemble method (similar to random forest). In other words, it use the average of the predictions by several decision trees when assigning the final anomaly score to a given data point. Unlike other anomaly detection algorithms, which first define what’s “normal” and then report anything else as anomalous, Isolation Forest attempts to isolate anomalous data points from the get go.

## Algorithm

Suppose we had the following data points:

![](https://miro.medium.com/v2/resize:fit:487/1*BjYpEAyEjC2Kmi_3nIQzzw.png)

The isolation forest algorithm selects a random dimension (in this case, the dimension associated with the x axis) and randomly splits the data along that dimension.

![](https://miro.medium.com/v2/resize:fit:451/1*mlF3ZLLvOmW9R3mvyby_DQ.png)

The two resulting subspaces define their own sub tree. In this example, the cut happens to separate a lone point from the remainder of the dataset. The first level of the resulting binary tree consists of two nodes, one which will consist of the subtree of points to the left of the initial cut and the other representing the single point on the right.

![](https://miro.medium.com/v2/resize:fit:385/1*lRN9BXzX15Du5J_LZxwx_A.png)

It’s important to note, the other trees in the ensemble will select different starting splits. In the following example, the first split doesn’t isolate the outlier.

![](https://miro.medium.com/v2/resize:fit:491/1*si2koPwb_h_hy9ZjpyfQOA.png)

We end up with a tree consisting of two nodes, one that contains the points to the left of the line and the other representing the points on the right side of the line.

![](https://miro.medium.com/v2/resize:fit:397/1*QLAPk6Z-ruDMfFJ14TWQMw.png)

The process is repeated until every leaf of the tree represents a single data point from the dataset. In our example, the second iteration manages to isolate the outlier.

![](https://miro.medium.com/v2/resize:fit:494/1*pDM_oRbhFo3omJ2V7vI2Bw.png)

After this step, the tree would look as follows:

![](https://miro.medium.com/v2/resize:fit:501/1*cbdK9oYQudsDajc8DqlBTw.png)

Remember that a split can occur along the other dimension as is the case for this 3rd decision tree.

![](https://miro.medium.com/v2/resize:fit:560/1*vB8rCx1ntfaAh88G_SVTVg.png)

==On average, an anomalous data point is going to be isolated in a bounding box at a smaller tree depth than other points.== When performing inference using a trained Isolation Forest model the final anomaly score is reported as the average across scores reported by each individual decision tree.

![](https://miro.medium.com/v2/resize:fit:700/1*rIX_jEaP1v3EqQPBOF2PrQ.png)

## Categorical Variables

If you’re like me, you’re probably asking yourself how this would work with categorical variables. Assuming that a value that is less observed is anomalous, the Isolation Forest algorithm can make use of categorical variables by representing them as rectangles where the size of rectangle is proportional to the frequency of occurrence.

![](https://miro.medium.com/v2/resize:fit:700/1*YgkxIfVxNYWDf_h4_NETXA.png)

We consider the set of possible values between the middle of the first value and the middle of the last value. We select a random point along the domain then determine the closest edge of a given rectangle. This is used for our split.

![](https://miro.medium.com/v2/resize:fit:700/1*T8THVI-aVrDPvhjZrkeSBw.png)

To ensure fairness, the other trees in the forest will use a different ordering.

![](https://miro.medium.com/v2/resize:fit:700/1*8MuRxNynS9U94_m-D0bS9w.png)

## Python

To start, import the following libraries:

```
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pandas as pd
```

In the proceeding tutorial, we’ll be working with the breast cancer dataset from the UCI machine learning repository. Fortunately, the `scitkit-learn` library provides a wrapper function for downloading the data.

```
breast_cancer = load_breast_cancer()
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
df["benign"] = breast_cancer.target
```

As we can see, the dataset contains 30 numerical features and a target value of 0 and 1 for benign and malignant tumors, respectively.

```
df.head()
```
![](https://miro.medium.com/v2/resize:fit:700/1*pZC8Wn-Rrzd7bcgY5eboCA.png)

For our use case, we will assume that a malignant label is anomalous. The dataset contains a relatively high number of malignant tumors. Thus, we make use of downsampling.

```
majority_df = df[df["benign"] == 1]
minority_df = df[df["benign"] == 0]
minority_downsampled_df = resample(minority_df, replace=True, n_samples=30, random_state=42)
downsampled_df = pd.concat([majority_df, minority_downsampled_df])
```

After downsampling, there are over 10x more samples of the majority class than the minority class.

```
downsampled_df["benign"].value_counts()1    357
0     30
Name: benign, dtype: int64
```

We save the features and target as separate variables.

```
y = downsampled_df["benign"]
X = downsampled_df.drop("benign", axis=1)
```

We set a portion of the total data aside for testing.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

Next, we create an instance of the `IsolationForest` class.

```
model = IsolationForest(random_state=42)
```

We train the model.

```
model.fit(X_train, y_train)
```

We predict the data in the test set.

```
y_pred = model.predict(X_test)
```

The `IsolationForest` assigns a value of `-1` instead of 0. Therefore, we replace it to ensure we only have 2 distinct values in our confusion matrix.

```
y_pred[y_pred == -1] = 0
```

As we can see, the algorithm does a good job of predicting what data points are anomalous.

```
confusion_matrix(y_test, y_pred)array([[ 7,  2],
       [ 5, 83]])
```