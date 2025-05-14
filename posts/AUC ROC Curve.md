---
title: "AUC ROC Curve in Machine Learning - GeeksforGeeks"
source: "https://www.geeksforgeeks.org/auc-roc-curve/"
author:
  - "[[GeeksforGeeks]]"
published: 2020-11-25
created: 2025-02-13
description: "The AUC-ROC curve is a vital metric for evaluating the performance of binary classification models by plotting the True Positive Rate against the False Positive Rate at various thresholds, indicating the model's ability to distinguish between positive and negative classes."
tags:
  - "clippings"
---
In machine learning model evaluation is crucial to ensure that the model performs well. Common evaluation metrics for classification tasks include accuracy, precision, recall, F1-score and AUC-ROC. In this article we’ll focus on the AUC-ROC curve a popular metric used to evaluate classification models.

## How AUC-ROC curve is used?

The ****AUC-ROC curve**** is an essential tool used for evaluating the performance of binary classification models. It plots the ****True Positive Rate (TPR)**** against the ****False Positive Rate (FPR)**** at different thresholds showing how well a model can distinguish between two classes such as positive and negative outcomes.

It provides a graphical representation of the model’s ability to distinguish between two classes like positive class for presence of a disease and negative class for absence of a disease.

Key Terms in AUC-ROC:

- ****TPR (True Positive Rate)****: The ratio of correctly predicted positive instances.
- ****FPR (False Positive Rate)****: The ratio of incorrectly predicted negative instances.
- ****Specificity****: The proportion of actual negatives correctly identified by the model (inverse of FPR).
- ****Sensitivity/Recall****: The proportion of actual positives correctly identified by the model (same as TPR).

![roc4](https://media.geeksforgeeks.org/wp-content/uploads/20250206150722551549/roc4.png)

Sensitivity versus False Positive Rate plot

These terms are derived from the [****confusion matrix****](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/) which provides the following values:

- ****True Positive (TP)****: Correctly predicted positive instances
- ****True Negative (TN)****: Correctly predicted negative instances
- ****False Positive (FP)****: Incorrectly predicted as positive
- ****False Negative (FN)****: Incorrectly predicted as negative

![roc1](https://media.geeksforgeeks.org/wp-content/uploads/20250206150101289181/roc1.png)

Confusion Matrix for a Classification Task

### Understanding ROC and AUC:

- ****ROC Curve****: [ROC Curve](https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/) plots TPR vs. FPR at different thresholds. It represents the trade-off between the sensitivity and specificity of a classifier.
- ****AUC (Area Under the Curve)****: [AUC](https://www.geeksforgeeks.org/how-to-calculate-auc-area-under-curve-in-r/) measures the area under the ROC curve. A higher AUC value indicates better model performance as it suggests a greater ability to distinguish between classes. An AUC value of 1.0 indicates perfect performance while 0.5 suggests it is random guessing.

## How AUC-ROC Works

AUC-ROC curve helps us understand how well a classification model distinguishes between the two classes (positive and negative).

Imagine we have 6 data points and out of these:

- ****3 belong to the positive class:**** Class 1 for people who have a disease.
- ****3 belong to the negative class:**** Class 0 for people who don’t have disease.

![AUC-ROC-Curve](https://media.geeksforgeeks.org/wp-content/uploads/20250206150241961244/AUC-ROC-Curve.webp)

ROC-AUC Classification Evaluation Metric

Now the model will give each data point a predicted probability of belonging to Class 1 (the positive class). The ****AUC**** measures the model’s ability to assign ****higher predicted probabilities to the positive class**** than to the negative class. Here’s how it work:

1. ****Randomly choose a pair****: Pick one data point from the positive class (Class 1) and one from the negative class (Class 0).
2. ****Check if the positive point has a higher predicted probability****: If the model assigns a higher probability to the positive data point than to the negative one for correct ranking.
3. ****Repeat for all pairs****: We do this for all possible pairs of positive and negative examples.

### When to Use AUC-ROC

AUC-ROC is effective when:

- The dataset is balanced and the model needs to be evaluated across all thresholds.
- False positives and false negatives are of similar importance.

> In cases of highly imbalanced datasets AUC-ROC might give overly optimistic results. In such cases the Precision-Recall Curve is more suitable focusing on the positive class.

### Model Performance with AUC-ROC

- ****High AUC (close to 1)****: The model effectively distinguishes between positive and negative instances.
- ****Low AUC (close to 0)****: The model struggles to differentiate between the two classes.
- ****AUC around 0.5****: The model doesn’t learn any meaningful patterns i.e it is doing random guessing.

In short, the ****AUC**** gives you an overall idea of how well your model is doing at sorting positives and negatives, without being affected by the threshold you set for classification. A higher ****AUC**** means your model is doing good.

## Implementation using two different models

#### Installing Libraries

`   ```python3 import numpy as np import pandas as pd import matplotlib.pyplot as plt from sklearn.datasets import make_classification from sklearn.model_selection import train_test_split from sklearn.linear_model import LogisticRegression from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import roc_curve, auc ```    `

In order to train the [Random Forest](https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/) and [Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/) models and to present their ROC curves with AUC scores, the algorithm creates artificial binary classification data.

#### Generating data and splitting data

`   ```python3 X, y = make_classification(     n_samples=1000, n_features=20, n_classes=2, random_state=42)  X_train, X_test, y_train, y_test = train_test_split(     X, y, test_size=0.2, random_state=42) ```    `

Using an 80-20 split ratio, the algorithm creates artificial binary classification data with 20 features, divides it into training and testing sets, and assigns a random seed to ensure reproducibility.

#### Training the different models

`   ```python3 logistic_model = LogisticRegression(random_state=42) logistic_model.fit(X_train, y_train)  random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42) random_forest_model.fit(X_train, y_train) ```    `

Using a fixed random seed to ensure repeatability, the method initializes and trains a logistic regression model on the training set. In a similar manner, it uses the training data and the same random seed to initialize and train a Random Forest model with 100 trees.

#### Predictions

`   ```python3 y_pred_logistic = logistic_model.predict_proba(X_test)[:, 1] y_pred_rf = random_forest_model.predict_proba(X_test)[:, 1] ```    `

Using the test data and a trained Logistic Regression model, the code predicts the positive class’s probability. In a similar manner, using the test data, it uses the trained Random Forest model to produce projected probabilities for the positive class.

#### Creating a dataframe

`   ```python3 test_df = pd.DataFrame(     {'True': y_test, 'Logistic': y_pred_logistic, 'RandomForest': y_pred_rf}) ```    `

Using the test data, the code creates a DataFrame called test\_df with columns labeled “True,” “Logistic,” and “RandomForest,” adding true labels and predicted probabilities from the Random Forest and Logistic Regression models.

#### Plot the ROC Curve for the models

`   ```python3 plt.figure(figsize=(7, 5))  for model in ['Logistic', 'RandomForest']:     fpr, tpr, _ = roc_curve(test_df['True'], test_df[model])     roc_auc = auc(fpr, tpr)     plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')  plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')  plt.xlabel('False Positive Rate') plt.ylabel('True Positive Rate') plt.title('ROC Curves for Two Models') plt.legend() plt.show() ```    `

****Output:****

![roc-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20231206153808/roc.png)

The code generates a plot with 8 by 6 inch figures. It computes the AUC and ROC curve for each model (Random Forest and Logistic Regression), then plots the ROC curve. The ROC curve for random guessing is also represented by a red dashed line, and labels, a title, and a legend are set for visualization.

## ROC-AUC for a multi-class model

For a multi-class setting, we can simply use one vs all methodology and you will have one ROC curve for each class. Let’s say you have four classes A, B, C and D then there would be ROC curves and corresponding AUC values for all the four classes, i.e. once A would be one class and B, C, and D combined would be the others class, similarly, B is one class and A, C, and D combined as others class, etc.

The general steps for using AUC-ROC in the context of a multiclass classification model are:

#### ****One-vs-All Methodology:****

- For each class in your multiclass problem, treat it as the positive class while combining all other classes into the negative class.
- Train the binary classifier for each class against the rest of the classes.

#### Calculate AUC-ROC for Each Class:

- Here we plot the ROC curve for the given class against the rest.
- Plot the ROC curves for each class on the same graph. Each curve represents the discrimination performance of the model for a specific class.
- Examine the AUC scores for each class. A higher AUC score indicates better discrimination for that particular class.

### Implementation of AUC-ROC in Multiclass Classification

#### Importing Libraries

`   ```python3 import numpy as np import matplotlib.pyplot as plt from sklearn.datasets import make_classification from sklearn.model_selection import train_test_split from sklearn.preprocessing import label_binarize from sklearn.multiclass import OneVsRestClassifier from sklearn.linear_model import LogisticRegression from sklearn.ensemble import RandomForestClassifier from sklearn.metrics import roc_curve, auc from itertools import cycle ```    `

The program creates artificial multiclass data, divides it into training and testing sets, and then uses the [One-vs-Restclassifier](https://www.geeksforgeeks.org/one-vs-rest-strategy-for-multi-class-classification/) technique to train classifiers for both Random Forest and Logistic Regression. Lastly, it plots the two models’ multiclass ROC curves to demonstrate how well they discriminate between various classes.

#### Generating Data and splitting

`   ```python3 X, y = make_classification(     n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42)  y_bin = label_binarize(y, classes=np.unique(y))  X_train, X_test, y_train, y_test = train_test_split(     X, y_bin, test_size=0.2, random_state=42) ```    `

Three classes and twenty features make up the synthetic multiclass data produced by the code. After label binarization, the data is divided into training and testing sets in an 80-20 ratio.

#### Training Models

`   ```python3 logistic_model = OneVsRestClassifier(LogisticRegression(random_state=42)) logistic_model.fit(X_train, y_train)  rf_model = OneVsRestClassifier(     RandomForestClassifier(n_estimators=100, random_state=42)) rf_model.fit(X_train, y_train) ```    `

The program trains two multiclass models: a Random Forest model with 100 estimators and a Logistic Regression model with the One-vs-Rest approach. With the training set of data, both models are fitted.

#### Plotting the AUC-ROC Curve

`   ```python3 fpr = dict() tpr = dict() roc_auc = dict()  models = [logistic_model, rf_model]  plt.figure(figsize=(6, 5)) colors = cycle(['aqua', 'darkorange'])  for model, color in zip(models, colors):     for i in range(model.classes_.shape[0]):         fpr[i], tpr[i], _ = roc_curve(             y_test[:, i], model.predict_proba(X_test)[:, i])         roc_auc[i] = auc(fpr[i], tpr[i])         plt.plot(fpr[i], tpr[i], color=color, lw=2,                  label=f'{model.__class__.__name__} - Class {i} (AUC = {roc_auc[i]:.2f})')  plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')  plt.xlabel('False Positive Rate') plt.ylabel('True Positive Rate') plt.title('Multiclass ROC Curve with Logistic Regression and Random Forest') plt.legend(loc="lower right") plt.show() ```    `

****Output:****

![multi-Geeksforgeeks](https://media.geeksforgeeks.org/wp-content/uploads/20231206155921/multi.png)

The Random Forest and Logistic Regression models’ ROC curves and AUC scores are calculated by the code for each class. The multiclass ROC curves are then plotted showing the discrimination performance of each class and featuring a line that represents random guessing. The resulting plot offers a graphic evaluation of the models’ classification performance.

## FAQs for AUC ROC Curve in Machine Learning

****What is the AUC-ROC curve?****

> For various classification thresholds, the trade-off between true positive rate (sensitivity) and false positive rate (specificity) is graphically represented by the AUC-ROC curve.

****What does a perfect AUC-ROC curve look like?****

> An area of 1 on an ideal AUC-ROC curve would mean that the model achieves optimal sensitivity and specificity at all thresholds.

****What does an AUC value of 0.5 signify?****

> AUC of 0.5 indicates that the model’s performance is comparable to that of random chance. It suggests a lack of discriminating ability.

****Can AUC-ROC be used for multiclass classification?****

> AUC-ROC is frequently applied to issues involving binary classification. Variations such as the macro-average or micro-average AUC can be taken into consideration for multiclass classification.

****How is the AUC-ROC curve useful in model evaluation?****

> The ability of a model to discriminate between classes is comprehensively summarized by the AUC-ROC curve. When working with unbalanced datasets, it is especially helpful.

  

**Get IBM Certification** and a **90% fee refund** on completing 90% course in 90 days! [Take the Three 90 Challenge today.](https://www.geeksforgeeks.org/courses/data-science-live?utm_source=geeksforgeeks&utm_medium=bottomtextad&utm_campaign=three90)

Master Machine Learning, Data Science & AI with this complete program and also get a 90% refund. What more motivation do you need? [Start the challenge right away!](https://www.geeksforgeeks.org/courses/data-science-live?utm_source=geeksforgeeks&utm_medium=bottomtextad&utm_campaign=three90)